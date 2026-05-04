# diart_server

> Offline-capable, multi-connection FastAPI WebSocket service that wraps [diart](../../) for real-time speaker diarization. Outputs `(speaker_label, raw embedding)` per prediction; voiceprint matching is left to the caller.

完整设计与协议规范见 [STANDALONE_SERVER.md](../../STANDALONE_SERVER.md)；容器化背景见 [CONTAINERIZATION.md](../../CONTAINERIZATION.md)；in-process 嵌入对照方案见 [MYGPT_INTEGRATION.md](../../MYGPT_INTEGRATION.md)。

---

## 一、Quick start

### 1.1 本地开发

```bash
cd /path/to/diart
pip install -e ".[server]"
uvicorn diart_server.app:app --host 0.0.0.0 --port 7007
```

启动时 `lifespan` 会从 `$DIART_MODELS_DIR`（默认 `/app/models`）加载并 warm-up 全部模型；warm-up 完成前 `/readyz` 返回 503。

开发机上若没有预下载模型，可：
- 提前跑 `python docker/download_models.py --models-dir ./models` 把模型下到本地，再 `export DIART_MODELS_DIR=$PWD/models`
- 或临时 `export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`，让 pyannote 首次启动时联网下载（慢，且需 HF token）

### 1.2 Docker（离线镜像）

```bash
# CPU
docker run --rm -p 7007:7007 ghcr.io/juanmc2005/diart-server:latest-cpu

# GPU
docker run --rm --gpus all -p 7007:7007 ghcr.io/juanmc2005/diart-server:latest-gpu
```

镜像内已 bundle 全部 10 个模型（~470MB），运行期 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`，零网络访问亦可启动。

---

## 二、协议

WebSocket 端点 `/v1/diarize/stream` 接受二进制 PCM 帧，返回 JSON 文本帧。完整协议见 [STANDALONE_SERVER.md §4](../../STANDALONE_SERVER.md#四websocket-协议)。

速查：

- **Client → Server**：二进制帧（PCM float32 LE，单声道，采样率由 `?sample_rate=` query 参数声明，默认 16000）。控制帧用 JSON `{"type":"end"}` 标记流结束。
- **Server → Client**：JSON 一行一条。消息类型：
  - `ready` — 连接成功立即发，含 `session_id` / 模型清单 / 配置
  - `prediction` — 每帧推理结果，含 `speaker` / `start` / `duration` / `embedding`（512 维 raw 向量）/ `samples_count`
  - `final` — `end` 控制帧或断连前发，含 RTTM 文本与统计
  - `error` — 异常，含 `code`（见 [STANDALONE_SERVER.md §4](../../STANDALONE_SERVER.md#四websocket-协议)）+ `message`

---

## 三、配置

全部配置通过环境变量注入，定义在 [`config.py`](config.py)：

| Env var | 默认 | 说明 |
|---|---|---|
| `DIART_PORT` | `7007` | HTTP 端口 |
| `DIART_HOST` | `0.0.0.0` | 绑定地址 |
| `DIART_DEVICE` | `cpu` | 推理设备：`cpu` / `cuda` |
| `DIART_MODELS_DIR` | `/app/models` | bundle 模型目录 |
| `DIART_MAX_CONCURRENT_SESSIONS` | `8` | 并发 WebSocket 会话上限（超出返回 `SESSION_CAPACITY_FULL`） |
| `DIART_MAX_FRAME_BYTES` | `10485760` | 单帧最大字节数（10MB），超出返回 `RATE_LIMIT_EXCEEDED` |
| `DIART_LOG_LEVEL` | `INFO` | logging 级别 |
| `HF_HUB_OFFLINE` | `1`（镜像内） | 禁止 HuggingFace Hub 联网 |
| `TRANSFORMERS_OFFLINE` | `1`（镜像内） | 禁止 transformers 联网 |

---

## 四、探针 / 元信息端点

| 路径 | 用途 |
|---|---|
| `GET /healthz` | 存活探针，进程在线即 200 |
| `GET /readyz` | 就绪探针：模型 warm-up 完成且容量未满返回 200，否则 503 |
| `GET /v1/info` | 服务能力（模型清单、版本、device） |
| `GET /metrics` | Prometheus exposition |

---

## 五、示例

### 5.1 Python 客户端

完整可运行脚本：[`examples/diart_server_client.py`](../../examples/diart_server_client.py)。

最小片段：

```python
import asyncio, json, websockets

async def stream():
    url = "ws://localhost:7007/v1/diarize/stream?sample_rate=16000"
    async with websockets.connect(url) as ws:
        ready = json.loads(await ws.recv())
        print(f"ready: session_id={ready['session_id']}")

        # send binary PCM float32 frames（此处省略文件读取与切片）
        await ws.send(pcm_bytes)
        await ws.send(json.dumps({"type": "end"}))

        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "prediction":
                print(f"speaker={msg['speaker']} embedding[:5]={msg['embedding'][:5]}")
            elif msg["type"] == "final":
                print(msg["rttm"])
                break
            elif msg["type"] == "error":
                raise RuntimeError(msg)

asyncio.run(stream())
```

跑一遍：

```bash
pip install -e ".[server]"          # 包含 websockets
python examples/diart_server_client.py --wav tests/data/audio/sample.wav \
    --host localhost --port 7007
```

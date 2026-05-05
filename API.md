# Diart Server API 参考

> **日期：2026-05-05**
> **服务版本：0.1.0**
> **目标读者**：调用方开发者（mygpt 后端 / 任何下游消费者）
> **配套文档**：[DEPLOY.md](DEPLOY.md)（部署）、[STANDALONE_SERVER.md](STANDALONE_SERVER.md)（设计）

---

## 一、端点总览

| 路径 | 方法 | 用途 |
|---|---|---|
| [`/v1/diarize/stream`](#四websocket-v1diarizestream) | WebSocket | **实时说话人分离 + 嵌入提取（核心）** |
| [`/v1/info`](#二http-v1info) | GET | 服务能力描述 |
| [`/healthz`](#三http-healthz--readyz) | GET | 存活探针 |
| [`/readyz`](#三http-healthz--readyz) | GET | 就绪探针 |
| [`/metrics`](#五http-metrics) | GET | Prometheus 指标 |

**Base URL**：`http://<host>:7007`（HTTP）/ `ws://<host>:7007`（WebSocket）

**鉴权**：本期无鉴权（假定内网部署或外层 ingress 处理）。

---

## 二、HTTP `/v1/info`

服务能力描述，用于客户端发现可用模型 / 设备 / 容量上限。

### 请求

```http
GET /v1/info HTTP/1.1
```

### 响应

`200 OK`

```json
{
  "version": "0.1.0",
  "pipelines": ["diarization", "vad"],
  "models": {
    "pyannote/segmentation": "segmentation",
    "pyannote/segmentation-3.0": "segmentation",
    "pyannote/embedding": "embedding",
    "pyannote/wespeaker-voxceleb-resnet34-LM": "embedding",
    "hbredin/wespeaker-voxceleb-resnet34-LM": "embedding",
    "speechbrain/spkrec-xvect-voxceleb": "embedding",
    "speechbrain/spkrec-ecapa-voxceleb": "embedding",
    "speechbrain/spkrec-ecapa-voxceleb-mel-spec": "embedding",
    "speechbrain/spkrec-resnet-voxceleb": "embedding",
    "nvidia/speakerverification_en_titanet_large": "embedding"
  },
  "device": "cuda",
  "max_concurrent_sessions": 8
}
```

### 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `version` | string | 服务版本 |
| `pipelines` | string[] | 支持的 pipeline 类型（`vad` 待 T22 后续启用） |
| `models` | object | 可用模型 → kind 映射，`segmentation` / `embedding` |
| `device` | string | 推理设备：`cpu` / `cuda` |
| `max_concurrent_sessions` | int | 单容器并发上限（容量满时新连接被拒） |

### 调用示例

```bash
curl -s http://localhost:7007/v1/info | jq
```

---

## 三、HTTP `/healthz` / `/readyz`

K8s 风格的探针。

### `GET /healthz`

进程存活检查。**不**做任何模型推理，毫秒级返回。

```http
GET /healthz HTTP/1.1
```

`200 OK`：

```json
{"status": "ok"}
```

### `GET /readyz`

就绪检查：**模型 warm-up 完成** + **当前活跃连接 < 容量上限**。

`200 OK`（就绪）：

```json
{"status": "ready", "active_sessions": 3.0}
```

`503 Service Unavailable`（未就绪）：

```json
// 未 warm-up：
{"status": "not_ready", "reason": "warm-up incomplete"}

// 容量满：
{"status": "not_ready", "reason": "at capacity", "active_sessions": 8.0}
```

### K8s 探针示例

```yaml
readinessProbe:
  httpGet: { path: /readyz, port: 7007 }
  initialDelaySeconds: 30   # 给 warm-up 留时间
  periodSeconds: 10
livenessProbe:
  httpGet: { path: /healthz, port: 7007 }
  periodSeconds: 30
```

---

## 四、WebSocket `/v1/diarize/stream`

**核心实时端点**。客户端推 PCM 二进制帧，服务端推 JSON 文本帧。

### 4.1 连接

```
ws://<host>:7007/v1/diarize/stream?<query>
```

#### Query 参数

| 参数 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `sample_rate` | 16000 | 8000–48000 | 客户端推流采样率（≠ 16k 时服务端重采样） |
| `pipeline` | `diarization` | `diarization` / `vad` | pipeline 类型（vad 占位，未启用） |
| `segmentation_model` | `pyannote/segmentation-3.0` | 见 `/v1/info` | 分割模型 |
| `embedding_model` | `pyannote/embedding` | 见 `/v1/info` | 嵌入模型 |
| `tau_active` | 0.507 | [0.0, 1.0] | 激活阈值（AMI 调优） |
| `rho_update` | 0.006 | [0.0, 1.0] | 中心更新阈值 |
| `delta_new` | 1.057 | [0.0, 2.0] | 新说话人阈值 |
| `latency` | 0.5 | [0.5, 5.0] | 滑动窗口延迟（秒） |
| `max_speakers` | 10 | [2, 20] | 单 session 内最大说话人数 |
| `duration` | 5.0 | > 0 | 滑动窗口长度（秒，建议保持默认） |
| `step` | 0.5 | [0.1, 1.0] | 滑动步长（秒） |

**所有参数都是字符串形式（query string），服务端做类型转换 + 范围校验**。失败 → 立即关闭连接（见 [§4.5](#45-错误处理)）。

#### 连接 URL 示例

```
ws://diart:7007/v1/diarize/stream?sample_rate=16000
ws://diart:7007/v1/diarize/stream?sample_rate=48000&embedding_model=speechbrain%2Fspkrec-ecapa-voxceleb
ws://diart:7007/v1/diarize/stream?sample_rate=16000&segmentation_model=pyannote%2Fsegmentation&tau_active=0.6
```

模型名里的 `/` 必须 URL-encode 为 `%2F`。

---

### 4.2 客户端 → 服务端

#### 二进制帧（推流）

**格式**：`PCM float32 little-endian, 单声道, 采样率由 query 声明`

无附加 header，纯 PCM 数据。建议每帧 ~500ms（与默认 step 对齐）。

**最大帧**：`DIART_MAX_FRAME_BYTES`（默认 10MB）。超限直接关闭连接，错误码 `RATE_LIMIT_EXCEEDED`。

```python
# Python 示例
import numpy as np
pcm = np.zeros(8000, dtype=np.float32)  # 0.5s @ 16kHz silence
await ws.send(pcm.tobytes())
```

#### 文本控制帧

发送 JSON：

```json
{"type": "end"}
```

效果：触发服务端进入收尾路径（drain → 发 `final` → 关闭连接）。建议在不再有音频时主动发，确保拿到 `final`。

不识别的文本帧（包括语法错误的 JSON）**会被静默忽略**，连接保持。

---

### 4.3 服务端 → 客户端

所有出站消息都是 **JSON 文本帧**，一次一条。

#### 4.3.1 `ready`（连接建立后立即发）

```json
{
  "type": "ready",
  "session_id": "9603a856-9079-40b2-8ac7-539f89bb6387",
  "sample_rate": 16000,
  "pipeline": "diarization",
  "models": {
    "segmentation": "pyannote/segmentation-3.0",
    "embedding": "pyannote/embedding"
  },
  "config": {
    "tau_active": 0.507,
    "rho_update": 0.006,
    "delta_new": 1.057,
    "latency": 0.5,
    "max_speakers": 10,
    "duration": 5.0,
    "step": 0.5
  }
}
```

`session_id` 是本次连接的 UUID，用于 RTTM 关联和日志追踪。

#### 4.3.2 `prediction`（每条说话人 turn 一条）

```json
{
  "type": "prediction",
  "start": 6.675,
  "duration": 0.633,
  "speaker": "speaker0",
  "embedding": [0.0214, -0.0851, ..., 0.0397],
  "samples_count": 4
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `start` | float | 该 turn 起始时间（秒，从连接建立计时） |
| `duration` | float | 该 turn 持续时间（秒） |
| `speaker` | string | 会话内说话人标签，格式 `"speaker0"` / `"speaker1"` ... |
| `embedding` | float[] | **512 维 L2-单位向量**（运行平均后归一化），可直接做余弦相似度 |
| `samples_count` | int | 该 speaker label 自首次出现以来累计的 prediction 计数（≥3 时嵌入趋稳，建议匹配阈值用此） |

**重要**：
- `speaker` 标签**仅在本连接内有效**——不同 session 的 `speaker0` 不一定是同一人。跨 session 身份识别请用 `embedding` 对比。
- `embedding=[]`（空数组）表示聚类还没初始化（音频中无明显语音、或 first-window）。客户端应跳过这种 prediction 的声纹匹配。
- `embedding` 已 L2 归一化（`|emb|=1.0`），下游可直接 dot product 算余弦相似度。

#### 4.3.3 `final`（连接关闭前）

```json
{
  "type": "final",
  "rttm": "SPEAKER 9603a856-... 1 6.675 0.633 <NA> <NA> speaker0 <NA> <NA>\nSPEAKER ...",
  "duration": 10.01,
  "num_speakers": 1,
  "stats": {
    "predictions_count": 7
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `rttm` | string | 标准 RTTM 格式（多行，`\n` 分隔），可直接喂 pyannote.metrics 做评测 |
| `duration` | float | 整段音频时长（秒） |
| `num_speakers` | int | 本次会话识别到的说话人数 |
| `stats.predictions_count` | int | 累计 prediction 消息数 |

`final` 后服务端会主动关闭 WS（close code 1000）。

#### 4.3.4 `error`

```json
{
  "type": "error",
  "code": "BAD_AUDIO",
  "message": "empty frame"
}
```

错误码全集：

| code | 含义 | 是否关闭连接 | close code |
|---|---|---|---|
| `BAD_AUDIO` | 帧为空、采样率超出范围、解码失败 | 视情况（query 错 → 关闭；单帧错 → 保持） | 1008（query）/ — |
| `RESAMPLE_FAILED` | torchaudio.resample 抛错 | 否 | — |
| `MODEL_NOT_FOUND` | query 指定的模型不在 manifest | 是 | 1008 |
| `INFERENCE_CRASH` | 后台推理线程异常 | 是 | 1011 |
| `RATE_LIMIT_EXCEEDED` | 单帧超过 `DIART_MAX_FRAME_BYTES` | 是 | 1009 |
| `SESSION_CAPACITY_FULL` | 服务器活跃连接达上限 | 是（accept 后立即关） | 1013 |
| `INTERNAL_ERROR` | 其他未预期异常 | 是 | 1011 |

---

### 4.4 完整生命周期

```
客户端                                       服务端
   │                                           │
   ├─── WS Connect (ws://.../stream?...) ─────►│
   │                                           │ accept + 解析 query
   │                                           │ 容量检查 + 实例化 session
   │◄───────────────────  {type:"ready",...} ──┤ session.start()
   │                                           │ (背景线程跑 inference)
   │                                           │
   ├──── PCM frame (binary, 0.5s) ─────────────►│
   │                                           │ push 到队列
   │                                           │ ...inference 处理...
   │◄──────  {type:"prediction",speaker:...} ──┤ 0.5s 后发出
   │                                           │
   ├──── PCM frame ────────────────────────────►│
   │◄──────  {type:"prediction",...} ──────────┤
   │              ...                          │
   │                                           │
   ├──── {"type":"end"} (text) ────────────────►│
   │                                           │ drain queue
   │                                           │ session.close()
   │◄───────────  {type:"final",rttm:...} ─────┤
   │◄───────────  WS close (code=1000) ────────┤
```

### 4.5 错误处理

服务端会在以下情况主动关闭连接：

1. **Query 校验失败**（query 类型错或超出范围）：accept 后立即发 `error{BAD_AUDIO}` + close 1008
2. **容量满**：accept 后立即发 `error{SESSION_CAPACITY_FULL}` + close 1013
3. **构造 session 失败**（NotImplementedError vad / ModelNotFoundError）：发对应 error + close 1008
4. **推理线程崩溃**：发 `error{INFERENCE_CRASH}` + close 1011
5. **客户端 disconnect**：服务端走 finally → close → 发 final（如果还能发）+ close

客户端应对 close code 做语义化处理：

| close code | 客户端动作 |
|---|---|
| 1000 | 正常结束 |
| 1008 | 请求格式错（query / model 名）→ 不重试 |
| 1009 | 帧太大 → 修正客户端切片逻辑 |
| 1011 | 服务端内部错 → 指数退避重连 |
| 1013 | 容量满 → 排队 / 路由到其他副本 |

---

## 五、HTTP `/metrics`

Prometheus exposition format。

### 请求

```http
GET /metrics HTTP/1.1
```

### 响应

`200 OK`，`Content-Type: text/plain; version=0.0.4`：

```
# HELP diart_active_sessions Number of active WebSocket sessions
# TYPE diart_active_sessions gauge
diart_active_sessions 3.0

# HELP diart_predictions_total Total predictions emitted
# TYPE diart_predictions_total counter
diart_predictions_total{pipeline="diarization",model="pyannote/embedding"} 142.0

# HELP diart_prediction_latency_seconds End-to-end latency
# TYPE diart_prediction_latency_seconds histogram
diart_prediction_latency_seconds_bucket{le="0.1"} 0.0
diart_prediction_latency_seconds_bucket{le="0.25"} 0.0
diart_prediction_latency_seconds_bucket{le="0.5"} 87.0
...
```

完整指标列表见 [DEPLOY.md §8](DEPLOY.md#八监控接入)。

---

## 六、调用示例

### 6.1 Python（异步 + websockets 库）

```python
"""完整 client 示例。基于 examples/diart_server_client.py 简化。"""
import asyncio, json, wave
import numpy as np
import websockets


async def stream(wav_path: str, host="localhost", port=7007):
    # 读取 WAV → float32 PCM
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    url = f"ws://{host}:{port}/v1/diarize/stream?sample_rate={sr}"
    async with websockets.connect(url) as ws:
        # 1. ready
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready"
        print(f"session_id={ready['session_id']}")

        # 2. 推流（边推边收）
        chunk = sr // 2  # 0.5s
        send_task = asyncio.create_task(_pump(ws, pcm, chunk))

        # 3. 收 predictions + final
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 10.0))
                if msg["type"] == "prediction":
                    print(f"[{msg['speaker']}] {msg['start']:.2f}s + {msg['duration']:.2f}s "
                          f"emb_dim={len(msg['embedding'])} samples={msg['samples_count']}")
                elif msg["type"] == "final":
                    print(f"[final] {msg['num_speakers']} speakers, {msg['duration']:.2f}s")
                    print(msg["rttm"])
                    break
                elif msg["type"] == "error":
                    print(f"[error] {msg['code']}: {msg['message']}")
        finally:
            send_task.cancel()


async def _pump(ws, pcm, chunk):
    for i in range(0, len(pcm), chunk):
        await ws.send(pcm[i:i+chunk].tobytes())
        await asyncio.sleep(len(pcm[i:i+chunk]) / 16000.0)  # real-time pacing
    await ws.send(json.dumps({"type": "end"}))


if __name__ == "__main__":
    asyncio.run(stream("tests/data/audio/sample.wav"))
```

### 6.2 Node.js（ws 库）

```javascript
const WebSocket = require('ws');
const fs = require('fs');

const ws = new WebSocket('ws://localhost:7007/v1/diarize/stream?sample_rate=16000');

ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());
  if (msg.type === 'ready') {
    console.log(`session: ${msg.session_id}`);
    // 推 PCM（这里假设 pcm 是 Float32Array）
    // ws.send(Buffer.from(pcm.buffer));
  } else if (msg.type === 'prediction') {
    console.log(`speaker=${msg.speaker} emb_dim=${msg.embedding.length}`);
  } else if (msg.type === 'final') {
    console.log(msg.rttm);
  }
});

ws.on('close', (code) => console.log(`closed: ${code}`));
```

### 6.3 mygpt 后端集成（伪码）

```python
# mygpt 收到客户端音频后，转发到 diart_server，把 embedding 拿回来匹配声纹库

async def handle_audio_chunk(meeting_id: str, pcm: bytes):
    if not meeting_id in self.diart_connections:
        ws = await websockets.connect(f"ws://diart:7007/v1/diarize/stream?sample_rate=16000")
        ready = json.loads(await ws.recv())
        self.diart_connections[meeting_id] = ws
        # 启动一个后台 task 持续读 prediction
        asyncio.create_task(self._consume(meeting_id, ws))

    await self.diart_connections[meeting_id].send(pcm)


async def _consume(meeting_id: str, ws):
    async for raw in ws:
        msg = json.loads(raw)
        if msg["type"] == "prediction" and msg["embedding"]:
            user_id = await self.voiceprint_store.match(msg["embedding"])  # 余弦匹配
            await self.broadcast_speaker_change(meeting_id, msg["speaker"], user_id, msg["start"])
        elif msg["type"] == "final":
            await self.persist_rttm(meeting_id, msg["rttm"])
            break
```

---

## 七、性能与限制

### 实测延迟（CPU，dummy / 真实模型）

| 场景 | p50 | p95 | p99 |
|---|---|---|---|
| 1 路（pyannote/segmentation-3.0 + embedding，CPU） | ~33ms / chunk | — | < 1s |
| 1 路（pyannote/segmentation v1，CPU） | ~30ms / chunk | — | < 1s |
| 4 路并发 | ~500ms p50 | < 1.5× 单路 | — |

GPU 实测约 5–10× 加速，待 GPU 镜像构建后基线见 [followups.md](followups.md)。

### 限制

- **单容器并发上限 8 路**（`DIART_MAX_CONCURRENT_SESSIONS`）；超限横向扩容
- **单帧上限 10MB**；超限直接关闭连接
- **跨容器不共享 speaker 标签**；声纹匹配需要客户端维护
- **不支持 stereo PCM**；客户端必须 mix-down 到 mono
- **不支持 WebM / Opus**；客户端必须 PCM 化

---

## 八、变更日志

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.1.0 | 2026-05-05 | 首次发布。10 个 bundled 模型，CPU + GPU 双 tag |

---

## 九、相关文档

- 部署手册：[DEPLOY.md](DEPLOY.md)
- 设计与决策：[STANDALONE_SERVER.md](STANDALONE_SERVER.md)
- 客户端示例：[examples/diart_server_client.py](examples/diart_server_client.py)
- 测试用例：[tests/diart_server/](tests/diart_server/)

# Diart Server 部署指南

> **日期：2026-05-05**
> **目标读者**：运维 / DevOps / 首次部署的开发者
> **配套文档**：[API.md](API.md)（接口契约）、[STANDALONE_SERVER.md](STANDALONE_SERVER.md)（设计）

---

## 一、概览

`diart_server` 是一个**完全离线**的 FastAPI WebSocket 服务，用于实时说话人分离 + 嵌入提取。镜像内置全部 10 个支持的模型（合计约 550MB），运行期不需要任何外网访问。

### 镜像两条 tag

| Tag | 基底 | 体积估算 | 运行依赖 |
|---|---|---|---|
| `image:cpu` | `python:3.11-slim` | ~2GB | 普通 docker |
| `image:gpu` | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` | ~5–7GB | nvidia-container-runtime |

### 服务能力

- WebSocket：`/v1/diarize/stream`（多路并发，默认上限 8 路）
- 健康/就绪/信息：`/healthz`、`/readyz`、`/v1/info`
- Prometheus 指标：`/metrics`

---

## 二、前置条件

### 必须

- **Docker 20.10+**（已验证 28.3.3）
- **HuggingFace token**：[创建 token](https://huggingface.co/settings/tokens)
- **接受三个 pyannote 模型协议**（gated repos）：
  - https://huggingface.co/pyannote/segmentation
  - https://huggingface.co/pyannote/segmentation-3.0
  - https://huggingface.co/pyannote/embedding

### GPU 镜像额外需要

- NVIDIA 驱动（已验证 575.57.08）
- CUDA 12.x runtime（镜像自带）
- `nvidia-container-toolkit`：
  ```bash
  # Ubuntu / Debian
  sudo apt install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

---

## 三、构建 CPU 镜像

### 状态：⚠️ 代码已验证、镜像未跑过 build

> 我们用 `docker/download_models.py` 直接下载模型 + 真实 pyannote 模型跑通了端到端流程（[followups.md](followups.md) 的 "Verified" 章节），但 **multi-stage docker build + 镜像内 uvicorn 启动**这一步还需要本机首跑。下面命令逐字可用，但 **build 失败可能性最大的是 stage-2 的 site-packages COPY 路径**。

### 步骤

```bash
cd /path/to/diart

# 1) 把 HF token 写到一个临时文件（用 BuildKit secret 注入更安全）
echo "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > /tmp/hf_token

# 2) 启用 BuildKit + 构建
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.cpu \
  --secret id=hf_token,src=/tmp/hf_token \
  --build-arg HF_TOKEN="$(cat /tmp/hf_token)" \
  -t diart-server:cpu-0.1.0 \
  -t diart-server:cpu-latest \
  .

# 3) 清理 secret 文件
rm /tmp/hf_token

# 4) 检查镜像大小（预期 < 2.5GB）
docker images diart-server:cpu-0.1.0
```

### 可能的坑（提前打预防针）

| 现象 | 原因 | 修复 |
|---|---|---|
| `403 Cannot access gated repo` | 没接受 pyannote 协议 | 上 HF 网页接受三个 gated 模型 |
| `CalledProcessError ... pip install -e .[server]` | python:3.11-slim 缺 `git`/`build-essential` | Dockerfile.cpu 已加，确认 builder stage 完整 |
| Stage-2 `COPY ... site-packages` 找不到目录 | python:3.11-slim 用 `/usr/local/lib/python3.11/site-packages`（已对齐） | 已对齐；如失败，改为 stage-2 重新 `pip install` |
| 镜像 > 3GB | torch wheel 没指定 `+cpu` 后缀，下了 GPU 版 | Dockerfile.cpu 已固定 `torch==2.4.1+cpu` |

### 离线启动 smoke 测试（自动）

```bash
chmod +x docker/smoke_test.sh
bash docker/smoke_test.sh diart-server:cpu-0.1.0
# 预期最后一行: [smoke] OK
```

脚本逻辑：`docker run --network none` 启动容器 → 轮询 `/readyz` → `/v1/info` 至少返回 10 个模型。**用 `--network none` 是关键**——它强制走离线路径，验证 `HF_HUB_OFFLINE=1` 真的生效。

---

## 四、构建 GPU 镜像

### 状态：⚠️ 同上，代码完整、build 未跑

GPU 镜像的额外风险点是 **Ubuntu 22.04 base 的 Python 3.11 site-packages 路径**——可能是 `/usr/local/lib/python3.11/dist-packages`（Debian/Ubuntu 约定）而不是 `site-packages`。Dockerfile.gpu 里有内联注释。

### 步骤

```bash
cd /path/to/diart

echo "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > /tmp/hf_token

DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.gpu \
  --secret id=hf_token,src=/tmp/hf_token \
  --build-arg HF_TOKEN="$(cat /tmp/hf_token)" \
  -t diart-server:gpu-0.1.0 \
  -t diart-server:gpu-latest \
  .

rm /tmp/hf_token

docker images diart-server:gpu-0.1.0
```

### GPU 验证

```bash
# 容器里能看到 GPU + torch 能用 cuda
docker run --rm --gpus all diart-server:gpu-0.1.0 \
  python -c "import torch; print('cuda:', torch.cuda.is_available()); print('devs:', torch.cuda.device_count())"
# 预期: cuda: True / devs: >=1

# 完整离线 smoke
bash docker/smoke_test.sh diart-server:gpu-0.1.0
```

---

## 五、启动容器

### 5.1 单机 / 开发环境

```bash
# CPU
docker run -d \
  --name diart-server \
  -p 7007:7007 \
  -e DIART_MAX_CONCURRENT_SESSIONS=4 \
  diart-server:cpu-latest

# GPU（推荐线上）
docker run -d \
  --name diart-server \
  --gpus all \
  -p 7007:7007 \
  -e DIART_MAX_CONCURRENT_SESSIONS=8 \
  diart-server:gpu-latest

# 验证
curl -sf http://localhost:7007/readyz
# 预期: {"status":"ready","active_sessions":0.0}
```

### 5.2 生产建议（docker-compose 示例）

`docker-compose.yml`：

```yaml
services:
  diart-server:
    image: ghcr.io/your-org/diart-server:gpu-0.1.0
    container_name: diart-server
    restart: unless-stopped
    ports:
      - "7007:7007"
    environment:
      DIART_DEVICE: cuda
      DIART_MAX_CONCURRENT_SESSIONS: "8"
      DIART_LOG_LEVEL: INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:7007/readyz"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"
```

启动：

```bash
docker compose up -d
docker compose logs -f diart-server
```

### 5.3 Kubernetes（GPU pod 模板）

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diart-server
spec:
  replicas: 1
  selector:
    matchLabels: { app: diart-server }
  template:
    metadata:
      labels: { app: diart-server }
    spec:
      containers:
        - name: diart-server
          image: ghcr.io/your-org/diart-server:gpu-0.1.0
          ports: [{ containerPort: 7007 }]
          env:
            - { name: DIART_MAX_CONCURRENT_SESSIONS, value: "8" }
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "8Gi"
              cpu: "2"
          readinessProbe:
            httpGet: { path: /readyz, port: 7007 }
            initialDelaySeconds: 30
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /healthz, port: 7007 }
            periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: diart-server
spec:
  selector: { app: diart-server }
  ports:
    - port: 7007
      targetPort: 7007
```

**关键**：WebSocket 长连接 + 容器内有状态聚类 → 必须 sticky session，不能轮询。Ingress（如 nginx-ingress）需要：

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "diart-session"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
```

---

## 六、配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `DIART_PORT` | 7007 | 监听端口 |
| `DIART_HOST` | 0.0.0.0 | bind 地址 |
| `DIART_DEVICE` | `cpu` 或 `cuda`（镜像默认） | torch device |
| `DIART_MODELS_DIR` | `/app/models` | 模型本地根目录（镜像内已 bake） |
| `DIART_MAX_CONCURRENT_SESSIONS` | 8 | 单容器并发 WS 上限 |
| `DIART_MAX_FRAME_BYTES` | 10485760 | 单个 WS 帧上限（10MB） |
| `DIART_LOG_LEVEL` | INFO | 日志级别 |
| `HF_HUB_OFFLINE` | 1（镜像强制） | 禁止运行期联网 |
| `TRANSFORMERS_OFFLINE` | 1（镜像强制） | 同上 |

**强烈不建议**线上把 `HF_HUB_OFFLINE` 改成 0——会让进程在网络抖动时启动失败。

---

## 七、验证清单

部署后跑一遍：

```bash
HOST=localhost
PORT=7007

# 1. 存活
curl -sf http://$HOST:$PORT/healthz
# 预期: {"status":"ok"}

# 2. 就绪（warm-up 完成 + 容量未满）
curl -sf http://$HOST:$PORT/readyz
# 预期: {"status":"ready","active_sessions":0.0}

# 3. 服务能力
curl -sf http://$HOST:$PORT/v1/info | jq '.models | length'
# 预期: 10

# 4. Prometheus 指标
curl -sf http://$HOST:$PORT/metrics | grep diart_active_sessions
# 预期: # HELP / # TYPE / diart_active_sessions 0.0

# 5. WebSocket 端到端（用 examples/ 里的脚本）
docker exec diart-server python /app/examples/diart_server_client.py --wav /app/tests/data/audio/sample.wav
# 预期: ready → 若干 prediction → final RTTM
```

---

## 八、监控接入

### Prometheus

`/metrics` 暴露 8 个指标，加到 prometheus.yml：

```yaml
scrape_configs:
  - job_name: diart-server
    metrics_path: /metrics
    scrape_interval: 15s
    static_configs:
      - targets: ['diart-server:7007']
```

关键指标（[STANDALONE_SERVER.md §5.9](STANDALONE_SERVER.md)）：

| 指标 | 类型 | 用途 |
|---|---|---|
| `diart_active_sessions` | Gauge | 当前活跃 WS 连接 |
| `diart_predictions_total{pipeline,model}` | Counter | 累计预测数 |
| `diart_prediction_latency_seconds` | Histogram | 端到端推理延迟 |
| `diart_session_duration_seconds` | Histogram | 单连接时长 |
| `diart_session_errors_total{code}` | Counter | 各错误码累计 |
| `diart_queue_overflow_total` | Counter | 背压丢帧次数 |
| `diart_audio_bytes_received_total` | Counter | 累计音频字节 |
| `diart_model_load_seconds{model}` | Gauge | 启动期模型加载耗时 |

### 日志

容器输出结构化 JSON 到 stdout（uvicorn + 自定义 logger）。生产用 `--log-driver json-file` 或接 fluentbit / promtail / Loki。

---

## 九、运维命令速查

```bash
# 查看活跃 session 数
curl -s localhost:7007/metrics | grep '^diart_active_sessions '

# 查看推理延迟分位
curl -s localhost:7007/metrics | grep diart_prediction_latency_seconds_bucket

# 错误统计
curl -s localhost:7007/metrics | grep diart_session_errors_total

# 重启容器（drain 现有 session 后退出，最多 10s）
docker restart diart-server

# 进入容器排查
docker exec -it diart-server bash
ls /app/models                    # 查模型目录
python -c "import diart_server; print(diart_server.__version__)"

# 离线性自检（应该返回 1）
docker exec diart-server printenv HF_HUB_OFFLINE
```

---

## 十、未完成 / 后续

- [ ] 在 CI runner 实际跑 `docker build -f Dockerfile.cpu`（[`.github/workflows/build-images.yml`](.github/workflows/build-images.yml) 已写好，待首次触发）
- [ ] GPU 镜像需要 self-hosted GPU runner，CI 里是 manual `workflow_dispatch` 触发
- [ ] 镜像签名 / SBOM（cosign + syft）
- [ ] 多架构支持（arm64 / amd64）

详细遗留事项见 [followups.md](followups.md)。

---

## 十一、参考

- API 接口契约：[API.md](API.md)
- 设计与决策：[STANDALONE_SERVER.md](STANDALONE_SERVER.md)
- diart 库 README：[README.md](README.md)
- 客户端示例：[examples/diart_server_client.py](examples/diart_server_client.py)

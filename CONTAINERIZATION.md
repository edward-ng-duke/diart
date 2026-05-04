# Diart 容器化与 API 设计分析

> 评估将 diart 封装为容器化服务的可行性，并给出可执行的 API 蓝图。
> 本文档基于对 [src/diart/](src/diart/) 全量代码的探索得出。

---

## 一、Diart 是什么（一句话）

Diart 是一个基于 pyannote.audio + RxPY 的**流式实时说话人日志库**，核心能力：边接收音频（文件 / 麦克风 / WebSocket）边输出"谁在什么时间说了话"的 RTTM 标注，同时支持纯 VAD（语音活动检测）。详见 [README.md](README.md)、[src/diart/__init__.py](src/diart/__init__.py)。

**两条核心 pipeline**：
- `SpeakerDiarization` — 分割 + 嵌入 + 在线聚类
- `VoiceActivityDetection` — 仅 VAD

---

## 二、它是不是"无状态"的？——结论：**部分是，关键有坑**

| 维度 | 状态情况 | 影响 |
|---|---|---|
| **模型权重** | 进程启动时加载、只读、跨请求共享 | ✅ 适合容器化，模型放镜像或 PVC |
| **Pipeline 实例** | **每个 session 内有状态**：聚类中心、滑动窗口缓冲、speaker 计数器（[src/diart/blocks/clustering.py](src/diart/blocks/clustering.py)、[src/diart/blocks/diarization.py](src/diart/blocks/diarization.py)） | ⚠️ 不能跨请求复用同一个 pipeline 实例 |
| **线程安全** | `OnlineSpeakerClustering` 没有任何锁，`centers` / `active_centers` 是可变共享状态 | ⚠️ 同一 pipeline 实例不能被多 session 并发使用 |
| **Speaker ID 一致性** | 仅在**一段连续流内**一致；调用 `reset()` 或新建实例后 speaker 标签会重置 | ⚠️ 跨 session 不能保证"speaker_0 还是同一个人" |
| **WebSocket Server** | 现有 `diart.serve` 明确**只允许一个客户端**（[src/diart/sources.py](src/diart/sources.py) 第 247 行附近） | ⚠️ 不能多路复用 |
| **HF Token** | 启动时需要（除非用 ONNX 本地模型） | ⚠️ 需要 secret 注入；冷启动 30–60s |
| **GPU** | 自动用 CUDA，可 `--cpu` fallback | ✅ 容器需要 GPU runtime（或接受 CPU 慢） |

### 严格意义上的"无状态"是否成立？

**不严格成立**，但可以做到**容器级别可替换**：
- 容器内不持久化任何会话状态（无磁盘写入、无外部缓存依赖）
- 每个 session 的状态在内存中、绑定到一条 WebSocket 连接的生命周期，连接断开即销毁
- 容器本身可以随时被 K8s 替换/扩缩容（前提是用 sticky session 或每条连接绑定到一个 pod）

**真正的无状态**（每个 HTTP 请求独立、状态外置到 Redis）**不适合 diart**：聚类中心是 numpy 数组、每 0.5s 更新一次、数据量小但结构复杂，序列化进 Redis 反而成为瓶颈。**不建议**走这条路。

### 已确定的部署模式

> **多 session 共享容器 + 双镜像 tag**：
> - 一个容器内可同时跑多条 WebSocket 会话，每条连接绑定一个独立 pipeline 实例
> - 模型权重在进程级共享（pyannote 模型 forward 是线程安全的，只有 diart 的 `OnlineSpeakerClustering` 不是，所以聚类逻辑必须每连接独立实例）
> - 镜像构建两个 tag：`diart-api:0.9.2-gpu`（基于 `nvidia/cuda`）和 `diart-api:0.9.2-cpu`（用 ONNX 量化模型）

---

## 三、推荐暴露的 API 列表

按用途分四组。**核心是 WebSocket 流式接口**（diart 本质是流式的），HTTP 用于辅助和离线批处理。

### 组 1：流式实时日志（核心，必做）

#### `WS /v1/diarize/stream`
实时说话人日志的双向 WebSocket。

- **请求查询参数**（连接时）：
  - `pipeline=diarization|vad`（默认 diarization）
  - `sample_rate=16000`
  - `duration=5.0`、`step=0.5`、`latency=0.5`
  - `tau_active=0.5`、`rho_update=0.3`、`delta_new=1.0`
  - `max_speakers=20`
  - `segmentation_model=pyannote/segmentation-3.0`（可选，覆盖默认）
  - `embedding_model=pyannote/embedding`（可选）
- **客户端 → 服务端**：二进制帧（PCM float32 little-endian）或 JSON 控制帧（`{"type":"end"}` 收尾）
- **服务端 → 客户端**：JSON 行
  ```json
  {"type":"prediction","start":1.23,"duration":2.10,"speaker":"speaker_0","confidence":0.87}
  {"type":"rttm","line":"SPEAKER stream 1 1.23 2.10 <NA> <NA> speaker_0 <NA> <NA>"}
  {"type":"final","rttm":"...full RTTM...","duration":35.6}
  ```
- **复用现有代码**：[src/diart/sources.py](src/diart/sources.py) `WebSocketAudioSource`（但需要改造支持多客户端，或在外层用 FastAPI/`websockets` 包一层，每条连接独立创建 pipeline 实例）

#### `WS /v1/vad/stream`
同上但只跑 VAD。可以是 `/v1/diarize/stream?pipeline=vad` 的别名，或独立路径。

---

### 组 2：离线批量处理（同步，必做）

#### `POST /v1/diarize/file`
上传整段音频，**同步**返回完整 RTTM。

- **请求**：`multipart/form-data`（audio 文件字段）+ 表单字段或 query 参数携带 config（`pipeline`、`tau_active`、`max_speakers` 等，与流式接口同名）
- **响应**：
  ```json
  {
    "rttm": "SPEAKER stream 1 0.50 2.70 <NA> <NA> speaker_0 <NA> <NA>\n...",
    "segments": [
      {"start": 0.50, "end": 3.20, "speaker": "speaker_0"},
      {"start": 3.40, "end": 5.10, "speaker": "speaker_1"}
    ],
    "duration": 35.6,
    "num_speakers": 3,
    "processing_time_ms": 4210
  }
  ```
- **复用**：内部走 `FileAudioSource`（[src/diart/sources.py](src/diart/sources.py)）+ `StreamingInference`（[src/diart/inference.py](src/diart/inference.py)）+ `PredictionAccumulator`（[src/diart/sinks.py](src/diart/sinks.py)），最后把 `pyannote.core.Annotation` 转 RTTM 字符串和 segments 数组返回。
- **建议限制**：单文件 ≤ 100MB / ≤ 30 分钟；超长音频应在客户端切块或走流式接口。在 FastAPI 层加请求体大小限制。

---

### 组 3：服务发现与运维（必做）

#### `GET /healthz`
存活检查（pod 是否在跑）。**不**做模型推理，毫秒级返回。

#### `GET /readyz`
就绪检查（模型是否加载完毕、GPU 是否可用）。K8s readiness probe 用。

#### `GET /v1/info`
服务能力描述：
```json
{
  "version": "0.9.2",
  "pipelines": ["diarization", "vad"],
  "default_models": {
    "segmentation": "pyannote/segmentation-3.0",
    "embedding": "pyannote/embedding"
  },
  "device": "cuda:0",
  "supported_sample_rates": [16000],
  "max_concurrent_sessions": 4
}
```

#### `GET /metrics`
Prometheus 格式：活跃 session 数、推理延迟 histogram、GPU 内存、错误计数。运维必备。

---

### 组 4：可选高级端点（按需加）

#### `POST /v1/embed`（可选）
仅暴露说话人嵌入提取（不做聚类）。输入音频，输出 `(N, embedding_dim)` 的向量数组。对接外部声纹库时有用。复用 [src/diart/blocks/embedding.py](src/diart/blocks/embedding.py)。

#### `POST /v1/segment`（可选）
仅暴露说话人分割（不做嵌入和聚类）。输出每个片段的语音/非语音区间。复用 [src/diart/blocks/segmentation.py](src/diart/blocks/segmentation.py)。

> **注：ASR 集成不在本期范围内**。RTTM/segments 输出后由调用方自行用 RTTM 切片送 Whisper/WhisperX 等 ASR 服务。

---

## 四、实现关键点

### 必须解决的问题

1. **多客户端 WebSocket**：现有 `WebSocketAudioSource`（[src/diart/sources.py](src/diart/sources.py)）只支持单客户端。要在外层用 FastAPI/`websockets` 库重写一层 server，每条连接独立创建 pipeline 实例。
2. **每连接独立 pipeline**：每个 WebSocket 连接 `__init__` 时实例化一个新的 `SpeakerDiarization`，连接结束 `del`。模型层（segmentation/embedding 的 `nn.Module`）做成进程级 singleton 共享（pyannote 模型本身的 forward 是线程安全的；只有 diart 的 `OnlineSpeakerClustering` 不是）。
3. **音频协议统一**：现有 `diart.serve` 用 base64 编码 float32（[src/diart/utils.py](src/diart/utils.py)），效率不高。建议新 API 用**二进制 WebSocket 帧 + 头部商定 sample_rate**，省一次 base64 开销。
4. **背压（backpressure）**：客户端推送速率快于推理速率时要丢帧或拒绝。RxPY 默认无背压保护。
5. **HF Token**：启动时通过环境变量 `HF_TOKEN` 注入（参考 [src/diart/console/serve.py](src/diart/console/serve.py)）；或在镜像构建期把模型权重 bake 进去避免冷启动联网。

### 关键文件改动地图

| 新增 | 路径建议 | 内容 |
|---|---|---|
| FastAPI 入口 | `src/diart/server/app.py` | 路由、生命周期、依赖注入；启动时加载共享模型 |
| WebSocket 处理 | `src/diart/server/ws.py` | 每连接一个 pipeline 实例，处理背压 |
| HTTP 文件处理 | `src/diart/server/http.py` | 离线批量端点（同步） |
| Pydantic schema | `src/diart/server/schemas.py` | 请求/响应模型 |
| Dockerfile.gpu | `docker/Dockerfile.gpu` | 基于 `nvidia/cuda:12.1-runtime-ubuntu22.04`，预下载 PyTorch 模型 |
| Dockerfile.cpu | `docker/Dockerfile.cpu` | 基于 `python:3.11-slim`，用 ONNX 量化模型 |
| 入口脚本 | 在 [setup.cfg](setup.cfg) 加 `diart.api` console_script |

### 复用的现有组件

- [src/diart/blocks/diarization.py](src/diart/blocks/diarization.py) `SpeakerDiarization` / `SpeakerDiarizationConfig`
- [src/diart/blocks/vad.py](src/diart/blocks/vad.py) `VoiceActivityDetection`
- [src/diart/inference.py](src/diart/inference.py) `StreamingInference`（编排器）
- [src/diart/sinks.py](src/diart/sinks.py) `PredictionAccumulator`（离线模式收集结果）
- [src/diart/models.py](src/diart/models.py) `PyannoteLoader`（模型加载）

---

## 五、验证方式（实现后如何测试）

1. **单 session 流式**：用 `diart.client microphone --host localhost` 风格的客户端连接 `/v1/diarize/stream`，对比输出 RTTM 与离线 `diart.stream` 的结果一致。
2. **并发流式**：起 4 条 WebSocket 同时推不同音频，验证 speaker_0 标签**不会跨 session 串扰**。
3. **批量**：`curl -F 'audio=@test.wav' POST /v1/diarize/file`，与 `diart.benchmark` 计算的 DER 对齐。
4. **健康检查**：`curl /readyz` 在 GPU 不可用时应返回 503。
5. **背压**：用脚本以 10× 实时速率推帧，验证服务不 OOM。
6. **冷启动**：测量从容器启动到 `/readyz` 返回 200 的时间，预期 30–60s（含模型下载缓存）。

---

## 六、明确不推荐的设计

- ❌ **完全无状态**（每帧 HTTP 调用）：聚类需要历史，往返 Redis 序列化 numpy 数组得不偿失。
- ❌ **直接复用 `diart.serve` + Nginx 多副本**：[src/diart/sources.py](src/diart/sources.py) 单客户端限制会浪费资源。
- ❌ **gRPC 双向流**作为唯一接口：浏览器/curl 不友好；可作为 WebSocket 的备选，不应替代。
- ❌ **暴露 HF token 给客户端**：token 只应在容器启动时注入。
- ❌ **本期集成 ASR**：输出停留在 RTTM/segments，ASR 由调用方对接。

---

## 七、最终 API 清单（速览）

| 方法 | 路径 | 用途 | 必做？ |
|---|---|---|---|
| `WS` | `/v1/diarize/stream` | 实时说话人日志（核心） | ✅ |
| `WS` | `/v1/vad/stream` | 实时 VAD（可与上面合并为参数） | ✅ |
| `POST` | `/v1/diarize/file` | 离线音频文件批量日志（同步） | ✅ |
| `GET` | `/healthz` | 存活探针 | ✅ |
| `GET` | `/readyz` | 就绪探针（含模型/GPU 检查） | ✅ |
| `GET` | `/v1/info` | 服务能力描述 | ✅ |
| `GET` | `/metrics` | Prometheus 指标 | ✅ |
| `POST` | `/v1/embed` | 仅做说话人嵌入提取 | ⚪ 可选 |
| `POST` | `/v1/segment` | 仅做说话人分割 | ⚪ 可选 |

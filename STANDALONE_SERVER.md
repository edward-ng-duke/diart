# Diart 离线 WebSocket 服务：详细开发计划书

> 在 diart 仓库内新增独立的 FastAPI WebSocket 服务（`src/diart_server/`），把 diart 作为库引入，打成**完全离线的双 tag 容器镜像**（cpu / gpu，所有支持的模型 bundled），供下游（如 [mygpt-audio-streaming-frontend](../mygpt-audio-streaming-frontend)）通过 WebSocket 实时调用说话人区分 + 嵌入提取。
>
> 本文档是 [CONTAINERIZATION.md](CONTAINERIZATION.md) 的具体落地，与 [MYGPT_INTEGRATION.md](MYGPT_INTEGRATION.md)（in-process 嵌入方案）并列为 diart 的两条部署路径。

---

## 一、背景与目标

### 1.1 决策背景

| 问题 | 选择 | 理由 |
|---|---|---|
| 部署形态 | **独立容器服务**（非 in-process 嵌入） | 隔离 mygpt 主进程的 GPU 占用；给未来非 Python 后端留扩展口；简化 mygpt 依赖管理 |
| 状态模型 | **有状态**（每连接独立聚类状态） | 拒绝完全无状态（聚类历史不可丢）；不做跨连接共享（语义复杂度过高） |
| 声纹匹配 | **不在容器内做**，只输出 raw embedding | 容器单一职责；声纹库由调用方 MongoDB 维护 |
| 模型分发 | **全部 bundle 进镜像** | 模型总大小 ~470MB 可接受；运行期完全离线，部署环境无需公网 |
| 镜像分发 | **双 tag**（image:cpu / image:gpu） | CPU 部署轻量，GPU 部署性能；运维按环境选 |

### 1.2 目标能力

| 目标 | 验收方式 |
|---|---|
| 独立 FastAPI 容器，接受多路 WebSocket 并发 | 4 路并发推不同音频，speaker label 互相独立 |
| 全模型 bundled，运行期零网络访问 | `docker run --network none` 启动正常，`/readyz=200` |
| 双 tag 镜像 | CI 同步产出 `:cpu` 和 `:gpu`；体积 < 2.5GB / < 8GB |
| 输出 `(speaker_label, raw embedding)` | WebSocket 消息含 512 维嵌入向量 |
| 端到端 p99 延迟 < 1s | push_pcm → 收到 prediction 实测 |

### 1.3 范围（in / out）

| In | Out |
|---|---|
| 新建 `src/diart_server/`（diart 库本身零改动） | 修改 diart 自身代码（如 `OnlineSpeakerClustering`） |
| `Dockerfile.cpu` / `Dockerfile.gpu` + 模型预下载脚本 | K8s / Helm Chart（用户自行维护） |
| WebSocket 协议 + Pydantic schema | gRPC / HTTP REST 接口 |
| 多模型切换（query 参数） | 集成新模型支持 |
| Prometheus metrics + 健康探针 | 鉴权、TLS（假定内网或外层 ingress 处理） |
| 离线启动 CI 验证 | 镜像签名 / SBOM |

---

## 二、架构

### 2.1 系统拓扑

```
                    ┌───────────────────────────────────────────────┐
                    │ Container (image:cpu | image:gpu)            │
                    │                                               │
  Client(s) ──WS───►│ FastAPI + Starlette (asyncio main loop)      │
  (mygpt 后端)      │   ├── /v1/diarize/stream  (WebSocket)        │
                    │   ├── /v1/info /healthz /readyz              │
                    │   └── /metrics  (Prometheus)                 │
                    │           │                                   │
                    │           ▼                                   │
                    │   sessions: dict[id, DiarizationSession]     │
                    │           │                                   │
                    │           ▼                                   │
                    │   每连接：daemon 线程 跑 StreamingInference │
                    │           │                                   │
                    │           ▼                                   │
                    │   diart 库（pip / submodule）                │
                    │   ├── SpeakerDiarization (per-connection)    │
                    │   ├── MeetingAudioSource (thread-safe queue) │
                    │   └── models_registry (process-level)        │
                    │           │                                   │
                    │           ▼                                   │
                    │   /app/models/ (build 期预下载)              │
                    │     pyannote/{segmentation, segmentation-3.0,│
                    │              embedding, wespeaker-...}       │
                    │     hbredin/wespeaker-...-LM (ONNX)          │
                    │     speechbrain/{xvect, ecapa-*, resnet}     │
                    │     nvidia/speakerverification_titanet       │
                    └───────────────────────────────────────────────┘
```

### 2.2 关键不变量

1. **模型权重进程级共享** —— pyannote / SpeechBrain 的 `nn.Module.forward` 是线程安全的，多连接共用一个 `SegmentationModel` / `EmbeddingModel` 实例
2. **聚类状态每连接独立** —— [`OnlineSpeakerClustering`](src/diart/blocks/clustering.py) 非线程安全，每连接持有独立实例
3. **离线**：`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` 强制 transformers / HuggingFace Hub 不联网；模型从本地路径加载
4. **冷启动收敛**：启动期同步 warm-up 全部模型，readyz 才返回 200

### 2.3 线程与异步模型

| 线程 | 数量 | 角色 |
|---|---|---|
| asyncio main | 1 | uvicorn 主循环、WebSocket I/O、handler 协程 |
| inference daemon | N（每连接 1 个） | 阻塞跑 `StreamingInference()`，消费 `MeetingAudioSource` 内部 queue |

**反向通道**：inference daemon 在 `attach_hooks` 回调里通过 `asyncio.run_coroutine_threadsafe` 把 prediction 投到 main loop 的 `asyncio.Queue`，handler 协程消费后 send 给 WebSocket。

### 2.4 资源测算（4 路并发）

| 资源 | 用量 |
|---|---|
| 线程数 | 1 + 4 = 5 |
| 模型权重 | ~470MB（共享） |
| 聚类状态 | 4 × ~30KB = 120KB |
| 5s 滑动窗口 | 4 × ~320KB = 1.3MB |
| GPU 推理峰值显存 | ~1.5–2GB |
| CPU 推理 RAM | ~3GB |
| WebSocket 出带宽 | ~4KB/s × 4 = ~16KB/s |

---

## 三、项目布局

```
src/diart_server/
  __init__.py
  app.py                    # FastAPI 应用 + lifespan
  ws_routes.py              # /v1/diarize/stream + 探针 + info
  session.py                # DiarizationSession（每连接一个）
  meeting_audio_source.py   # 自定义 AudioSource（thread-safe queue）
  audio_decoder.py          # 重采样 + stereo→mono
  protocol.py               # Pydantic 消息 schema
  models_registry.py        # 模型 singleton + 离线路径解析
  config.py                 # env 配置
  metrics.py                # Prometheus 指标
  errors.py                 # 错误码 + 异常类

docker/
  Dockerfile.cpu
  Dockerfile.gpu
  download_models.py        # build 期一次性下载所有模型到 /app/models
  models.json               # 模型清单（HF 名 → 本地路径 + 类型）
  entrypoint.sh
  smoke_test.sh             # 离线启动验证脚本

tests/diart_server/
  test_session.py
  test_meeting_audio_source.py
  test_audio_decoder.py
  test_protocol.py
  test_e2e_websocket.py
  test_load_4_concurrent.py
  data/
    2speakers_30s.wav
    1speaker_60s.wav

setup.cfg                   # 修改：加 [options.extras_require] server / cpu / gpu
.github/workflows/
  build-images.yml          # 双 tag CI
```

### 文件职责详述

| 文件 | 行数预估 | 核心职责 |
|---|---|---|
| `app.py` | ~80 | FastAPI 实例 + lifespan（启动 warm-up，关闭 graceful drain） |
| `ws_routes.py` | ~150 | WebSocket handler、查询参数解析、探针端点 |
| `session.py` | ~200 | `DiarizationSession` 类：lifecycle、push_pcm、attach_hooks 桥接到 asyncio queue |
| `meeting_audio_source.py` | ~80 | `MeetingAudioSource` 类（继承 `AudioSource`），thread-safe queue，背压 |
| `audio_decoder.py` | ~60 | `torchaudio.functional.resample` + stereo→mono 平均 |
| `protocol.py` | ~120 | 所有消息的 Pydantic schema |
| `models_registry.py` | ~150 | 模型 singleton + 离线路径解析 + warm-up |
| `config.py` | ~50 | pydantic-settings 读 env |
| `metrics.py` | ~80 | prometheus_client 指标定义 |
| `errors.py` | ~40 | 错误码枚举 + 自定义异常 |
| `download_models.py` | ~120 | 读 models.json，循环 hf_hub_download / from_pretrained 到本地路径 |

---

## 四、WebSocket 协议

### 4.1 端点

| 路径 | 方法 | 用途 |
|---|---|---|
| `/v1/diarize/stream` | WebSocket | 主流式端点 |
| `/v1/info` | GET | 服务能力（模型清单、版本、device） |
| `/healthz` | GET | 存活探针（毫秒返回） |
| `/readyz` | GET | 就绪探针（验证模型加载、容量未满） |
| `/metrics` | GET | Prometheus |

### 4.2 流式协议详情

#### 连接 query 参数

| 参数 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `sample_rate` | 16000 | 8000–48000 | 客户端推流采样率（≠ 16k 时服务端重采样） |
| `pipeline` | `diarization` | `diarization` / `vad` | 管线类型 |
| `segmentation_model` | `pyannote/segmentation-3.0` | 见六.1 模型清单 | |
| `embedding_model` | `pyannote/embedding` | 见六.1 模型清单 | 仅 diarization 用 |
| `tau_active` | 0.507 | 0–1 | 激活阈值（AMI 调优） |
| `rho_update` | 0.006 | 0–1 | 中心更新阈值 |
| `delta_new` | 1.057 | 0–2 | 新说话人阈值 |
| `latency` | 0.5 | 0.5–5.0 | 延迟（秒） |
| `max_speakers` | 10 | 2–20 | 上限 |
| `duration` | 5.0 | 固定 | 滑动窗口长度（建议不改） |
| `step` | 0.5 | 0.1–1.0 | 步长 |

#### Client → Server

**二进制帧**（默认）：纯 PCM float32 little-endian，单声道，采样率由 query 声明。每帧大小不限，建议 ~500ms。

**控制帧**（罕见）：JSON
```json
{"type":"end"}    // 流结束（不关闭连接），等服务端把缓冲跑完后发 final，由服务端关闭
```

#### Server → Client

所有消息都是 JSON 文本帧，一行一条。

**`ready`**（连接成功立即发）：
```json
{
  "type": "ready",
  "session_id": "uuid-...",
  "sample_rate": 16000,
  "pipeline": "diarization",
  "models": {"segmentation": "pyannote/segmentation-3.0", "embedding": "pyannote/embedding"},
  "config": {"tau_active": 0.507, "rho_update": 0.006, "delta_new": 1.057, "latency": 0.5, "max_speakers": 10}
}
```

**`prediction`**（diart 每输出一帧）：
```json
{
  "type": "prediction",
  "start": 1.23,
  "duration": 0.50,
  "speaker": "speaker_0",
  "embedding": [0.123, ..., 0.456],
  "samples_count": 6
}
```

`embedding`：当前帧抽出的 raw 嵌入（512 floats），客户端可直接用于声纹匹配。
`samples_count`：该 speaker label 自首次出现以来累计 prediction 数（客户端可用此判断是否积累足够时长再做匹配）。

**`final`**（end 控制帧或连接关闭前）：
```json
{
  "type": "final",
  "rttm": "SPEAKER stream 1 1.23 2.10 <NA> <NA> speaker_0 <NA> <NA>\n...",
  "duration": 35.6,
  "num_speakers": 3,
  "stats": {"predictions_count": 71, "audio_seconds": 35.6}
}
```

**`error`**（异常）：
```json
{"type":"error","code":"BAD_AUDIO","message":"sample rate 22050 not in [8000, 48000]"}
```

错误码：

| code | 含义 |
|---|---|
| `BAD_AUDIO` | 音频帧大小为 0、采样率不支持、解码失败 |
| `MODEL_NOT_FOUND` | query 指定的模型未 bundle |
| `RESAMPLE_FAILED` | 重采样异常 |
| `INFERENCE_CRASH` | inference 线程内部异常（连接将关闭） |
| `RATE_LIMIT_EXCEEDED` | 单帧 > 10MB 或推流速率过快 |
| `SESSION_CAPACITY_FULL` | 服务器活跃连接达上限 |
| `INTERNAL_ERROR` | 其他未预期异常 |

---

## 五、模块详细规格

### 5.1 `app.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    config = load_config()
    models_registry.warm_up(config.models_dir, config.device)
    metrics.bootstrap()
    yield
    # shutdown
    await session_manager.close_all(timeout=10)

app = FastAPI(title="diart-server", lifespan=lifespan)
app.include_router(ws_router)
app.include_router(probe_router)
```

### 5.2 `ws_routes.py`

```python
@ws_router.websocket("/v1/diarize/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        config = parse_query(websocket.query_params)
    except QueryValidationError as e:
        await websocket.send_json({"type":"error","code":"BAD_AUDIO","message":str(e)})
        await websocket.close(code=1008); return

    if not session_manager.has_capacity():
        await websocket.send_json({"type":"error","code":"SESSION_CAPACITY_FULL", ...})
        await websocket.close(code=1013); return

    session = DiarizationSession(config, on_message=websocket.send_json)
    session.start()
    await websocket.send_json(session.ready_message())

    try:
        async for frame in websocket.iter_bytes():
            pcm = audio_decoder.to_pcm_16k_mono(frame, src_sr=config.sample_rate)
            session.push_pcm(pcm)
    except WebSocketDisconnect:
        pass
    finally:
        final = await session.close()
        try: await websocket.send_json(final)
        except: pass
```

关键：用 `iter_bytes()` 而非 `iter_text()`，因为我们走二进制协议。`final` 在 close 后从 `PredictionAccumulator` 拿完整 RTTM。

### 5.3 `session.py` - DiarizationSession

```python
class DiarizationSession:
    def __init__(self, config: SessionConfig, on_message: Callable[[dict], Awaitable[None]]):
        self.id = str(uuid.uuid4())
        self.loop = asyncio.get_event_loop()
        self.on_message = on_message

        seg = models_registry.get_segmentation(config.segmentation_model)
        emb = models_registry.get_embedding(config.embedding_model)
        diart_config = SpeakerDiarizationConfig(
            segmentation=seg, embedding=emb,
            duration=5.0, step=config.step, latency=config.latency,
            tau_active=config.tau_active, rho_update=config.rho_update,
            delta_new=config.delta_new, max_speakers=config.max_speakers,
            sample_rate=16000,
            device=torch.device(settings.device),
        )
        self.pipeline = SpeakerDiarization(diart_config)
        self.source = MeetingAudioSource(uri=self.id)
        self.inference = StreamingInference(self.pipeline, self.source, do_plot=False, show_progress=False)
        self.inference.attach_hooks(self._on_prediction)
        self.accumulator = PredictionAccumulator(self.id)
        self.inference.attach_observers(self.accumulator)
        self._speaker_counts: dict[str, int] = defaultdict(int)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        session_manager.register(self)
        self._thread.start()

    def push_pcm(self, pcm: np.ndarray):
        self.source.push_pcm(pcm)

    def _run(self):
        try: self.inference()
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                self.on_message({"type":"error","code":"INFERENCE_CRASH","message":str(e)}),
                self.loop)

    def _on_prediction(self, ann_wav):
        annotation, waveform = ann_wav
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            self._speaker_counts[speaker] += 1
            embedding = self._extract_embedding_for(speaker)  # 从聚类中心或 OverlapAware 抽
            msg = {
                "type": "prediction",
                "start": turn.start, "duration": turn.duration,
                "speaker": speaker,
                "embedding": embedding.tolist(),
                "samples_count": self._speaker_counts[speaker],
            }
            asyncio.run_coroutine_threadsafe(self.on_message(msg), self.loop)

    async def close(self) -> dict:
        self.source.close()
        await asyncio.get_event_loop().run_in_executor(None, self._thread.join, 5.0)
        session_manager.unregister(self.id)
        prediction = self.accumulator.get_prediction()
        return {
            "type": "final",
            "rttm": prediction.to_rttm(),
            "duration": prediction.get_timeline().extent().end if prediction else 0,
            "num_speakers": len(prediction.labels()),
            "stats": {"predictions_count": sum(self._speaker_counts.values())},
        }
```

**embedding 提取**：从 [`OnlineSpeakerClustering.centers`](src/diart/blocks/clustering.py) 取该 speaker label 对应的聚类中心；这是经过累积平均的稳定向量，质量优于单帧 raw embedding。

### 5.4 `meeting_audio_source.py`

```python
class MeetingAudioSource(AudioSource):
    def __init__(self, uri: str, sample_rate=16000, maxsize=50):
        super().__init__(uri, sample_rate)
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = False

    def push_pcm(self, pcm: np.ndarray) -> None:
        try: self._queue.put_nowait(pcm)
        except queue.Full:
            metrics.queue_overflow_total.inc()
            try: self._queue.get_nowait()
            except queue.Empty: pass
            self._queue.put_nowait(pcm)

    def read(self) -> None:  # 由 inference 线程调
        while not self._closed:
            pcm = self._queue.get()
            if pcm is None: break
            tensor = torch.from_numpy(pcm).unsqueeze(0)
            self.stream.on_next(tensor)
        self.stream.on_completed()

    def close(self) -> None:
        self._closed = True
        self._queue.put(None)
```

### 5.5 `models_registry.py`

```python
@dataclass
class ModelSpec:
    hf_name: str
    local_path: Path
    kind: Literal["segmentation", "embedding"]
    backend: Literal["pyannote", "onnx", "speechbrain", "nemo"]

class ModelsRegistry:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.specs: dict[str, ModelSpec] = self._load_manifest()
        self._segmentations: dict[str, SegmentationModel] = {}
        self._embeddings: dict[str, EmbeddingModel] = {}

    def warm_up(self, device: torch.device):
        # 启动期同步加载 default seg + emb；其他模型 lazy
        self.get_segmentation(DEFAULT_SEG).to(device).eval()
        self.get_embedding(DEFAULT_EMB).to(device).eval()

    def get_segmentation(self, name: str) -> SegmentationModel:
        if name not in self._segmentations:
            spec = self.specs.get(name)
            if not spec or spec.kind != "segmentation":
                raise ModelNotFoundError(name)
            if spec.backend == "onnx":
                self._segmentations[name] = SegmentationModel.from_onnx(spec.local_path)
            else:
                self._segmentations[name] = SegmentationModel.from_pyannote(str(spec.local_path), use_hf_token=False)
        return self._segmentations[name]
    # 同理 get_embedding
```

**离线路径解析**：`pyannote.audio.Model.from_pretrained(path)` 支持本地路径；同时设置 `HF_HUB_OFFLINE=1` 防止任何 fallback 联网。

### 5.6 `audio_decoder.py`

```python
def to_pcm_16k_mono(frame_bytes: bytes, src_sr: int) -> np.ndarray:
    pcm = np.frombuffer(frame_bytes, dtype=np.float32)
    if pcm.size == 0:
        raise BadAudioError("empty frame")
    if src_sr != 16000:
        tensor = torch.from_numpy(pcm).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, src_sr, 16000)
        pcm = tensor.squeeze(0).numpy()
    return pcm.astype(np.float32)
```

注：本协议假定客户端推 mono；如要支持 stereo，需要 query 加 `channels=2`。本期只支持 mono。

### 5.7 `protocol.py`

```python
class ReadyMessage(BaseModel):
    type: Literal["ready"]
    session_id: str
    sample_rate: int
    pipeline: Literal["diarization", "vad"]
    models: dict[str, str]
    config: dict

class PredictionMessage(BaseModel):
    type: Literal["prediction"]
    start: float
    duration: float
    speaker: str
    embedding: list[float]
    samples_count: int

class FinalMessage(BaseModel):
    type: Literal["final"]
    rttm: str
    duration: float
    num_speakers: int
    stats: dict

class ErrorMessage(BaseModel):
    type: Literal["error"]
    code: ErrorCode
    message: str

class EndControl(BaseModel):
    type: Literal["end"]
```

### 5.8 `config.py`

| Env 变量 | 默认 | 说明 |
|---|---|---|
| `DIART_PORT` | 7007 | 监听端口 |
| `DIART_HOST` | 0.0.0.0 | bind 地址 |
| `DIART_DEVICE` | `cpu` 或 `cuda`（镜像决定） | torch device |
| `DIART_MODELS_DIR` | `/app/models` | 模型本地根目录 |
| `DIART_MAX_CONCURRENT_SESSIONS` | 8 | 单容器并发上限（4 路推荐，留 2× 余量） |
| `DIART_MAX_FRAME_BYTES` | 10485760 | 单帧上限（10MB） |
| `DIART_LOG_LEVEL` | `INFO` | |
| `HF_HUB_OFFLINE` | `1` | （强制设置，禁止覆盖） |
| `TRANSFORMERS_OFFLINE` | `1` | （强制设置） |

### 5.9 `metrics.py`

| 指标 | 类型 | label | 说明 |
|---|---|---|---|
| `diart_active_sessions` | Gauge | — | 当前活跃连接数 |
| `diart_predictions_total` | Counter | pipeline, model | 累计预测数 |
| `diart_prediction_latency_seconds` | Histogram | — | push_pcm → callback 延迟 |
| `diart_audio_bytes_received_total` | Counter | — | 累计接收音频字节 |
| `diart_session_duration_seconds` | Histogram | — | 单连接时长 |
| `diart_model_load_seconds` | Gauge | model | warm-up 耗时 |
| `diart_session_errors_total` | Counter | code | 错误计数 |
| `diart_queue_overflow_total` | Counter | — | 队列丢帧次数 |

---

## 六、镜像构建

### 6.1 模型清单（docker/models.json）

```json
[
  {"hf_name": "pyannote/segmentation",        "local_path": "pyannote/segmentation",        "kind": "segmentation", "backend": "pyannote"},
  {"hf_name": "pyannote/segmentation-3.0",    "local_path": "pyannote/segmentation-3.0",    "kind": "segmentation", "backend": "pyannote"},
  {"hf_name": "pyannote/embedding",           "local_path": "pyannote/embedding",           "kind": "embedding",    "backend": "pyannote"},
  {"hf_name": "pyannote/wespeaker-voxceleb-resnet34-LM",   "local_path": "pyannote/wespeaker", "kind": "embedding", "backend": "pyannote"},
  {"hf_name": "hbredin/wespeaker-voxceleb-resnet34-LM",    "local_path": "onnx/wespeaker.onnx",  "kind": "embedding", "backend": "onnx"},
  {"hf_name": "speechbrain/spkrec-xvect-voxceleb",         "local_path": "speechbrain/xvect",   "kind": "embedding", "backend": "speechbrain"},
  {"hf_name": "speechbrain/spkrec-ecapa-voxceleb",         "local_path": "speechbrain/ecapa",   "kind": "embedding", "backend": "speechbrain"},
  {"hf_name": "speechbrain/spkrec-ecapa-voxceleb-mel-spec","local_path": "speechbrain/ecapa-mel","kind": "embedding", "backend": "speechbrain"},
  {"hf_name": "speechbrain/spkrec-resnet-voxceleb",        "local_path": "speechbrain/resnet",  "kind": "embedding", "backend": "speechbrain"},
  {"hf_name": "nvidia/speakerverification_en_titanet_large","local_path": "nvidia/titanet",     "kind": "embedding", "backend": "nemo"}
]
```

**总大小估算 ~470MB**（详见 [MYGPT_INTEGRATION.md](MYGPT_INTEGRATION.md) 第二节模型表）。

### 6.2 docker/download_models.py

```python
"""Build 期一次性下载所有模型到本地路径。"""
import json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from pyannote.audio import Model

def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).parent / "models.json").read_text())
    token = os.environ["HF_TOKEN"]   # build 期通过 docker secret 注入

    for spec in manifest:
        target = out_dir / spec["local_path"]
        if target.exists() and any(target.iterdir() if target.is_dir() else [target]):
            print(f"[skip] {spec['hf_name']}"); continue
        print(f"[download] {spec['hf_name']} → {target}")
        if spec["backend"] in ("pyannote", "speechbrain", "nemo"):
            snapshot_download(spec["hf_name"], local_dir=str(target), token=token)
        elif spec["backend"] == "onnx":
            target.parent.mkdir(parents=True, exist_ok=True)
            from huggingface_hub import hf_hub_download
            hf_hub_download(spec["hf_name"], filename="model.onnx", local_dir=str(target.parent), token=token)
        # smoke load 验证
        if spec["backend"] == "pyannote":
            Model.from_pretrained(str(target), use_auth_token=False)

if __name__ == "__main__":
    main(Path(sys.argv[1]))
```

### 6.3 Dockerfile.cpu

```dockerfile
# ───── Stage 1: builder ─────
FROM python:3.11-slim AS builder
ARG HF_TOKEN
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml setup.cfg setup.py README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e ".[server]"

COPY docker/download_models.py docker/models.json ./docker/
RUN HF_TOKEN=$HF_TOKEN python docker/download_models.py /app/models

# ───── Stage 2: runtime ─────
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 diart

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /app/models /app/models
COPY src/diart_server /app/diart_server
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown -R diart:diart /app

USER diart
WORKDIR /app
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DIART_MODELS_DIR=/app/models \
    DIART_DEVICE=cpu \
    DIART_PORT=7007 \
    PYTHONUNBUFFERED=1

EXPOSE 7007
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "diart_server.app:app", "--host", "0.0.0.0", "--port", "7007", "--workers", "1"]
```

### 6.4 Dockerfile.gpu

差异：
- base 改为 `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- 在 base 上手动 `apt install python3.11 python3.11-venv`
- torch 装 CUDA wheel：`torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121`
- `ENV DIART_DEVICE=cuda`

### 6.5 docker/entrypoint.sh

```bash
#!/bin/bash
set -e
echo "[entrypoint] device=$DIART_DEVICE models_dir=$DIART_MODELS_DIR"
ls -la "$DIART_MODELS_DIR" || { echo "models dir missing"; exit 1; }
exec "$@"
```

### 6.6 离线启动验证

`docker/smoke_test.sh`（CI 调用）：

```bash
#!/bin/bash
set -e
IMAGE=$1
PORT=17007
docker run -d --name diart-smoke --network none -p $PORT:7007 $IMAGE
trap "docker rm -f diart-smoke" EXIT

for i in {1..30}; do
  if curl -sf http://localhost:$PORT/readyz; then break; fi
  sleep 2
done
curl -sf http://localhost:$PORT/readyz || { echo "readyz timeout"; exit 1; }
curl -sf http://localhost:$PORT/v1/info | jq -e '.models | length >= 10'
echo "[smoke] OK"
```

---

## 七、配置与运行

### 7.1 单机部署示例

```bash
# CPU
docker run -d --name diart-server -p 7007:7007 \
  -e DIART_MAX_CONCURRENT_SESSIONS=4 \
  diart-server:cpu

# GPU
docker run -d --name diart-server --gpus all -p 7007:7007 \
  -e DIART_MAX_CONCURRENT_SESSIONS=8 \
  diart-server:gpu
```

### 7.2 mygpt 端 client 示例

```python
import asyncio, websockets, numpy as np

async def stream(wav_pcm: np.ndarray, sr=16000):
    url = "ws://diart:7007/v1/diarize/stream?sample_rate=16000&segmentation_model=pyannote/segmentation-3.0"
    async with websockets.connect(url) as ws:
        ready = await ws.recv()
        # 推流 0.5s 一帧
        chunk = int(sr * 0.5)
        for i in range(0, len(wav_pcm), chunk):
            await ws.send(wav_pcm[i:i+chunk].astype(np.float32).tobytes())
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                handle(msg)
            except asyncio.TimeoutError: pass
        await ws.send('{"type":"end"}')
        async for msg in ws: handle(msg)
```

---

## 八、可观测性

### 8.1 Logs（结构化 JSON）

```json
{"ts":"2026-05-04T10:00:00Z","level":"INFO","event":"session_open","session_id":"abc","models":{...}}
{"ts":"...","level":"INFO","event":"session_close","session_id":"abc","duration_sec":35.6,"predictions":71,"num_speakers":3}
{"ts":"...","level":"WARN","event":"queue_overflow","session_id":"abc","dropped_frames":1}
{"ts":"...","level":"ERROR","event":"inference_crash","session_id":"abc","traceback":"..."}
```

不打 prediction 级日志（量太大），只打开/关/错。

### 8.2 探针逻辑

| 探针 | 通过条件 |
|---|---|
| `/healthz` | 进程在跑（FastAPI 直接回 200） |
| `/readyz` | warm-up 完成 + active_sessions < MAX |

---

## 九、失效模式与缓解

| 失效 | 检测 | 响应 |
|---|---|---|
| 客户端推流采样率超出范围 | query 参数验证 | 拒绝连接，4400 |
| 客户端推空帧 | `pcm.size == 0` | 送 `BAD_AUDIO`，丢弃 |
| 重采样异常 | torchaudio 抛错 | 送 `RESAMPLE_FAILED`，丢弃 |
| inference 线程崩溃 | `_run` try-except | 送 `INFERENCE_CRASH` 后关闭连接；session 标记不健康 |
| 模型加载 OOM | warm-up 抛错 | 启动失败，readyz 永远 503，pod 重启 |
| 队列满 | `put_nowait` 抛 Full | 丢最旧帧 + 累计 metric `queue_overflow_total` |
| 单帧 > 10MB | `len(frame) > MAX` | 送 `RATE_LIMIT_EXCEEDED`，关闭连接 |
| 长连接静默 | 60s 无帧的 keepalive | 关闭连接 |
| 容量满 | `active_sessions >= MAX` | accept 后立即送 `SESSION_CAPACITY_FULL` 并关闭 |

---

## 十、测试计划

### 10.1 单元测试

| 测试 | 验证 |
|---|---|
| `test_meeting_audio_source.py` | push/read/close 正常路径；满队列丢最旧；close 后 read 退出 |
| `test_audio_decoder.py` | 8k/22k/44.1k/48k → 16k 正确；空帧抛 `BadAudioError` |
| `test_protocol.py` | 所有 schema 序列化/反序列化；非法字段报错 |
| `test_models_registry.py` | 路径解析；ModelNotFound；warm-up 加载真实模型（用 fixture wav 测一次推理） |

### 10.2 集成测试

| 测试 | 验证 |
|---|---|
| `test_session.py` | 用本地 wav 推 `DiarizationSession.push_pcm`，验证 `_on_prediction` 收到至少 N 条；close 返回正确 final |
| `test_e2e_websocket.py` | 起 FastAPI test server，WS 客户端推 2speakers.wav；断言收到 ready / prediction / final；与 `diart.stream` CLI 输出 RTTM 对比，speaker turn 时间差 ±0.5s |

### 10.3 负载与稳定性

| 测试 | 验证 |
|---|---|
| `test_load_4_concurrent.py` | 4 路并发推不同 wav；断言 speaker_0 互不串扰；端到端 p99 < 1s |
| 长稳测（手动） | 4 路连续推 1 小时；监控 RAM 增长 / 线程数 / 显存 |

### 10.4 离线验证

| 测试 | 验证 |
|---|---|
| smoke_test.sh | `docker run --network none` 启动正常，readyz=200，info 列出 ≥ 10 模型 |

---

## 十一、CI / 发布

### 11.1 .github/workflows/build-images.yml

```yaml
name: Build Images
on:
  push:
    branches: [main]
    tags: ['v*']
  workflow_dispatch:

jobs:
  build-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build CPU image
        uses: docker/build-push-action@v5
        with:
          file: docker/Dockerfile.cpu
          tags: ghcr.io/juanmc2005/diart-server:${{ github.ref_name }}-cpu
          build-args: HF_TOKEN=${{ secrets.HF_TOKEN }}
          push: false
          load: true
      - name: Smoke test
        run: bash docker/smoke_test.sh ghcr.io/juanmc2005/diart-server:${{ github.ref_name }}-cpu
      - name: Push
        if: success()
        run: docker push ghcr.io/juanmc2005/diart-server:${{ github.ref_name }}-cpu

  build-gpu:
    runs-on: [self-hosted, gpu]   # 需要自托管带 GPU 的 runner
    # 同上结构，dockerfile 换 .gpu
```

### 11.2 版本策略

| 镜像 tag | 含义 |
|---|---|
| `0.9.2-cpu`, `0.9.2-gpu` | 与 diart 库版本绑定 |
| `latest-cpu`, `latest-gpu` | 滚动指向最新 release |
| `dev-cpu`, `dev-gpu` | main 分支每次 push 覆盖 |

---

## 十二、里程碑

| M | 内容 | 工时 | 交付物 |
|---|---|---|---|
| **M1** | 骨架：FastAPI + WS + `DiarizationSession` + `MeetingAudioSource`；单连接推 wav e2e 跑通 | 3 天 | 单连接 demo |
| **M2** | 多连接并发 + 容量限制 + healthz/readyz/info 端点 | 1 天 | 4 路并发不串扰 |
| **M3** | Embedding 输出 + Pydantic schema + 多模型 query 切换 | 2 天 | 完整协议 |
| **M4** | `Dockerfile.cpu` + `download_models.py` + 离线启动验证 | 2 天 | image:cpu |
| **M5** | `Dockerfile.gpu` + GPU 测试 + 性能基线 | 1 天 | image:gpu |
| **M6** | 集成测试 + Prometheus metrics + 失效模式覆盖 | 2 天 | 测试套件 |
| **M7** | CI 双 tag 构建 + 文档 + mygpt 端 client 示例 | 1 天 | 可发布 |
| | **合计** | **12 天** | |

M3 / M4 / M5 / M6 在 M2 完成后可部分并行（M5 依赖 M4 的 build 流水线）。

---

## 十三、风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| HF 模型协议变更 / 下载失败 | 中 | build 失败 | retry + 镜像源 fallback；本地缓存层 |
| 模型加载顺序 OOM（特别是同时加载所有 embedding） | 中 | warm-up 崩 | warm-up 只加载 default；其他懒加载；监控内存峰值 |
| 客户端推 PCM 字节序错误 | 高 | 输出全是噪声 | 协议明确写 LE；server 接到异常分布的音频时打 warn metric |
| 多客户端共享模型权重的线程安全 | 低 | 偶发结果错乱 | pyannote forward 是线程安全的（torch eval 模式）；单元测试加并发推理 |
| ONNX runtime 与 PyTorch CUDA 冲突（GPU 镜像） | 中 | session 创建失败 | onnxruntime-gpu 与 torch 同 CUDA 版本；CI 验证 |
| 下游客户端误用：每条 prediction 都单独做声纹匹配 | 中 | 调用方性能差 | 文档明确推荐用 `samples_count` ≥ 6 (3s) 后再匹配 |
| diart 库升级 break 服务层 | 低 | 服务无法启动 | 在 setup.cfg 锁 diart 范围版本；CI 跑兼容性测 |
| GPU runner 资源稀缺 | 中 | CI 慢 | 自托管 + spot 实例 |

---

## 十四、范围之外（明确不做）

- **声纹匹配 / 声纹库**：容器只输出 raw embedding；声纹库由调用方维护
- **ASR / 翻译 / LLM 集成**：容器单一职责
- **跨容器横向扩展 / sticky session 协调**：1–4 路单容器够用
- **HTTP 文件上传离线批处理端点**：本期只做流式
- **gRPC 接口**：WebSocket 已够用
- **鉴权 / TLS 终结**：假定内网部署或外层 ingress 处理
- **修改 diart 库自身代码**：所有改动局限在新增的 `src/diart_server/`
- **多 GPU 调度**：单容器单 GPU；多 GPU 通过多容器实例化
- **模型量化 / 蒸馏**：保持原始精度

---

## 十五、参考

- 架构动机与多种方案对比：[CONTAINERIZATION.md](CONTAINERIZATION.md)
- in-process 嵌入的替代方案：[MYGPT_INTEGRATION.md](MYGPT_INTEGRATION.md)
- 模型清单与延迟基准：[README.md](README.md) §"Pre-trained models"
- diart 核心 API：[src/diart/blocks/diarization.py](src/diart/blocks/diarization.py)、[src/diart/sources.py](src/diart/sources.py)、[src/diart/inference.py](src/diart/inference.py)、[src/diart/models.py](src/diart/models.py)
- 复现性与调参建议：[README.md](README.md) §"Reproducibility"

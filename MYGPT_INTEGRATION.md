# Diart 集成到 mygpt-audio-streaming 实时会议系统：技术实现计划书

> 把 [diart](src/diart/) 作为 Python 库嵌入到现有的 [mygpt-audio-streaming-frontend](../mygpt-audio-streaming-frontend) FastAPI 后端，在不破坏现有 ASR / 翻译 / LLM 管线的前提下补齐**说话人区分 + 声纹识别**能力。
>
> 本文档基于对两个项目的代码全量探索得出，是 [CONTAINERIZATION.md](CONTAINERIZATION.md) 的姊妹篇——后者讨论 diart 独立服务化，本篇讨论 in-process 嵌入。

---

## 能力差距速览：diart 自带 vs 你的系统所需

在进入详细方案之前，先把"哪些功能已有、哪些要自己做"摆清楚。

### diart 库本身只提供这三样

1. **实时说话人分离**（`SpeakerDiarization` pipeline，输出 `speaker_0/1/2` 这种**会话内匿名**标签）
2. **VAD**（`VoiceActivityDetection` pipeline）
3. **说话人嵌入提取**（[`OverlapAwareSpeakerEmbedding`](src/diart/blocks/embedding.py) building block，输出 512 维向量）

注意：**"声纹识别"不是从零做** —— diart 已经能产生嵌入向量，缺的只是**库存储 + 匹配逻辑**。

### 完整能力差距表

按"在 mygpt-audio-streaming 整体系统里有没有现成"分类：

| 能力 | mygpt 已有？ | diart 已有？ | 还需自己做？ | 在本文档对应章节 |
|---|---|---|---|---|
| 浏览器音频采集 + WS 推流 | ✅ | — | 否 | — |
| ASR（语音转文字） | ✅（外部 gradio） | ❌ | 否 | — |
| 多客户端 / 会话管理 | ✅ | ❌（单客户端限制） | 否 | — |
| MongoDB 持久化 | ✅ | ❌ | 否 | — |
| LLM（总结 / 话题 / 待办 / 风险） | ✅ | ❌ | 否 | — |
| 翻译 | ✅ | ❌ | 否 | — |
| **实时说话人分离** | ❌ | ✅ | **接进去** | 五、5.1–5.3 |
| **嵌入提取** | ❌ | ✅ | **直接用** | 五、5.6 |
| **声纹库 + 匹配** | ❌ | ❌ | **要自己写** | 五、5.5 / 5.6（`voiceprint_store` + `voiceprint_matcher`） |
| **音频解码 + 重采样到 16k mono** | 部分（解到 WAV） | ❌ | **要写适配层** | 五、5.4（`audio_decoder`） |
| **Speaker × ASR 段对齐** | ❌ | ❌ | **要自己写** | 五、5.7（`segment_aligner`） |
| **实时 UI 指示灯（前端组件）** | 待确认 | ❌ | 看前端现状，可能要做 | 六（`_broadcast_speaker_change` handler） |
| **会后分析**（说话时长 / 打断次数 / 完整 RTTM 入库） | ❌ | 部分（`PredictionAccumulator` 给 RTTM 字符串） | **要自己写**统计逻辑 | 五、5.3（`get_full_rttm()`）+ 后续上层 |
| **声纹注册流程**（用户人声入库） | ❌ | ❌ | **要做后台 CLI / 小工具** | 七、7.4（注册流程描述） |

### 一句话结论

**除了"声纹库 + 匹配"外**，你还需要自己做：**ASR 与 speaker 的时间对齐**、**音频重采样适配层**、**会后说话时长统计**、**声纹注册工具**，以及（如果前端没有）**实时"谁在说话"UI 组件**。这些已**全部覆盖**在下文第五节"新增文件清单"和第六节"现有文件修改清单"里，没有遗漏的模块。

---

## 一、目标与范围

### 目标

补齐现有会议系统四个空白能力，**所有改动只在 mygpt 侧 + 一个 `pip install diart`**：

| 能力 | 用户可见效果 | 技术形式 |
|---|---|---|
| 实时 UI 指示灯 | 房间里"谁在说话"的发光头像 | WebSocket 推 `speaker_change` 事件 |
| 实时字幕带说话人 | 字幕条加 "Alice: ..." 前缀 | ASR segment 附 `speaker_id` / `user_id` 字段 |
| 声纹识别 | speaker_0 → 真实用户 alice | 启动期注册声纹 + 余弦相似度匹配 |
| 会后分析 | 说话时长 / 打断次数 / 完整 RTTM | MongoDB `meeting.diarization` 字段 |

### 范围（in / out）

| in | out |
|---|---|
| 改 mygpt 现有 WS 路由、ASR processor、client_session | 不改 ASR 模型 / 翻译 / LLM 管线 |
| 加 diart 进 requirements.txt，用其 Python API | 不开 diart 独立服务（K8s 那套留给上规模时再上） |
| 声纹库（MongoDB 集合）+ 注册接口 | 声纹自助 UI、跨会议身份联邦 |
| Feature flag `DIARIZATION_ENABLED` 默认关闭 | diart 自身代码改动 |

---

## 二、现状速览

### mygpt 后端（FastAPI + WebSocket）

| 维度 | 现状 |
|---|---|
| 入口 | [`ws_modules/real_time_server.py`](../mygpt-audio-streaming-frontend/ws_modules/real_time_server.py) |
| 主消息循环 | [`ws_modules/routes/websocket_routes.py:112-1020`](../mygpt-audio-streaming-frontend/ws_modules/routes/websocket_routes.py) |
| 接收的二进制帧 | `[flags(1B)][segment_id(4B BE)][audio_data(N)]`，flags bit0 = is_final |
| 音频编码 | 客户端用 MediaRecorder → WebM (Matroska) 或 WAV，分块大小约 1s |
| 缓冲与触发 | [`ws_modules/services/audio_buffer_service.py:69-200`](../mygpt-audio-streaming-frontend/ws_modules/services/audio_buffer_service.py)（VAD + 触发 partial / final） |
| 解码到 WAV 临时文件 | [`websocket_routes.py:606-609, 884-886`](../mygpt-audio-streaming-frontend/ws_modules/routes/websocket_routes.py) |
| ASR 入口 | [`ws_modules/services/audio_processor.py`](../mygpt-audio-streaming-frontend/ws_modules/services/audio_processor.py) → `transcribe_audio_only` / `transcribe_and_translate_multi` |
| Per-connection state | [`ws_modules/models/client_session.py`](../mygpt-audio-streaming-frontend/ws_modules/models/client_session.py)（语言、缓冲、各 manager） |
| 并发 | asyncio + 两个线程池（`final_executor=4`，`partial_executor=3`） |
| 持久化 | MongoDB（`MONGO_ENABLED=True`），断连可恢复 |
| 客户端类型 | 普通参与者（user_id）+ `Projection_*`（共享麦克风） |
| 说话人区分相关 | **完全空白** |

### diart 关键能力

| 维度 | 现状 |
|---|---|
| 主管线 | [`SpeakerDiarization`](src/diart/blocks/diarization.py)（分割 + 嵌入 + 在线聚类） |
| 默认参数 | `duration=5s, step=0.5s, latency=0.5s, sample_rate=16000` |
| 自定义流接口 | 继承 [`AudioSource`](src/diart/sources.py)，调 `self.stream.on_next(tensor)` |
| 推理编排 | [`StreamingInference`](src/diart/inference.py) + `attach_hooks` / `attach_observers` |
| 嵌入提取 | [`OverlapAwareSpeakerEmbedding`](src/diart/blocks/embedding.py) |
| 模型权重 | `pyannote/segmentation-3.0` + `pyannote/embedding`（HF Token 必需） |
| 状态特性 | 模型权重无状态共享 / 聚类**有状态、非线程安全** → 必须 per-connection 实例 |

---

## 三、架构决策

### 决策 1：In-process 库集成（不开独立服务）

**结论**：直接 `pip install diart`，作为 mygpt FastAPI 进程的一部分跑。

**理由**：

- 1–4 路并发规模，独立服务（K8s + sticky session）属于过度设计
- 模型权重可在主进程内 singleton 共享，省一份显存
- 跨进程通信（WebSocket / gRPC）增加复杂度但无收益
- 代价：FastAPI 进程显存 +2–3GB（torch + pyannote）→ 需 GPU runtime；接受

**何时转独立服务**：当并发 ≥ 数十路、或需要多语言后端共用 diart 时，按 [CONTAINERIZATION.md](CONTAINERIZATION.md) 方案上 K8s。

### 决策 2：每条 WebSocket 连接一个 `SpeakerDiarization` 实例

[`OnlineSpeakerClustering`](src/diart/blocks/clustering.py) 状态非线程安全，且每个会议的 speaker label 应当独立。**每连接一个实例**正好契合现有 `ClientSession` 模式。

模型权重通过 [`PyannoteLoader`](src/diart/models.py) 模块级 singleton 共享，per-instance 只持有聚类状态。

### 决策 3：两种模式按 client 类型切换

```
普通参与者 (1 mic / 1 人)    → embedding-only 模式：跳过聚类，只抽嵌入 → 声纹匹配
Projection_* (1 mic / 多人)  → full diarization：完整管线，聚类 + 声纹匹配
```

**理由**：单人流跑聚类必然只产生一簇，浪费 CPU；但仍需嵌入做声纹匹配。

### 决策 4：异步桥接走"后台线程 + 队列"

[`StreamingInference.__call__`](src/diart/inference.py) 是 blocking 的（用 RxPY 阻塞订阅）。在 asyncio 主循环里直接调会卡死。

**方案**：
- 每个 `DiarizationSession.start()` 起一个 daemon 线程跑 `inference()`
- 主 asyncio handler 用 `queue.Queue` 把 PCM 帧 push 进去（线程安全、非阻塞）
- 队列满时丢最旧帧（背压保护）

### 决策 5：feature flag + 默认关闭

`ws_config.DIARIZATION_ENABLED = False`，灰度开启。所有调用点都走 `if session.diarization_session is not None` 短路，确保关闭时零开销。

---

## 四、数据流

```
客户端 (MediaRecorder, WebM)
   │
   │ WebSocket 二进制帧 [flags|segment_id|audio_data]
   ▼
[现有] websocket_routes.py:743-955 (audio_chunk handler)
   │
   ▼
[现有] audio_buffer_service.add_audio_chunk()
   │  ├─ VAD 流式检测
   │  └─ WebM/WAV 解码（pydub）
   ▼
[现有] WAV bytes (临时文件 / 内存 buffer)
   │
   ├──────────────────────────────────────────┐
   │ (现有路径，不动)                          │ (新增分支)
   ▼                                           ▼
ASR 触发逻辑                       audio_decoder.to_pcm_16k_mono()
   │                                           │
   ▼                                           ▼
ws_asr_processors._process_(partial|final)  DiarizationSession.push_pcm(pcm)
   │                                           │
   ▼                                    [非阻塞 enqueue]
ASR text + segment [start, end]                │
   │                                           ▼
   │                          [后台线程] StreamingInference 持续消费
   │                                           │
   │                                           ▼
   │                          attach_hooks 回调 (Annotation, Waveform)
   │                                           │
   │                          ┌────────────┬───┴──────┬──────────────┐
   │                          ▼            ▼          ▼              ▼
   │                   voiceprint_     speaker_   MongoDB      内存中维护
   │                    matcher       change     RTTM         active_speaker
   │                   (≥3s 后)       broadcast  累积                │
   │                                                                │
   └─────────────────────[等待 final ASR 完成]──────────────────────┘
                                       │
                                       ▼
                       segment_aligner.align(asr_segment, annotation)
                                       │
                                       ▼
                       ASR segment 附 speaker_id / user_id
                                       │
                                       ▼
                       现有 LLM / 翻译 / MongoDB 持久化（不动）
```

---

## 五、新增文件清单

所有新增放在 `ws_modules/services/diarization/` 下。

### 5.1 `diart_singleton.py` — 进程级共享

```python
# 函数签名草图
_seg_model: Optional[SegmentationModel] = None
_emb_model: Optional[EmbeddingModel] = None
_lock = threading.Lock()

def warm_up() -> None:
    """启动期同步加载模型，避免首连接冷启动 30-60s。"""

def get_models() -> tuple[SegmentationModel, EmbeddingModel]:
    """Lazy + 双检锁。"""
```

调用点：[`real_time_server.py:79-80`](../mygpt-audio-streaming-frontend/ws_modules/real_time_server.py)（线程池初始化处之前）。

### 5.2 `meeting_audio_source.py` — 自定义流

继承 [`AudioSource`](src/diart/sources.py)：

```python
class MeetingAudioSource(AudioSource):
    def __init__(self, sample_rate=16000, uri="meeting", maxsize=50):
        super().__init__(uri, sample_rate)
        self._queue = queue.Queue(maxsize=maxsize)
        self._closed = False

    def push_pcm(self, pcm: np.ndarray) -> None:
        """主线程调用，非阻塞。队列满则丢最旧帧（背压）。"""
        try:
            self._queue.put_nowait(pcm)
        except queue.Full:
            try: self._queue.get_nowait()
            except queue.Empty: pass
            self._queue.put_nowait(pcm)

    def read(self) -> None:
        """diart 内部线程调用，阻塞循环消费。"""
        while not self._closed:
            pcm = self._queue.get()
            if pcm is None: break
            tensor = torch.from_numpy(pcm).unsqueeze(0)  # (1, samples)
            self.stream.on_next(tensor)
        self.stream.on_completed()

    def close(self) -> None:
        self._closed = True
        self._queue.put(None)  # 唤醒 read()
```

### 5.3 `diarization_session.py` — 每连接一个

封装一个完整的 diart 管线 + 后台线程。

```python
class DiarizationSession:
    def __init__(self, mode: Literal["full", "embedding_only"], on_speaker_change: Callable, on_embedding: Callable):
        self.mode = mode
        seg, emb = diart_singleton.get_models()
        config = SpeakerDiarizationConfig(
            segmentation=seg, embedding=emb,
            duration=5.0, step=0.5, latency=0.5,
            tau_active=cfg.DIARIZATION_TAU_ACTIVE,
            rho_update=cfg.DIARIZATION_RHO_UPDATE,
            delta_new=cfg.DIARIZATION_DELTA_NEW,
            max_speakers=cfg.DIARIZATION_MAX_SPEAKERS,
            sample_rate=16000,
            device=torch.device(cfg.DIARIZATION_DEVICE),
        )
        self.pipeline = SpeakerDiarization(config)
        self.source = MeetingAudioSource()
        self.inference = StreamingInference(self.pipeline, self.source, do_plot=False, show_progress=False)
        self.inference.attach_hooks(self._on_prediction)
        self.accumulator = PredictionAccumulator(self.source.uri)
        self.inference.attach_observers(self.accumulator)
        self._active_speaker: Optional[str] = None
        self._embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
        self._on_speaker_change = on_speaker_change
        self._on_embedding = on_embedding
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self): self._thread.start()
    def push_pcm(self, pcm): self.source.push_pcm(pcm)

    def _run(self):
        try: self.inference()
        except Exception: log.exception("diarization inference crashed")

    def _on_prediction(self, ann_wav):
        annotation, waveform = ann_wav
        # 1) 检测 active speaker 变化
        # 2) 累积 embedding（mode=embedding_only 时直接从 waveform + 当前帧 segmentation 抽）
        # 3) 当 speaker 累计时长 ≥ 3s，触发声纹匹配回调
        ...

    def get_full_rttm(self) -> str: return self.accumulator.get_prediction().to_rttm()
    def close(self):
        self.source.close()
        self._thread.join(timeout=2.0)
```

### 5.4 `audio_decoder.py` — 解码工具

```python
def webm_to_pcm_16k_mono(webm_bytes: bytes) -> np.ndarray:
    """WebM → float32 PCM @ 16kHz mono。复用 pydub + torchaudio.functional.resample。"""

def wav_to_pcm_16k_mono(wav_bytes: bytes) -> np.ndarray:
    """WAV → float32 PCM @ 16kHz mono。"""
```

### 5.5 `voiceprint_store.py` — MongoDB 声纹库

集合 `voiceprints`，文档结构：

```json
{
  "_id": ObjectId("..."),
  "user_id": "alice",
  "embedding": [0.123, -0.456, ...],   // 512 dim float
  "samples_count": 5,
  "updated_at": "2026-05-04T10:00:00Z"
}
```

接口：

```python
class VoiceprintStore:
    async def register(self, user_id: str, embedding: np.ndarray) -> None:
        """新增或滑动平均更新某用户的声纹。"""

    async def list_all(self) -> list[tuple[str, np.ndarray]]:
        """启动期一次性加载到内存（数量 < 1000，不需要 ANN）。"""
```

### 5.6 `voiceprint_matcher.py` — 余弦匹配

```python
class VoiceprintMatcher:
    def __init__(self, store: VoiceprintStore, threshold=0.7):
        self._cache: list[tuple[str, np.ndarray]] = []  # [(user_id, embedding)]

    async def warm_up(self): self._cache = await self.store.list_all()

    def match(self, embedding: np.ndarray) -> Optional[str]:
        """返回 user_id 或 None。余弦相似度 > threshold 视为命中。"""
```

### 5.7 `segment_aligner.py` — ASR × diart 对齐

```python
def align(asr_segments: list[ASRSegment], annotation: Annotation, speaker_to_user: dict[str, str]) -> list[ASRSegment]:
    """
    对每条 ASR segment [start, end]：
      1. 取 annotation 在该区间内所有 turn
      2. 按重叠时长加权，选最大者作为 speaker_label
      3. 查 speaker_to_user 映射拿 user_id（可能为 None）
      4. 给 segment 附 speaker_id / user_id 字段
    """
```

---

## 六、现有文件修改清单

| 文件 | 行号 / 锚点 | 修改内容 |
|---|---|---|
| [`ws_modules/config/ws_config.py`](../mygpt-audio-streaming-frontend/ws_modules/config/ws_config.py) | 末尾追加 | 新增 8 个 `DIARIZATION_*` / `VOICEPRINT_*` 配置项（见第八节） |
| [`ws_modules/real_time_server.py`](../mygpt-audio-streaming-frontend/ws_modules/real_time_server.py) | `~:79-80` 之前 | `if cfg.DIARIZATION_ENABLED: diart_singleton.warm_up(); voiceprint_matcher.warm_up()` |
| [`ws_modules/models/client_session.py`](../mygpt-audio-streaming-frontend/ws_modules/models/client_session.py) | `__init__` | 新增字段 `self.diarization_session: Optional[DiarizationSession] = None`；按 `client_id.startswith("Projection_")` 决定 `mode` |
| [`ws_modules/routes/websocket_routes.py`](../mygpt-audio-streaming-frontend/ws_modules/routes/websocket_routes.py) | `:743-955`（audio_chunk 分支） | 解码后 `if session.diarization_session: pcm = audio_decoder.to_pcm_16k_mono(wav); session.diarization_session.push_pcm(pcm)` |
| 同上 | 断连分支 | `if session.diarization_session: session.diarization_session.close()` |
| 同上 | 连接建立分支 | `if cfg.DIARIZATION_ENABLED: session.diarization_session = DiarizationSession(...); session.diarization_session.start()` |
| [`ws_modules/services/ws_asr_processors.py`](../mygpt-audio-streaming-frontend/ws_modules/services/ws_asr_processors.py) | `:200+`（`_process_final_asr` 末尾） | `segment_aligner.align(asr_segments, ann, session.speaker_to_user)` 给段加 speaker_id / user_id |
| [`ws_modules/handlers/ws_ai_handlers.py`](../mygpt-audio-streaming-frontend/ws_modules/handlers/ws_ai_handlers.py) | 新增 handler | `_broadcast_speaker_change(meeting_id, speaker_id, user_id, ts)` 推 `{"type":"speaker_change",...}` |
| `requirements.txt` | 末尾 | `diart>=0.9.2`（连带 pyannote.audio / torch / rx） |
| `Dockerfile` | RUN 阶段 | 预下载 pyannote 模型（避免冷启动联网），需要 build 期 HF token |

---

## 七、关键技术细节

### 7.1 Async ↔ RxPY 桥接（最大坑）

**问题**：[`StreamingInference.__call__`](src/diart/inference.py) 内部用 RxPY `subscribe(...)`，会阻塞调用线程。在 asyncio handler 里直接 `inference()` 会卡死整个 event loop。

**解决**：
1. `MeetingAudioSource.read()` 用阻塞 `queue.get()` 而不是迭代器
2. `DiarizationSession._thread`（daemon thread）执行 `self.inference()`，让 `read()` 在该线程里跑
3. 主 asyncio handler 只调 `push_pcm()`（put_nowait，非阻塞）
4. 关闭时 `source.close()` 给队列丢 `None`，read 退出，inference 自然结束

**线程数评估**：4 路并发 → 4 个 inference 线程 + 4 个聚类状态副本，总线程数可控。

### 7.2 解码与重采样

| 维度 | 现状 | 处理 |
|---|---|---|
| 客户端编码 | WebM (Opus / Vorbis)，48kHz stereo 居多 | pydub 解码到 PCM int16 |
| 现有解码点 | [`audio_buffer_service.py`](../mygpt-audio-streaming-frontend/ws_modules/services/audio_buffer_service.py) + [`websocket_routes.py:606-609`](../mygpt-audio-streaming-frontend/ws_modules/routes/websocket_routes.py)（输出 WAV） | 复用，不重复解码 |
| diart 期望 | float32 PCM @ 16kHz mono，shape `(1, samples)` | `audio_decoder` 做 int16→float32 / mono / 重采样 |
| 重采样 | torchaudio 比 pydub 快 5–10× | 用 `torchaudio.functional.resample` |

**注意**：MediaRecorder 的 WebM 是流式格式——首块含完整 EBML 头，后续仅 Cluster。现有 [`audio_buffer_service.py:24-66`](../mygpt-audio-streaming-frontend/ws_modules/services/audio_buffer_service.py) 已处理；diart 集成层在解码后接入，对此透明。

### 7.3 Speaker × ASR segment 对齐

ASR 段时间轴（`[start, end]`，相对会议起点）和 diart `Annotation` 时间轴**都基于音频内部时钟**，但来源不同（ASR 来自外部服务返回，diart 来自内部 RxPY 流）。**关键**：以 mygpt 这边的 segment_id 累积时间为准，diart 这边用 `MeetingAudioSource` 内部维护的累积 sample 数推时间。两者都从连接建立起点 0 开始，对齐天然成立。

对齐算法：

```python
def align_one(asr_seg, ann):  # ann: pyannote.core.Annotation
    overlaps = defaultdict(float)
    for turn, _, speaker in ann.itertracks(yield_label=True):
        intersect = max(0, min(asr_seg.end, turn.end) - max(asr_seg.start, turn.start))
        if intersect > 0:
            overlaps[speaker] += intersect
    return max(overlaps, key=overlaps.get) if overlaps else None
```

边界情况——ASR 段跨 speaker（多个 speaker 重叠时长接近）：本期取最大者，**不分裂**。如需分裂，将 ASR text 按时间比例切（粗略但够用）作为 v2 优化。

### 7.4 声纹库与匹配

**注册流程**（线下）：
1. 用户录一段 ≥10s 的清晰人声
2. 用 `OverlapAwareSpeakerEmbedding` 提取嵌入
3. `voiceprint_store.register(user_id, embedding)` 入库

**匹配流程**（在线）：
1. `DiarizationSession._on_prediction` 累积每个 speaker_label 的嵌入
2. 当某 speaker 累积时长 ≥ `VOICEPRINT_MIN_DURATION_SEC`（默认 3s），均值化嵌入
3. `voiceprint_matcher.match(emb)` → user_id 或 None
4. 命中后写入 `session.speaker_to_user[speaker_label] = user_id`，后续 segment / 广播都带上

**threshold**：cos sim > 0.7 视为命中（pyannote/embedding 在 VoxCeleb 上的经验值）。

### 7.5 模型 warm-up 与 HF token

| 项 | 处理 |
|---|---|
| HF token 注入 | `HF_TOKEN` 环境变量；启动期 `huggingface_hub.login(token)` |
| 模型 bake | Dockerfile 里 `RUN python -c "from diart.models import ..."` 触发下载，build 期需 token（用 `--secret`） |
| Warm-up 时机 | `real_time_server.py` 启动期同步执行，10–20s 阻塞 |
| 失败行为 | warm-up 失败 → 日志 ERROR，后续连接 `DIARIZATION_ENABLED` 自动降为 False（不阻塞 ASR） |

### 7.6 推荐 diart 配置

取自 [README:394-400](README.md) 的 AMI 数据集调优值（最贴近会议场景）：

```python
SpeakerDiarizationConfig(
    duration=5.0,       # 5s rolling window
    step=0.5,           # 每 0.5s 更新一次
    latency=0.5,        # 最低延迟
    tau_active=0.507,   # AMI tuning
    rho_update=0.006,
    delta_new=1.057,
    max_speakers=10,
    sample_rate=16000,
)
```

如果实测 false-positive speaker 太多（噪声被识别为新人），调大 `delta_new`（如 1.5）。如果漏 speaker（合并不同人），调小。

---

## 八、配置项（ws_config.py 追加）

```python
# 说话人区分（diart 集成）
DIARIZATION_ENABLED = os.getenv("DIARIZATION_ENABLED", "false").lower() == "true"
DIARIZATION_HF_TOKEN = os.getenv("HF_TOKEN", "")
DIARIZATION_DEVICE = os.getenv("DIARIZATION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DIARIZATION_TAU_ACTIVE = float(os.getenv("DIARIZATION_TAU_ACTIVE", "0.507"))
DIARIZATION_RHO_UPDATE = float(os.getenv("DIARIZATION_RHO_UPDATE", "0.006"))
DIARIZATION_DELTA_NEW = float(os.getenv("DIARIZATION_DELTA_NEW", "1.057"))
DIARIZATION_LATENCY = float(os.getenv("DIARIZATION_LATENCY", "0.5"))
DIARIZATION_MAX_SPEAKERS = int(os.getenv("DIARIZATION_MAX_SPEAKERS", "10"))

# 声纹识别
VOICEPRINT_ENABLED = os.getenv("VOICEPRINT_ENABLED", "false").lower() == "true"
VOICEPRINT_MATCH_THRESHOLD = float(os.getenv("VOICEPRINT_MATCH_THRESHOLD", "0.7"))
VOICEPRINT_MIN_DURATION_SEC = float(os.getenv("VOICEPRINT_MIN_DURATION_SEC", "3.0"))
```

---

## 九、验证与测试方案

### 9.1 单元测试

| 测试 | 验证 |
|---|---|
| `test_meeting_audio_source.py` | push_pcm 满队列时丢最旧帧；close 后 read 正确退出 |
| `test_audio_decoder.py` | WebM 48k stereo → 16k mono 长度 / 数值正确 |
| `test_segment_aligner.py` | 各种重叠场景（完全包含 / 部分重叠 / 跨 speaker） |
| `test_voiceprint_matcher.py` | 阈值边界、空库、缓存刷新 |

### 9.2 集成测试

| 测试 | 步骤 | 预期 |
|---|---|---|
| 离线对照 | 用 `diart.stream sample_2speakers.wav` 拿 RTTM → 用 mygpt WS 客户端推同一 wav 拿 RTTM | speaker turn 时间差 ≤ ±0.5s |
| 实时事件 | WS 推双人对话 | 客户端收到 `speaker_change` 事件 ≥ 2 次 |
| ASR 绑定 | 同上 | MongoDB `meeting.segments` 每条都有 `speaker_id` |
| 声纹匹配 | 先注册 alice，再推 alice 音频 | `segment.user_id == "alice"` |
| 声纹未命中 | 注册 alice，推 bob 音频 | `segment.user_id is None`，speaker_id 仍存在 |

### 9.3 并发与稳定性

| 测试 | 验证 |
|---|---|
| 4 路并发不串扰 | 每路推不同对话，session A 的 speaker_0 ≠ session B 的 speaker_0 |
| 端到端延迟 | push_pcm → speaker_change 广播 ≤ 1.0s |
| 资源回收 | 断连后 `nvidia-smi` 显存回收，`threading.active_count()` 不增长 |
| 模型 warm-up 失败降级 | mock HF token 无效 → 启动不崩，新连接 `DIARIZATION_ENABLED` 自动 False |
| 长会议 | 连续 60min push 音频，无 OOM、无聚类爆炸（speaker 数 ≤ max_speakers） |

### 9.4 灰度方案

1. 默认 `DIARIZATION_ENABLED=false` 部署
2. 单测试会议室手动开 → 观察 1 天
3. 全量开启

---

## 十、实施阶段

| 阶段 | 内容 | 预估工时 |
|---|---|---|
| **M1：骨架** | `MeetingAudioSource` + `DiarizationSession` + `audio_decoder` + `diart_singleton`；离线 wav 跑通 | 2 天 |
| **M2：嵌入 mygpt** | 改 `client_session` / `websocket_routes` / `ws_config`；feature flag 控制；端到端推 wav 验通 | 2 天 |
| **M3：ASR 绑定** | `segment_aligner`；改 `ws_asr_processors` final 路径；MongoDB schema 加字段 | 1 天 |
| **M4：实时广播** | `_broadcast_speaker_change` handler；前端约定事件格式（与前端协商） | 1 天 |
| **M5：声纹** | `voiceprint_store` / `voiceprint_matcher`；注册 CLI 工具；在线匹配集成 | 2 天 |
| **M6：测试与调优** | 单元 + 集成测试；4 路并发压测；调 `tau_active` / `delta_new` 等 | 2 天 |
| **M7：部署** | Dockerfile 加 GPU 基础镜像；模型 bake；HF secret；灰度上线 | 1 天 |

**合计**：~11 人日。可并行（M3/M4/M5 在 M2 完成后并行）。

---

## 十一、风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| diart 模型加载慢（30–60s 冷启动） | 启动期阻塞、首连接超时 | `warm_up()` 同步加载 + Dockerfile bake 模型 |
| GPU 显存不够（4 路并发各占聚类状态） | OOM | 模型权重共享 → 增量很小（聚类状态 KB 级）；监控显存 |
| 重采样 CPU 开销 | 主 asyncio 线程被拖慢 | 重采样放进 `partial_executor`（已存在线程池） |
| 客户端 MediaRecorder 编码异常 | diart 收到坏 PCM 崩溃 | `audio_decoder` 加 try-except + `len(pcm) < 100` 跳过 |
| RxPY 异常未捕获 | inference 线程死掉、push_pcm 静默丢帧 | `_run` 包 try-except，崩溃后标记 session unhealthy，下次 push 抛错 |
| 声纹库 embedding 维度变化（换模型） | 新旧 embedding 无法比较 | 库文档存 `model_version`，不一致时强制重注册 |
| 长会议 speaker 数爆炸 | 聚类崩溃 | `max_speakers=10`；超过时新人被合并到最近的现有 cluster |
| Final ASR 与 diart 时序错位 | speaker_id 绑错 | 对齐用绝对时间戳，不依赖事件顺序；segment 完成时再对齐 |

---

## 十二、明确不在本期做

- **diart 独立服务化**：见 [CONTAINERIZATION.md](CONTAINERIZATION.md)，等并发 ≥ 数十路再上
- **跨会议 speaker 持久化**：不同 meeting 的 speaker_0 不保证同一人；跨会议身份完全靠声纹库
- **声纹自助注册 UI**：先靠后台 CLI 批量录入
- **多 GPU 调度 / 模型量化**：4 路并发单 GPU 够用
- **ASR segment 跨 speaker 自动分裂**：v2 优化
- **离线 batch diarization 端点**：mygpt 现有 ASR 是流式驱动，离线场景另开 endpoint 再说
- **修改 diart 自身代码**：只用其公开 API

---

## 十三、参考路径速查

### diart 侧（库，不动）

| 文件 | 用途 |
|---|---|
| [src/diart/blocks/diarization.py](src/diart/blocks/diarization.py) | 主管线 |
| [src/diart/sources.py](src/diart/sources.py) | AudioSource 基类 |
| [src/diart/inference.py](src/diart/inference.py) | StreamingInference |
| [src/diart/blocks/embedding.py](src/diart/blocks/embedding.py) | 嵌入提取 |
| [src/diart/sinks.py](src/diart/sinks.py) | PredictionAccumulator |

### mygpt 侧（要改）

| 文件 | 用途 |
|---|---|
| [ws_modules/real_time_server.py](../mygpt-audio-streaming-frontend/ws_modules/real_time_server.py) | 启动入口、warm-up 钩子 |
| [ws_modules/routes/websocket_routes.py](../mygpt-audio-streaming-frontend/ws_modules/routes/websocket_routes.py) | WS 主循环、push_pcm 调用点 |
| [ws_modules/services/audio_buffer_service.py](../mygpt-audio-streaming-frontend/ws_modules/services/audio_buffer_service.py) | 现有缓冲（不动，只读） |
| [ws_modules/services/ws_asr_processors.py](../mygpt-audio-streaming-frontend/ws_modules/services/ws_asr_processors.py) | ASR processor，final 路径加 segment_aligner |
| [ws_modules/models/client_session.py](../mygpt-audio-streaming-frontend/ws_modules/models/client_session.py) | per-connection state，挂 diarization_session |
| [ws_modules/config/ws_config.py](../mygpt-audio-streaming-frontend/ws_modules/config/ws_config.py) | 配置追加 |
| [ws_modules/handlers/ws_ai_handlers.py](../mygpt-audio-streaming-frontend/ws_modules/handlers/ws_ai_handlers.py) | speaker_change 广播 handler |

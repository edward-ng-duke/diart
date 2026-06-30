# 设计：diart_server 上的「声纹身份 + ASR」近实时 Demo

- 日期：2026-06-30
- 分支：`feat/diart-server`
- 状态：设计已确认，待用户复核 spec
- 相关文档：[STANDALONE_SERVER.md](../../../STANDALONE_SERVER.md)、[src/diart_server/README.md](../../../src/diart_server/README.md)、[MYGPT_INTEGRATION.md](../../../MYGPT_INTEGRATION.md)

---

## 1. 目标与范围

在现有 `diart_server`（实时**说话人分离** + 每段嵌入向量）之上，做一个**近实时**的浏览器 Demo，把三件事串起来：

1. **说话人分离**（diart）：谁在什么时候说话；
2. **声纹识别**（新增）：把每个说话人的嵌入向量与**预注册**声纹库做余弦匹配 → 真实姓名，匹配不上标 `unknown`；
3. **ASR**（Qwen3-ASR，外部 HTTP API）：每个说话人轮次结束时，把那段音频转写成文字。

最终在浏览器时间线上呈现：`[张三 00:03–00:07] 你好世界`。

### 1.1 关键决策（已确认）

| 维度 | 决策 |
|---|---|
| 运行模式 | **近实时**（边录边出，按说话人轮次出结果） |
| 声纹策略 | **预注册真人 + `unknown` 兜底** |
| 交付形态 | **浏览器页面**（新开 `/demo/live`，不动现有上传 demo） |
| 架构 | **Approach A：扩展 `diart_server` 进程内**（一个 WS，嵌入不出进程） |
| 模型管理 | 全部模型下载到根目录 `./models`，写 `MODELS.md`，前端可切换 |

### 1.2 ASR 的根本性约束

Qwen3-ASR 是**文件/批接口**（`POST /v1/audio/transcriptions`，文档注明 >30s 延迟明显增加）。因此即便"近实时"，ASR 也是**按轮次（utterance）粒度**调用——某说话人一段话讲完时才送这段音频，而不是逐字流式。

---

## 2. 架构总览（Approach A）

```
浏览器麦克风 ──16k/原生采样 PCM 帧──▶ diart_server WS /v1/diarize/stream
  + 注册 UI(每人几秒)                        │
                                            ├─▶ diart 流式分离 ──prediction{speaker,start,dur,embedding}─┐
                                            └─▶ PCM 环形缓冲(按时间戳)                                    │
                                                                                                        ▼
                                            ① 声纹匹配：cosine(embedding, 声纹库[本会话模型]) → 姓名/unknown
                                            ② 轮次聚合：攒成完整一句（说话人切换/静音/超时 触发收尾）
                                            ③ 收尾时：切 [start,end] PCM → 16k wav → 异步 Qwen3-ASR → 解析 <asr_text>
                                                                                                        │
浏览器时间线 ◀──transcript{speaker, 姓名, score, start, end, text, lang}───────────────────────────────┘
```

所有新增编排逻辑（①②③ + 注册）都在 `diart_server` 进程内，拆成独立可测的小模块。

---

## 3. 新增模块（`src/diart_server/`）

每个模块单一职责、可独立单测。

### 3.1 `pcm_ring_buffer.py`
按时间戳保存最近 N 秒的 16k 单声道 float32 PCM，供轮次收尾时切片。

- 状态：一个预分配 `np.ndarray` 环形缓冲 + 写指针 + 累计样本计数。
- `append(pcm: np.ndarray) -> None`：追加；满则覆盖最旧。
- `slice(t0: float, t1: float) -> np.ndarray | None`：按秒切片（内部换算 `sample = round(t * 16000)`）；若请求区间已被覆盖出窗，返回能取到的部分或 `None`。
- 容量由 `ring_buffer_seconds`（默认 120s）约束，封顶内存。

### 3.2 `voiceprint_store.py`
声纹库的持久化，**按 embedding 模型分命名空间**（不同模型向量不可比，见 §6）。

- 结构：`{embedding_model: {name: list[np.ndarray]}}`（每人允许多条参考向量）。
- 落盘：`DIART_VOICEPRINTS_DIR`（默认 `./voiceprints`），元数据 JSON + 向量 `.npz`。
- 接口：
  - `enroll(model: str, name: str, vec: np.ndarray) -> None`
  - `vectors(model: str) -> dict[str, np.ndarray]`（每人取均值后的单位向量，供匹配）
  - `list(model: str) -> list[{name, n_clips}]`
  - `remove(model: str, name: str) -> None`
  - `load()/save()`：进程启动加载，写操作后落盘。

### 3.3 `voiceprint_matcher.py`
纯函数，无 I/O，最易测。

- `match(vec: np.ndarray, refs: dict[str, np.ndarray], threshold: float) -> (name: str | None, score: float)`
- 输入向量与参考向量都先 L2 归一化 → 余弦 = 点积；取 argmax；`score < threshold` → 返回 `(None, score)`（即 `unknown`）。

### 3.4 `embedding_extractor.py`
从一段注册音频抽参考向量，**必须复用与会话相同的 embedding 模型**（来自 `ModelsRegistry`）。

- `extract(model_name: str, pcm_16k: np.ndarray, registry) -> np.ndarray`
- 内部：拿 `registry.get_embedding(model_name)`，对整段（必要时分窗）跑嵌入 → 均值 → L2 归一化。

### 3.5 `turn_aggregator.py`
把逐窗 `prediction` 攒成「完整一句」，纯逻辑、可测。

- `feed(speaker: str, start: float, end: float, now: float) -> list[FinalizedTurn]`：喂入预测，返回**本次被收尾**的轮次（可能为空）。
- `flush() -> list[FinalizedTurn]`：流结束时收尾全部未决轮次。
- 每说话人维护一个「打开的轮次」`(start, last_end)`；新预测同说话人则延伸 `last_end`。
- **收尾触发**（任一）：
  - 说话人切换（出现另一个活跃说话人）；
  - 该说话人静默 `now - last_end > silence_gap`（默认 0.7s）；
  - 轮次时长 `last_end - start > max_turn_seconds`（默认 20s，贴合 ASR 30s 甜区）；
  - 流结束（`flush`）。
- `FinalizedTurn = {speaker, start, end}`。

### 3.6 `asr_client.py`
异步调用 Qwen3-ASR。

- `async transcribe(pcm_16k: np.ndarray) -> (lang: str, text: str)`
- 内部：float32 → int16，用 `wave` 在内存编 16k 单声道 WAV → `httpx.AsyncClient` multipart `POST {asr_base_url}/audio/transcriptions`（`model`, `file`, 可选 `language`/`prompt`）→ 解析响应 `text` 的 `<asr_text>` 段（见用户给的 `parse_asr_output`）。
- 超时 `asr_timeout`（默认 60s）；失败抛异常由会话层兜底（§9）。

---

## 4. 改动现有文件（全部加法，不破坏现有行为）

### 4.1 `protocol.py`
- `PredictionMessage` 增加可选字段：`name: str | None = None`、`score: float | None = None`。
- 新增 `TranscriptMessage`：`{type:"transcript", speaker, name, score, start, end, text, lang}`。
- 新增注册接口模型：`EnrollResponse`、`VoiceprintInfo` 等。
- ⚠️ 现有 e2e 测试若对 `prediction` 做严格键集断言，需同步放宽/更新（新字段默认 `None`、additive）。

### 4.2 `session.py`
- 构造时新建 `PcmRingBuffer`、`TurnAggregator`，持有 `registry`、声纹库引用、ASR 客户端、本会话 `embedding_model`/阈值。
- `push_pcm(pcm)`：在原有 `source.push_pcm` 基础上**额外** `ring_buffer.append(pcm)`，并累加样本计数（驱动 `now` 时钟）。
- `_on_prediction`：① 用本会话模型的声纹库做匹配，给 `PredictionMessage` 补 `name/score`；② 把 `(speaker,start,end,now)` 喂 `turn_aggregator`；③ 对返回的已收尾轮次，起**异步 ASR 任务**（见 §7）。
- `close()`：先 `turn_aggregator.flush()` 把尾巴 ASR 掉，再发 `final`。

### 4.3 `voiceprint_routes.py`（新）
- `POST /v1/voiceprints`（multipart：`name`、`embedding_model`、`file`）→ 解码音频到 16k → `embedding_extractor.extract` → `store.enroll` → 返回 `EnrollResponse`。
- `GET /v1/voiceprints?embedding_model=...` → 列出该模型下已注册的人。
- `DELETE /v1/voiceprints/{name}?embedding_model=...`。
- 在 `app.py` 挂载该 router；声纹库实例随 app 生命周期创建/加载。

### 4.4 `config.py`
新增 env（均 `DIART_` 前缀）：

| 设置 | 默认 | 说明 |
|---|---|---|
| `voiceprints_dir` | `./voiceprints` | 声纹库落盘目录 |
| `asr_base_url` | `http://10.0.0.32:6001/v1` | Qwen3-ASR Base URL |
| `asr_model` | `models/Qwen3-ASR-1.7B` | ASR 模型 ID |
| `asr_api_key` | `EMPTY` | 任意非空 |
| `asr_timeout` | `60` | 单次 ASR 超时(s) |
| `match_threshold` | `0.5` | 余弦阈值（每模型可覆盖，见 §6） |
| `silence_gap` | `0.7` | 轮次静音收尾阈值(s) |
| `max_turn_seconds` | `20` | 单轮最长(s) |
| `ring_buffer_seconds` | `120` | PCM 缓冲窗口(s) |
| `max_asr_concurrency` | `2` | 每会话并发 ASR 上限 |

### 4.5 前端 `static/live.html`（新，路由 `/demo/live`）
保留现有 `/demo/`（上传分离 demo）不动。新页含三块：

- **麦克风采集**：`getUserMedia` → `AudioContext` + `AudioWorklet`（`ScriptProcessor` 兜底）拿 float32；按 ~0.5s binary 帧发 WS，query 带 `?sample_rate=<ctx 采样率>&embedding_model=...&segmentation_model=...`（服务端负责重采样到 16k，浏览器不做 JS 重采样）。
- **注册 UI**：录几秒或上传一段 → 填姓名 → `POST /v1/voiceprints`（带当前选中的 `embedding_model`）。
- **实时时间线**：渲染 `prediction`（带 `name`）与 `transcript`（带姓名+文字），ASR 结果可能乱序到达，按 `start/end` 落位。
- **模型切换**：两个下拉（embedding / segmentation），选项由 `GET /v1/info` 拉取；切换 = 断开重连新会话。显示「某人是在哪个模型下注册的」。

---

## 5. 数据流与时间轴对齐（正确性关键）

- PCM 环形缓冲按「累计输入样本数 ÷ 16000」索引，与 diart 的 `prediction.start/duration` **同一时间基**（会话推送的就是 diart 消费的 16k 音频，`sample_rate=16000` 无二次重采样）。因此轮次 `[start,end]` 能精确切出对应音频。
- 声纹匹配在**轮次收尾时**用该说话人**最新**预测的质心向量（最稳）。
- ASR 在后台异步执行，结果乱序到达；`transcript` 自带 `start/end`，前端据此落位。

---

## 6. 可切换模型带来的约束（必须遵守）

不同 embedding 模型的向量**不可互比**（如 ECAPA 192 维 vs WeSpeaker 256 维，空间不同）。因此：

- 声纹库**按 `embedding_model` 分命名空间**；匹配只在同模型内进行。
- 注册记录所用模型；切到未注册过的模型 → 相关人显示 `unknown` 并提示「该模型下未注册」。
- 阈值按模型各自配置（`match_threshold` 为默认，后续可加 per-model 覆盖）。

### 6.1 可用 embedding 模型（详见根目录 `MODELS.md`）

下载到 `./models`（见 §8）后，本机可用：

| 模型 | 后端 | 典型维度 | 状态 |
|---|---|---|---|
| hbredin/wespeaker-resnet34 (ONNX) | onnx | 256 | ✅ 最快/纯离线 |
| pyannote/wespeaker-voxceleb-resnet34-LM | pyannote | 256 | ✅ 第一梯队 fp32 |
| speechbrain/spkrec-ecapa-voxceleb | speechbrain | 192 | ✅ SV 精度最强之一 |
| speechbrain/spkrec-ecapa-voxceleb-mel-spec | speechbrain | 192 | ✅ |
| speechbrain/spkrec-resnet-voxceleb | speechbrain | 256 | ✅ |
| speechbrain/spkrec-xvect-voxceleb | speechbrain | 512 | ✅ 经典 x-vector |
| pyannote/embedding | pyannote | 512 | ✅ diart 默认，精度偏弱 |
| nvidia/…titanet_large | nemo | 192 | ⚠️ 需 `pip install nemo_toolkit[asr]`；默认在切换器中**隐藏** |

---

## 7. 并发与延迟

- diart 推理在既有后台线程（不变）。
- 每个收尾轮次起一个 asyncio ASR 任务，经 `asyncio.run_coroutine_threadsafe` 提交到事件循环；每会话并发上限 `max_asr_concurrency`（默认 2），超出排队。WAV 编码很轻可内联；网络 IO 走 `httpx` 异步。
- 声纹匹配是点积，内联即可。
- ASR 慢/超时不阻塞推理与 WS，只是 `transcript` 晚到。

---

## 8. 模型本地化（批准后第 0 步）

```bash
# .env 里 huggingface-key=hf_... → 导出为 HF_TOKEN
HF_TOKEN=<from .env> python docker/download_models.py ./models
# 运行：
DIART_MODELS_DIR=./models DIART_DEVICE=cuda uvicorn diart_server.app:app --host 0.0.0.0 --port 7007
```

- `download_models.py` 读 `docker/models.json`，把 10 个模型按 registry 期望的 `local_path` 布局下到 `./models`；多数已在 HF 缓存，基本是本地拷贝。
- 根目录写 **`MODELS.md`**：每个模型的后端/维度/速度/质量/额外依赖 + 如何在前端切换 + nemo/titanet 注意事项。
- nemo/titanet 文件能下，但运行加载需 `nemo_toolkit`；默认隐藏，需要时再装。
- `./models` 与 `./voiceprints` 加入 `.gitignore`（不入库）。

---

## 9. 错误处理

| 情况 | 处理 |
|---|---|
| ASR 不可达/超时 | 发 `transcript`，`text="[ASR失败]"`，会话不崩；记 metric + 日志 |
| 空 `embedding=[]`（聚类未初始化） | 跳过匹配，`name=null`（unknown） |
| 注册音频损坏/过短 | `POST /v1/voiceprints` 返回 400 |
| 选了未下载/不可用模型 | 沿用现有 `MODEL_NOT_FOUND` 错误码 |
| 选了 nemo 但未装 | 建会话即报错，引导隐藏/安装 |

沿用现有错误协议（`ErrorMessage` + 关闭码）。

---

## 10. 测试策略

- **单元**：
  - `voiceprint_matcher`：合成向量，验证 argmax + 阈值 + unknown。
  - `turn_aggregator`：合成预测序列，验证四种收尾触发。
  - `pcm_ring_buffer`：切片正确性、出窗边界。
  - `asr_client`：mock `httpx`，验证 WAV 编码 + `<asr_text>` 解析（含 `language None` 分支）。
  - `voiceprint_store`：落盘往返 + 按模型命名空间隔离。
  - `embedding_extractor`：dummy 模型，验证输出形状 + L2 归一化。
- **集成**：monkeypatch `asr_client` + 复用现有 dummy diart 模型，驱动 WS 跑一段 wav，断言出现带姓名的 `transcript`；声纹未注册时为 `unknown`。
- **手动**：浏览器真麦克风 + 真 Qwen3-ASR 端到端。
- 复用现有 `tests/diart_server/` 测试骨架与夹具。

---

## 11. 刻意不做（YAGNI）

- 不做鉴权 / TLS（内网假设，沿用现状）。
- 不做多用户声纹命名空间（按 embedding 模型分即可，单一共享库）。
- 不动 VAD pipeline。
- 不做跨会话/跨重启的匿名 re-ID（只认注册库 + 单会话内 diart 标签）。
- 前端够用即可，不追求精致。
- titanet/nemo 默认不启用。

---

## 12. 开放风险

1. **ASR 可达性**：`http://10.0.0.32:6001` 是否在线、返回格式是否如文档 → **批准后第一件事冒烟测**。
2. **阈值调参**：实时质心向量 vs 注册整段均值向量，最佳余弦阈值需实测；前端显示 `score` 便于调。
3. **时间轴漂移**：长会话中 diart 时间基与缓冲对齐若有系统性偏移，需在集成阶段校验切片音频内容正确。
4. **新字段破坏旧测试**：`prediction` 加 `name/score` 可能触动严格断言，需同步更新。

---

## 13. 落地顺序（供 writing-plans 细化）

0. 冒烟测 ASR + `download_models.py ./models` + 写 `MODELS.md`。
1. 纯逻辑模块 + 单测：`pcm_ring_buffer` / `voiceprint_matcher` / `turn_aggregator` / `voiceprint_store` / `asr_client`(mock)。
2. `embedding_extractor` + 注册端点 `voiceprint_routes`。
3. `session.py` 接线（缓冲 + 匹配 + 聚合 + 异步 ASR）+ 协议新消息。
4. 集成测试（假 ASR + dummy 模型）。
5. 前端 `live.html`（采集 + 注册 + 时间线 + 模型切换）。
6. 手动端到端联调 + 阈值调参。

# 声纹选型计划（测试系统 / 无状态 API）

- 日期：2026-08-06 · 基线 `main` @ `23ca40e` · 取代 [2026-08-06-voiceprint-upgrade-plan.md](2026-08-06-voiceprint-upgrade-plan.md)（其 §3 裁决与 §11 实测附录仍作为证据引用）
- 单测门禁：`python -m pytest tests/diart_server -q --ignore=tests/diart_server/test_e2e_websocket.py --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider` → `167 passed, 1 skipped`
- **数字标注**：【实测】亲自跑出并注明来源；【估计】我的推算；【引用】文献。

## 1. 目标与旧计划错在哪

**目标**：用 ≤7 人日让 `/demo/live` 在真人测试里"大概可用"，并交付一个无状态 `/v1/identify` 契约 + 一份带数字的选型裁决（模型 / 打分 / 判决策略 / 阈值）。

旧计划（33 人日 / 20 spec / 6 Phase）的根本错误是**把测试系统当生产系统规划**：鉴权、原子写、数据损坏窗口、迁移横幅、前向兼容白名单、CI 数值门禁、runbook——没有一项能让被命名率涨一个百分点。同时它**低估了两个真杠杆**（时长门、簇级累积），并在 7 说话人 / ~18 个同性别有序对的探针集上得出两个错误结论：「eres2netv2 与 campplus 统计不可分」和「FA=0」——都是统计功效不足的产物。

删除原则：**默认砍掉。每保留一项都要能一句话说清它如何让系统更快可用，或让选型结论更可信。**

## 2. "可用"的验收标准（一次 5 分钟真人测试，可当场判真假）

| # | 判据 | 判定 | 现状 |
|---|---|---|---|
| **V1 短句能认出** | 已注册者说 1~2 秒短句显示正确姓名 | 连说 10 句，**≥8 句带对名字** | ❌ 被命名率 14.0%【实测·§11】 |
| **V2 名字不跳变** | 同一人连续说话，姓名状态只变一次（无名→有名），之后不消失不换人 | 连说 12 句，变化 **≤1 次** | ❌ 2.7 次/12 轮【实测·设计B仿真】 |
| **V3 陌生人零人名（硬红线）** | **未注册者连说 2 分钟，屏幕不出现任何已注册者姓名**，也无 `≈某某 0.79?` | 出现任何人名 = 失败 | ❌ pyannote@0.82、N=5 约 45% 概率至少错一次【实测·审查】 |
| **V4 两人不串名（硬红线）** | A、B 均注册，轮流说 2 分钟，0 行把 A 的话署成 B | 出现串名 = 失败 | ⚠️ 未系统测过 |
| **V5 注册立即生效** | 录一段注册不重启，后续 3 句内出现姓名 | 3 句内有名 | ⚠️ 能生效但常被 V1 拖累 |

**优先级 V3 > V4 > V1 > V2 > V5。**「宁可显示『说话人 2』，也不显示错的名字」是第一优先级——V3 是用户自己看不见却最致命的失效。
**离线门禁数字**（每次改阈值必看）：未注册者**簇泄漏率 ≤ 1%**。

## 3. 现状诊断

| # | 原因 | 定量归因 | 修复 |
|---|---|---|---|
| ① | **`min_identify_seconds=2.0`**（`config.py:63`） | 天花板硬钉在 17.6%（探针集 ≥2s 占 49/278）【实测·设计B】；同阈值删门 14.0%→47.5%【实测·§11】 | T3 |
| ② | **身份模型是 2024 年的英语模型** | CN-Celeb1 196 人开集 `DIR@FAR1%`（N=20/2s/centered）：eres2netv2 **69.3%** / campplus 61.4% / cnceleb-r34 56.1% / pyannote-vox 53.1%\*【实测·设计C，审查已复现】 | T2+T3 |
| ③ | **阈值不随注册人数变**，0.82 对陌生人太宽松 | pyannote@0.82 未注册者逐 turn 误命名 N=3 **1.17%** / N=5 **1.96%** / N=20 **6.93%**【实测·审查】；2 分钟≈30 turn ⇒ N=5 约 45% 概率至少错一次【估计】 | T3 |
| ④ | **逐句判决，无会话内累积** | 196 人开集 12 turn/簇：逐句@0.82 簇内正确覆盖 N=5 **61.2%**/N=20 64.2% → 加粘滞 **96.5%/94.6%**（≈+30pp）【实测·审查】；且是 V2 唯一手段 | T4 |
| ⑤ | **前端 `≈某某 0.79?` 提示** | 不吃召回，但是 V3 唯一结构性失败源：未注册者出现率 **100%**、正确率 **0%**【实测·§11 T4】 | T1 |

\* pyannote 那行来自另一构造（句内滑窗）的 dump，仅作量级参考，不进裁决表。

**诊断**：不可用主要由三个工程决策造成——一道把天花板压到 17.6% 的时长门、一个英语身份模型、一个不随人数变的固定阈值；外加一个 100% 错的前端提示。修复都很便宜，但**必须按依赖顺序落地**。

## 4. 任务清单

**强制约束（来自审查 blocker，违反会让系统比现在更糟）**
- **B1 删时长门不得单独发布**，必须与「换 eres2netv2」或「pyannote 阈值抬到 0.86」同一 commit。单独删门 = 把「被命名率 47.5%」和「陌生人 45% 概率被安上名字」一起交付。
- **B2 换模型的硬前置是身份/聚类解耦**：`session.py` 用 `params.embedding_model` 同时构建 diart 在线聚类**和** `_identify_slice` 的 embedder，而 `delta_new=1.057`（`protocol.py:186`）是在 pyannote 256 维空间标定的；直接换 192 维模型会连聚类一起换掉。注册路由的 `embedding_model` 与 store 分桶键必须跟同一开关，否则「A 空间注册、B 空间识别」→ `store.vectors()` 空 dict → 识别全程静默 None。
- **B3 换模型必须与生成 cohort 均值同一 commit**：cohort `.npy` 缺失时服务端**静默禁用**识别，表现是"全部认不出"而不是报错。

| # | 任务 | 人日 | 用户看到什么 | 依赖 |
|---|---|---|---|---|
| **T1** | **前端去错**：删 `live.html:102-110` 的 `≈候选` 与 `lastVerified` 跨轮猜测，未确认一律渲染中性「说话人 N」（无分数、无人名）；`test_live_route.py` 4 条断言改反向。服务端零改动 | 0.25 | 陌生人说话不再冒出「≈吴泽 0.79?」。**V3 从必然失败变结构性通过** | — |
| **T2** | **通用 ONNX 后端 + 身份/聚类解耦**：搬 `scratchpad/asnorm_embed.py` 的 `OnnxEmbedder` 建 `speaker_onnx` 后端（fbank+CMN，元数据自适应，不硬编码张量名）；新增唯一真源 `DIART_VP_IDENTITY_MODEL`（默认 `None` = 沿用 `params.embedding_model`，零行为变更），同时驱动 ①`_identify_slice` embedder ②cohort 加载 ③per-model 阈值 ④注册路由与 store 分桶键 | 1.0 | 无可见变化（纯前置） | T1 |
| **T3** | **换模型 + 删门 + N 感知阈值（同一 commit）**：默认身份模型 → `3dspeaker/eres2netv2-zh-cn`（cohort 均值从现有 dump 的 `cohC` 组直接算，零前向）；`min_identify_seconds` 2.0→0.6 与 `min_candidate_seconds` 合一（候选层作废，`cand_*` 恒 None）；有效阈值 `lo(N)=0.82+0.0086·log2(N/5)` | 0.5 | **短句开始有名字**，被命名率 14%→【估计】65~75%；陌生人误报从不受控收敛到约 1% | T2 |
| **T4** | **簇级粘滞**（每个 diart 标签 3 个状态字段，≈50-60 行）：`hi=lo+0.04` 单轮强证据立刻确认，否则 `2-of-5` 同名投票；**争议锁**（同标签出现第二个不同姓名 → 该标签永久中性）；前端复用现有 `prediction` 的 `speaker+name` 回显做**回溯改写**（≈15 行 JS）；外加一行 partial 规则（见 D7） | 1.25 | **名字第 ~2 轮出现后不再消失、不再换人**，之前几行补上名字。**V2 达标** | T3 |
| **T5** | **离线扫描** `scripts/eval/offline_sweep.py`：复用 `scratchpad/asnorm_cache/emb_*.npz`（**零新前向，10 秒出表**），打印 §6 四张表 + **未注册者簇泄漏率一列**（`adv_cluster_check.py` 已是原型） | 0.5 | 改阈值/换模型有一个 10 秒可复跑的门禁数字 | T2 |
| **T6** | **无状态 API**：判决抽成无 I/O 纯函数 `identity.decide(scores, policy)`，WS 与 HTTP 共用；`POST /v1/identify`；`POST /v1/voiceprints` 扩展为同时返回 `vector_b64`；WS 新增 `roster` 控制消息（query string 冻结）；`space_fp` + 409（≈20 行）；协议删 `cand_name/cand_score`、加 `identity` 判别联合；注册改 `ENROLL_MS 6000→10000` + 非阻断 `warnings` | 1.25 | 一条 curl 就能在任意语料复跑同一套判决；换模型/阈值只改请求体 | T4 |
| **T7** | **验收**：`docs/acceptance-5min.md` 写 V1–V5 逐条脚本（V3 的「未注册者连说 2 分钟」为强制项）；跑真人测试；跑 T5 复核；跑单测门禁；顺手扫 `DIART_SILENCE_GAP` 0.7→1.0/1.2 | 0.5 | 「可用」有了当场可判真假的定义 | T6 |
| | **核心合计 5.25 ＋ 缓冲 1.75（T2 张量名探测、T4 回归）＝ 上限 7.0 人日**（10 人日预算内留 3 天） | | | |

T5 与 T2~T4 无代码交集可并行。测试系统不需要开关矩阵，`git revert` 就是回滚。

## 5. 无状态 API 契约

**边界定义（裁决 D5）**：**`/v1/identify` 与 WS `roster` 路径是权威的、严格无状态的契约——roster 随请求携带，这两条路径不读写 `VoiceprintStore`。** 现有 store（81 行，已在跑）保留为 **demo 页的注册便利层**：demo 从 `GET /v1/voiceprints` 取名单、自己组装 roster 发给 WS。守卫：一条单测断言这两条路径不触碰 `app.state.store`。

```jsonc
// GET /v1/info（扩展；工作点是选型的唯一权威）
{"voiceprint":{"models":[{"name":"3dspeaker/eres2netv2-zh-cn","dim":192,
  "space_fp":"vp1|3dspeaker/eres2netv2-zh-cn|cohort:a3f1c9d2|pre:2",
  "operating_point":{"lo_base":0.82,"lo_n_slope":0.0086,"hi_delta":0.04,
                     "min_identify_seconds":0.6,"vote":"2of5"}}]}}

// POST /v1/voiceprints  multipart: name, model, file[]
// 既写 store（供 demo），也返回向量（供无状态客户端持有）
{"name":"吴泽","model":"3dspeaker/eres2netv2-zh-cn","dim":192,"space_fp":"vp1|…","n_clips":2,
 "clips":[{"vector_b64":"hZ0…","net_speech_seconds":6.1}],
 "warnings":["clip[0]: net speech 6.1s below recommended 15s"]}   // 非阻断

// POST /v1/identify —— 纯函数：无副作用、不落盘、不可寻址
{"model":"3dspeaker/eres2netv2-zh-cn","space_fp":"vp1|…",
 "roster":[{"name":"吴泽","vectors":["hZ0…","kQ4…"]},{"name":"张三","vectors":["mR7…"]}],
 "segments":[{"audio_b64":"…"},{"audio_b64":"…"}],      // 同一个簇的全部片段一起送
 "policy":{"mode":"cluster","lo":null,"hi":null,"n_aware":true,"debug":false}}  // null=用服务端工作点
→ 200
{"operating_point":{"lo_effective":0.820,"hi_effective":0.860,"roster_size":2},
 "decisions":[{"index":0,"identity":{"status":"insufficient","reason":"net_speech_below_floor"}},
              {"index":1,"identity":{"status":"confirmed","name":"吴泽","score":0.871}}]}
// 未过阈值：{"status":"unknown"} —— 该对象没有 name 槽位，也没有 score
// debug:true 时顶层多 "debug":{"per_segment":[{"index":0,"gate":"below_threshold",
//   "scores":[{"name":"吴泽","score":0.712}]}]}
// 错误 400 BAD_AUDIO | 404 MODEL_NOT_FOUND | 422 ROSTER_INVALID | 413 ROSTER_TOO_LARGE(>200)
//     409 SPACE_MISMATCH（message 可操作，回传 expected_space_fp）

// WS /v1/diarize/stream —— query string 冻结（一个字段都不加）
// client→server（ready 之后任意时刻，replace 语义，生存期=连接；[] 表示关闭识别）
{"type":"roster","model":"…","space_fp":"vp1|…","speakers":[{"name":"吴泽","vectors":["hZ0…"]}]}
// server→client
{"type":"roster_ack","roster_size":2,"operating_point":{"lo_effective":0.820}}
{"type":"transcript","speaker":"speaker0","turn_id":"…","partial":false,
 "identity":{"status":"confirmed","name":"吴泽","score":0.871}}   // cand_name/cand_score 已删
```

**三条协议级不变量**
1. **`Identity` 是判别联合**：`unknown` / `insufficient` 里**物理上不存在 `name`、`score` 字段**（不是置 None）。要泄漏姓名必须显式构造 `IdentityConfirmed`——没有"顺手把最像的人带出去"的路径。
2. **`unknown` 与 `insufficient` 对用户渲染成同一个中性标签**；区分只对 debug 有意义。客户端不得让用户区分两者——区分会诱导脑补。
3. **`space_fp = "vp1|<model>|cohort:<sha8>|pre:<n>"`，不匹配就 409。这不是可选项**：【实测·审查】eres2netv2 与 campplus **都是 192 维**，而 `voiceprint_matcher.py:101` 唯一防护是 `r.shape != u.shape`——用 campplus 向量打 eres2netv2 的分会**静默算出一堆无意义的高分**（旧计划 R4 标为灾难性）。现状靠 store 的 `manifest[model]` 分桶挡住，roster 走请求体后这道墙没了。`pre:<n>` 是手工递增常量，配一条单测：预处理函数（`_trim_silence`/重采样/归一化）源码 hash 变化时该常量必须变。

**X-Tuning 警告**（写进契约）：`policy.lo/hi` 的客户端覆盖是选型刚需（扫阈值出曲线），但它让任何调用方能把阈值降到 0 逼出名字。测试系统里是特性；若这份契约被当生产版继承，这条必须删除或加鉴权。

## 6. 选型方法

### 6.1 四道选择题与裁决

| 题 | 裁决 | 依据 |
|---|---|---|
| **Q1 嵌入模型** | **eres2netv2-zh-cn**；campplus 留 manifest 作 env 可切备选 | 【实测·设计C，审查已原样复跑复现】196 人开集，eres2netv2 在 **4 个注册库规模 × 3 种打分 = 12 个独立条件下全部领先 campplus 6~8pp**，delta 远超抽库离散度。旧计划「统计不可分」**被推翻** |
| **Q2 打分** | **保持 cohort 中心化**；AS-Norm 带门推迟；**QMF 砍** | raw vs centered 差 0.6pp【实测·设计C】= 噪声，而现役阈值都挂在 centered 刻度上 ⇒ 零收益+有成本=不动。AS-Norm 去泄漏 +4.4pp 但需 ≥200 人域外 cohort（20GB）。QMF 在 FA=0 操作点 50.0%，输给零成本对照 56.1%；C_llr 判别损失占 73.5%【实测·§11】——**它已经输了** |
| **Q3 判决策略** | **簇级粘滞**（hi/lo + 2-of-5 + 争议锁 + 回溯改写） | 【实测·审查】簇内正确覆盖 61~64% → 94~96%（≈+30pp），比换模型（+6~8pp）和 AS-Norm（+4.4pp）加起来还大；且是 V2 唯一手段。成本 50-60 行 |
| **Q4 阈值** | **`lo(N)=0.82+0.0086·log2(N/5)`，`hi=lo+0.04`** | 【实测·设计C】`thr@FAR1%` N=5 0.7890→N=50 0.8177，斜率 +0.0086/翻倍，与旧计划 T7 的 0.0087~0.0092 一致。【实测·审查】该操作点 N=5 簇覆盖 96.5%/泄漏 **0.0%**；N=20 88.5%/**0.2%**；固定 0.82 在 N=20 泄漏 1.5% |

**争议锁的作用域必须写死**：它是 **diart 混簇守卫，不是未注册者守卫**。【实测·审查】它在 impostor 簇上触发率 **0.0%**——陌生人的 top-1 最近邻是其嗓音的稳定属性，**53.8% 的 turn 命中同一个错误姓名**（N=20 随机水平仅 5%）；`2-of-5 同名`对 impostor 近乎免费通过（组内相关 ρ≈0.81 ⇒ 5 样本有效独立样本数 n_eff≈1.2）。**未注册者的唯一守卫是阈值**，所以 `lo` 必须按开集 FAR 定，不能按探针集定。

### 6.2 最小评测（T5，0.5 人日，零新前向）

复用 `scratchpad/asnorm_cache/emb_{eres2netv2-zh,campplus-zh,cnceleb-resnet34-LM}.npz`（CN-Celeb1 eval 196 人注册向量 + 1563 句 ×{2s,4s,整句} + 764 cohort 片段，三模型同批音频同批切片全部已嵌入），0.1 秒/模型。输出：模型×打分、阈值×N、k 累积、**未注册者簇泄漏率**四张表。
**不做**（旧计划 P0-7 全砍）：三段分离 + `trials_hash` 血缘、`eval_metrics.py` + 12 条单测、speaker-block bootstrap 实现、`gate.py`、DET 曲线、性别/genre 分层。**选型只要排序和斜率**，稳健性由「12 个独立条件方向一致」提供，比一条 CI 更有说服力。

### 6.3 判定规则（先写死，跑完不许事后调）

- **R1 硬红线一票否决**（优先于任何覆盖率）：真人验收「未注册者连说 2 分钟出现 ≥1 个人名」= 失败；离线未注册者簇泄漏 > 1% = 失败。不满足则 `lo` 整体上抬 0.01 重跑。
- **R2 模型**：某模型在 ≥3 个注册库规模（N∈{5,10,20,50}）领先 ≥3pp → 直接选，不看延迟。**已满足**。打平（全部 |Δ|<3pp）时按延迟选。
- **R3 AS-Norm 准入**（三条全满足才做）：(a) 去泄漏 ≥+3pp【已满足 +4.4pp】；(b) cohort ≥200 人且与评测集不相交【未满足】；(c) 硬红线不上升【未测】⇒ **推迟到第二轮**。
- **R4 打平顺序**：**少一个产物 > 低延迟 > 保持现状**。「少一个产物」= 不新增必须随代码同步的二进制文件——现役 `assets/cohort` 就是活例子（容器里从没被 COPY 进去，声纹全程静默禁用）。
- **R5 两个都留比选出来划算**：eres2netv2/campplus 都留（`DIART_VP_IDENTITY_MODEL` 切换）；raw/centered 都留（env）；per_turn/cluster 都留（`policy.mode`）。**时长门、QMF、`cand_name` 不留**——留着 = 留着一个已被证伪的默认值。
- **R6 第二轮触发器**：V4 失败 ⇒ 正确反应是**换前端（Streaming Sortformer）**，而不是在坏前端上再叠守卫。【引用】DIHARD III 1s：diart 27.6 vs Sortformer 13.3–18.9 DER；`nvidia/diar_streaming_sortformer_4spk-v2` 已在本机 HF 缓存，接入【估计】2~3 人日。

### 6.4 置信边界（这套方法与本计划**不能**支持什么）

1. **不支持绝对操作点。** CN-Celeb 不是 Edward 的麦、房间、浏览器；它只给**排序**和**斜率**，阈值截距必须由真人验收校准。
2. **绝对准确率偏乐观。** 3D-Speaker 用 20 万中文说话人训练，CN-Celeb 很可能在训练集里。相对排序受影响较小（eres2netv2 与 campplus 同出一家、互比公平），但**它们对 wespeaker 系的优势幅度可能被高估**。
3. **不支持任何 FAR < 1% 的结论**（30 次抽库、每次约 140 个库外说话人）。**绝不要再引用 278 探针集上那个「FA=0/1668」**——rule of three 下它只证明 FAR ≤ 0.18%，而同一阈值在 N=20 的真实 FAR 是 7.0%【实测·审查】。那个 0 是 ~18 个同性别说话人对造成的假象。
4. **不支持带误差棒的陈述**（报的是抽库离散度，非 speaker-block CI）：可说「6~8pp 的一致差距是真的」，不可说「好 6.3 ± X pp」。**结论只在 N ≤ 8 时宣称**。
5. **不解决 <1 秒中文同性别短句的物理极限。**【引用·survey】EER 随时长每减半翻 2~2.4 倍，中文比英语难一个数量级，同性别再乘 2.4，行业无人在 <5s 上承诺身份。V1 定的是「1~2 秒句 8/10」，不是「每句都对」。**若"可用"的期待包含每句都认出来，本计划全做完仍会不满意——需在开工前对齐。**
6. **不解决 diart 前端代差**（混簇错名残差主要来自这里，见 R6）；**不解决大注册库 N≥20 的误报**；**完全未覆盖**噪声/远场/电话信道、非中文、跨天信道漂移、真实会议的打断与抢话动态。
7. **WS `roster` 的端到端行为没有自动化覆盖**：3 个 WS 测试是历史 flaky，永久排除在验证外，只能人工冒烟。**`onnxruntime` 本机只有 CPU EP**、`EMBED_EXECUTOR` 单 worker 与注册共用：单路 demo 无碍，多路并发会排队（缓解：切 campplus 或提到 2 worker）。

## 7. 明确砍掉的东西

| 砍掉 | 旧工时 | 理由 |
|---|---|---|
| P0-1 docker 修复 + assets 打包 + G3 容器门禁 | 1.4d | 实测部署是直接 `uvicorn`。**推迟，不是修复。**（保留 B3：`verify_speaker_cross_matrix.py:199-212` 把 import 失败当 `SKIPPED` 并 `return 3`，5 行顺手修）|
| P0-2 契约冻结 + `QueryParams` 的 `x_` 白名单（D12、R14） | 0.8d | query string 冻结、roster 走消息体 ⇒ 新前端连旧服务端时消息被「未知控制类型静默忽略」吃掉 = **降级而非断线**，兼容零代码成立 |
| P0-4 store 原子写 + version + sidecar + 损坏窗口（R12、D11） | 1d | 声纹是随时可 40 秒重录的测试数据；为它做原子写 = 为草稿纸做备份 |
| P0-7 评测底座三段分离 + `trials_hash` + `eval_metrics.py` + G4 CI 数值门禁 | ~5d | 生产级评测基础设施；我们只需**一次裁决**，排序已被 0.3 秒的 numpy 脚本定完 |
| P0-8 留出集录制（10 人 / 2.5h 人力） | 人力 | **推迟到第二轮（gated）**：5 分钟真人测试 + 一个未注册者说 2 分钟能廉价拿到 80% |
| P2-1 `audio_quality.py` + SNR 门；P2-2 六道门控 + 20 秒净语音 + **音频留存/rebuild/隐私权衡** | ~4d | 质量门不值：第 2 条注册只值 +2.2/+6.0pp【实测·设计B】；留存是为免于全员重录，而全员 = 2~3 人 × 40 秒。**同时否决设计 B 的 `MIN_SPEECH_SECONDS 1.5→4.0`**——注册是链路里唯一不能失败的一步，不该为 2pp 更容易 400 |
| P3-2 迁移横幅 + 全员重录提示 + 一键重建（R6） | 1.5d | 作废声纹的代价是重录 2 条约 40 秒；旧向量带 `space_fp`，不匹配就 409 |
| P4-1 voiceprint pack（weights sha256 + 张量名 + 原子绑定 + fail-closed 加载器） | 1.5d | 要防的 R4 风险没变，但只需 **20 行 `space_fp` + 409** |
| P4-2 AS-Norm 四后端 A/B；P4-3 QMF / LLR 校准 | 5.5d | AS-Norm 降为带门尾项（R3）；QMF **已实测证伪**：FA=0 操作点 50.0% vs 零成本对照 56.1%，C_llr 判别损失 73.5% ⇒ 校准最多吃 26.5% |
| P5-1 ρ 收缩 / SPRT / anchor 检测器；P5-2 epoch 协议 / 三态词表 / debug 面板 | 7d | 被 T4 的 50-60 行取代。回溯改写复用现有 `prediction` 回显（已确认 diart 簇标签会话内**不会被回收复用**，`active_centers` 只增不删，粘 label 安全）；pending 与 unknown 对用户是同一个动作 |
| P6-1 九场景 + 性别/genre 分层；P6-2 runbook + 开关矩阵 + ~50 个 `DIART_VP_*`；鉴权/多租户/SLA | 3.3d | 没有运维对象、没有真实用户；本计划只新增 3 个环境变量。分层的价值是找最难负样本，真人测试里排一组同性别近亲对更便宜 |
| D6 roster 先验作为独立功能；D7 用户纠错回流 | 含 P1-1 | roster **就是请求体**，`N=len(roster)` 直接摆在服务端眼前——把高影响的 R2 降级成协议内建属性。纠错回流服务于已砍掉的标定链路，**推迟不砍死** |
| 设计 C 的 C3「turn 内全长」独立工作项 | 0 | **它不是一个开关**：`_apply_overlap`（`session.py:533-556`）返回的 `identity_pcm` 本来就是全部非重叠语音的拼接；C 表里 2s/4s/整句的差异是 CN-Celeb 裁剪效应。真正能动的是 `DIART_SILENCE_GAP`（T7 顺手扫） |
| 设计 A 的彻底无状态化（拆 store + demo localStorage + 导出导入） | 2.4d | 见 D5 |
| Streaming Sortformer 前端评测 | 2~3d | 换前端不改善「陌生人被盖名」；改为 R6 触发条件 |

**关键裁决（我推翻或修正了输入的地方）**

- **D1 否决设计 A 的 S0「一步降到 0.6、不需灰度」**：单独落地会把「更可用」和「什么都敢认」一起交付。删门降级为 T3 的一部分；若 T2/T3 拖延而想先看效果，唯一允许的中间态是「删门 + pyannote 阈值 0.82→0.86」（【实测·审查】N=5 逐 turn FAR 1.96%→0.05%）。
- **D2 推翻设计 A 砍掉 Phase 5**：**簇级粘滞加回来**（换模型后最大的单点杠杆 +30pp，且是 V2 唯一手段）；但继续砍掉 ρ 收缩、SPRT、anchor 检测器、epoch 协议、三态词表、debug 面板。
- **D3 修正设计 B 的争议锁**：保留为**混簇**守卫（混簇错名行 50%→14.3%【实测·设计B】），但文档写死「不是未注册者守卫」，投票下限改为按开集 FAR 定的 `lo(N)`。
- **D4 修正设计 C**：补上身份/聚类解耦硬前置（+0.5d 不可压缩）；E3 录音 + E4 会话打分器（2d）推迟到第二轮，那 2 天投给簇级粘滞与真正的端点实现——C 自己承认合成会话的 `cluster_purity` 是乐观上界、4 个说话人不能用于排序。
- **D5 修正设计 A 的无状态边界**：**`/v1` 契约无状态，`VoiceprintStore` 保留为 demo 便利层。** 用户要的是「一个无状态的 **API**」，那是 `/v1` 契约的属性，不是 demo 的属性；被砍掉的原子写/鉴权/迁移/pack 是因为「测试系统」被砍的，不是因为「无状态」。T6 因此 2.4d→1.25d，设计 A 自造的「清 localStorage 丢全部声纹」风险消失，且 store 的 `manifest[model]` 分桶顺带成为 `space_fp` 之外的第二道防线。
- **D6 修正设计 A 对旧计划 D9 的推翻**：**只在 `/v1/identify` 响应与 WS `transcript.identity` 引入判别联合，`_last_match` 仍保持 `(name, score)` 二元组**——T4 的回溯改写本来就需要这个形状来喂 `prediction` 回显，而 WS 路径的姓名泄漏源是 `cand_name/cand_score`，删掉它们就消除了。15 处现存断言零改动，省【估计】0.5 人日。**这不是"为不改测试而扭曲 API"，是泄漏面已被另一条路径消除后不必要的破坏性改动。**
- **D7 新增（三份设计都漏了）**：**partial 与 final 的身份会互相打架。** 删门后 partial 也带名字，而 `_transcribe_and_emit` 对 partial/final 走同一路径且都写 `_last_match`（`session.py:723-727`）：同一 turn 的 partial（较短音频）可能命中名字、final（更长）反落回 unknown，前端按 `turn_id` 原地替换 ⇒ 名字闪一下又消失。这是与簇级粘滞正交的闪烁源，V2 覆盖不到。修复：**身份只在 `partial=false` 盖章，partial 继承该标签当前粘滞名且不写 `_last_match`**（3 行，并入 T4）。

## 8. 需要 Edward 拍板的问题

1. **默认模型选 eres2netv2 还是 campplus？** **推荐 eres2netv2**：12 个独立条件下领先 6~8pp。campplus 快 4 倍（4 线程 2s：9.4ms vs 39.5ms），但 40ms 对一个每几秒才出一个 turn 的系统毫无意义——**为 40ms 放弃 6~8pp 是荒谬的交易**。campplus 留 manifest 作 env 备选（零成本）。**只有你要在 CPU 上跑多路并发时才该反过来定。**
2. **接受「`/v1` 契约无状态 + demo 保留 store 便利层」这个边界吗？**（vs 彻底删 store + demo localStorage）**推荐接受**：省 1.15 人日、消掉自造的"清缓存丢声纹"风险，而你要的无状态 API 一分不少。代价是未来若要把 demo 也做成纯客户端，那部分工作还在。
3. **确认误报红线与注册人数上限。** **推荐**：离线「未注册者簇泄漏 ≤1%」+ 真人「未注册者连说 2 分钟 = 0 个人名」为硬红线，**结论只在 N ≤ 8 时宣称**。需要你确认接受「探针集 FA=0 是小样本假象」这个前提。
4. **若 V4（两人对话串名）失败走哪条路？** **推荐先接受「混簇时不显示名字」**——争议锁已把失败模式从"错名"退化成"没名字"，符合 V3>V1 的优先级；换前端（Sortformer，【估计】2~3 人日）留到第二轮，因为它不改善「陌生人被盖名」这个更致命的问题。

**推迟到第二轮（gated，不砍死）**：4 人真人留出集 + 合成会话打分器（触发：红线失败或需本机信道校准）、AS-Norm（触发：R3 三条全满足且 T7 后覆盖仍不够）、Sortformer 前端（触发：R6）、用户纠错回流。**本计划的诚实边界见 §6.4。**

# 实时声纹识别调研:物理极限、行业实践与升级路线

- **日期**:2026-08-06
- **动机**:live demo 实测体感差("陌生人 0.8 分"、短句无法识别、身份逐句跳变),需要回答:这是物理极限还是实现差距?别人怎么做?
- **方法**:四路并行调研(学术流式 diarization SOTA / 短语音打分工程 / 商用产品规格 / 中文生态与开源)+ 本机实测(278 探针分数 dump、5 模型 A/B、预处理消融)。

---

## 1. "实时声纹就这么难吗?"——难的部分是物理极限,量化如下

**时长定律:时长每减半,EER 翻 2~2.4 倍。**
VoxCeleb 上 ECAPA:full 2.05% → 2s 5.39% → 1s 12.82% → 0.5s 26.83%(接近抛硬币)
([arXiv:2506.14226](https://arxiv.org/abs/2506.14226));SOTA ERes2NetV2 也从 full 0.61% → 2s 1.48%
([arXiv:2406.02167](https://arxiv.org/abs/2406.02167))。

**中文域整体比英语难一个数量级。** 同一批模型 VoxCeleb1-O 0.6–0.8% EER,CN-Celeb 6.0–7.7%
([3D-Speaker toolkit, arXiv:2403.19971](https://arxiv.org/abs/2403.19971));CNSRC 2022 冠军也只到 4.9%。

**同性别+同语种冒充者再乘 ~2.4**(Vox1-O vs Vox1-H 的构造差)。三项复合:中文、同性别、2 秒片段,
EER 外推 ~14%;1 秒 ~30%。**我们实测的"1.5s 以下 genuine/impostor 分布重叠"完全符合物理规律,不是 bug。**

文献直接表态:"when T < 2s, TI (text-independent) remains inferior to TD"
([Hybrid Enrollment 2026](https://arxiv.org/html/2606.16115));**没有任何生产系统在 <5s 音频上承诺身份**
(AWS Voice ID 判决窗口最低可配 5s 且明示牺牲精度;Gong 对发言太少者干脆不显示 speaker track)。

## 2. "别人怎么做?"——三个跨学术/工业的一致共识

### 共识一:身份是会话内累积的状态,不是逐句判决

- **AWS Connect Voice ID**:注册 30s 净语音;判决窗口默认 10s;流式聚合音频,结果随新增音频更新
  ([docs](https://docs.aws.amazon.com/connect/latest/adminguide/voice-id.html))
- **Pindrop/Nuance**(呼叫中心反欺诈):通话前几秒出首个风险分,之后**持续累积多因子证据**
- **学术**:在线 TS-VAD 滚动更新说话人 profile([arXiv:2207.05920](https://arxiv.org/pdf/2207.05920));
  Streaming Sortformer 的 AOSC 只保留高分帧进入说话人缓存([arXiv:2507.18446](https://arxiv.org/abs/2507.18446));
  微软 t-vector 流式 SA-ASR 检测到说话人切换后**延迟 2–8 词再定身份**,delay=2 最优
  ([arXiv:2203.16685](https://arxiv.org/abs/2203.16685))
- **TST-Bench(2026,与我们任务定义完全相同的公开基准)**:实验证明"朴素 diarization+声纹验证串联"
  显著劣于专门设计([arXiv:2606.14091](https://arxiv.org/pdf/2606.14091))

### 共识二:生产级打分后端 = AS-Norm → 时长感知校准(QMF)→ LLR

VoxSRC 2020–2023 每一年冠军的标配链路([VoxSRC Retrospective](https://arxiv.org/abs/2408.14886)):

1. **AS-Norm**(top-K≈200–300 最像的冒充者做 per-trial 均值方差归一,
   [Matejka 2017](https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html)):
   我们的"减全局 cohort 均值"是它的零阶退化版——全局均值被异性别/不像的人主导,而 AS-Norm 的
   cohort 统计天然由"同性别同语种最像的那批人"决定(实测 92% 同性别),恰好压制我们最痛的区间。
   典型增益:域匹配 -8~16% EER,跨域/中文/短时长 -20~30%。
2. **QMF 校准**:`llr = w_s·s + w_q^T·q + b`,质量向量 q 含 log(时长)、‖emb‖、SNR、impostor 均分
   ([IDLab VoxSRC-20](https://arxiv.org/abs/2010.11255)、[DKU-MSXF](https://arxiv.org/abs/2308.08766))。
   **原始分域的有效阈值自动变成时长的函数**——短片段自动抬门槛,上层只见一个固定 LLR 阈值。
   这是"per-model × per-duration 阈值矩阵"问题的解析解。AS-Norm 之后再叠 QMF:-11~27%。
3. **LLR/后验概率**:给用户看 `P = sigmoid(llr + logit(prior))`,"陌生人 0.8"变成"属于吴泽的概率 12%"。
   LLR 可加性支撑跨句累积(SPRT 三段判决:确认/继续攒证据/未注册),同一簇 n 句的 LLR 和
   等效于一次 n 倍长的观测——这是绕过短句物理极限的唯一正道。

PLDA 已退场(cosine 全面胜出 27–35%,[arXiv:2211.00815](https://arxiv.org/abs/2211.00815));
但注意 [VoxWatch](https://arxiv.org/abs/2307.00169) 警告 AS-Norm 对强模型可能有害,必须自测 A/B。

### 共识三:UX 规范(全行业一致,我们全部违反)

| 规范 | 行业做法 | 我们现状 |
|---|---|---|
| 注册量 | 20–30s 净语音或多 clip(AWS 30s/Azure 20s/Speechmatics 5-30s×多条);Eagle 用**进度条到 100%** | 6s 单条 |
| 音频不足 | 一等状态(AWS `NOT_ENOUGH_SPEECH`),不是低分 | 显示 ≈候选+分数 |
| 原始分 | **从不给终端用户看**;开发者见校准分+离散决策 | 直接显示 cosine |
| 未注册者 | 中性标签(Speaker 1/S2),**从不显示"最像的人"** | ≈吴泽 0.79? |
| 候选集 | roster 先验(Teams 限受邀 ≤50 人) | 全库 1:N |

注册人数警告:开集识别的误报随注册人数**急剧**上升(watchlist 50→500 人,FAR@FRR=5% 涨 7 倍,
[VoxWatch](https://arxiv.org/abs/2307.00169))——注册人越多,AS-Norm 越不可省。

## 3. 模型选型结论(中文)

**换 `iic/speech_eres2netv2_sv_zh-cn_16k-common`(3D-Speaker,200k 中文说话人训练,Apache-2.0)**:

| 模型 | CN-Celeb EER | 短段实测 d′(1s) | CPU 延迟/1s |
|---|---|---|---|
| wespeaker cnceleb-resnet34-LM(现役) | 6.65% | 3.38 | 8.6ms |
| **ERes2NetV2 zh-cn 200k** | **3.81%**(-43%) | **4.06** | 20.9ms |
| CAM++ zh-cn 200k(延迟敏感备选) | 4.32% | 3.63 | 6.0ms |
| ERes2Net-large zh-cn(GPU 极限档) | 2.79% | 4.24 | 90.8ms |

现成 ONNX 直接从 [sherpa-onnx release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) 下载。
**注意坑**(本机消融实测):
- CAM++/ECAPA 之前出垃圾的真因是 **ONNX 张量名不匹配**(pyannote wrapper 硬编码 `feats`/`embs`,
  3D-Speaker 导出是 `x`/`embedding`),不是 CMVN——pyannote 的 wrapper 确实做了 CMN,与 wespeaker
  官方 infer_onnx.py 逐字一致。需要写一个**读元数据的通用 ONNX backend**(~30 行)。
- **CMN 永远要开**(margin 差 0.09–0.17,决定性);×32768 缩放和窗型无所谓(被 CMN 抵消)。
- sherpa-onnx 对 wespeaker 模型**漏做 CMN**(元数据缺 `feature_normalize_type`),勿参照其阈值经验。
- 训练集与 CN-Celeb 可能有说话人重叠,3.81% 或有高估,**上线前必须用自己的留出集复测**。

**排除项**:ReDimNet(中文档 7.47%,比现状差)、VoxBlink2 系(无中文检查点)、FunASR(只有会话内
聚类,无 enrollment/身份绑定)、WavLM/SSL(推理贵一个量级)、商业 API(都不支持流式,规格比需求窄)。

**前端(diarization)**:diart 的 DER 已被代差超越(DIHARD III 1s 延迟:diart 27.6 vs
Streaming Sortformer 13.3–18.9,[HF 模型卡](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2),
CC-BY-4.0,≤4 说话人)。中期值得评测替换;>4 人场景看 LS-EEND。

## 4. 升级路线图(按性价比)

**Phase 1(打分后端 + 模型,~1 周,预期合计相对 EER -50~60%)**
1. 通用 ONNX backend(读元数据、修 IO 名、保 CMN)+ 换 ERes2NetV2 zh-cn(-43% EER)
2. AS-Norm(top-K 200–300;cohort:CN-Celeb dev 1000 人每人一条均值嵌入 + 现有 2news/AISHELL)
3. QMF 时长校准:用现有 278-probe 分数 dump 训逻辑回归(特征:score、log 时长、‖emb‖、impostor 均分),
   输出 LLR;先算 C_llr vs C_llr^min 量化"校准损失 vs 判别损失"
4. UI:显示校准后验概率,不再显示原始 cosine

**Phase 2(决策架构,~1 周,主要改善体感)**
5. 簇级 LLR 累积 + SPRT 三段判决(锁定/识别中…/未注册);<1.2s 片段只贡献证据不单独判;
   margin 门:top1-top2 的 LLR 差 > 3.0(≈95:5 赔率)
6. 注册协议:≥15s 净语音进度条(Eagle 式百分比+质量反馈)、多条 length-norm 平均、质量门控
   (净语音/SNR/‖emb‖/impostor 均分/多条互相一致性)

**Phase 3(可选)**
7. Streaming Sortformer 前端评测;roster 先验(会前声明在场者);被确认用户的高置信片段回灌声纹
   (EMA+锚点防漂移);TST-Bench 对标;分层指标(时长桶 × 同/异性别 + DIR@FAR)进回归测试

## 5. 关键参考(全文 URL 见各节内联)

学术:Streaming Sortformer([2507.18446](https://arxiv.org/abs/2507.18446))、LS-EEND
([2410.06670](https://arxiv.org/pdf/2410.06670))、t-SOT+t-vector([2203.16685](https://arxiv.org/abs/2203.16685))、
TST-Bench([2606.14091](https://arxiv.org/pdf/2606.14091))、PET-TSVAD([2309.12521](https://arxiv.org/abs/2309.12521))。
打分:Matejka 2017(AS-Norm)、IDLab VoxSRC-20/21(QMF)、Ferrer&Riera([2102.01760](https://arxiv.org/abs/2102.01760),
校准诊断)、VoxWatch(开集/注册人数)。
商用:AWS Voice ID、Teams voice profile、Speechmatics Realtime Speaker ID(与我们形态最接近的公开对标)、
Picovoice Eagle、pyannote.ai。
中文:3D-Speaker([2403.19971](https://arxiv.org/abs/2403.19971))、CN-Celeb、wespeaker cnceleb recipe、
sherpa-onnx speaker-identification。

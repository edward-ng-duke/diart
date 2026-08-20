# diart 在静音 ≥5s 后把两个人并进同一个标签，标签级身份回显随之串名

- **Date:** 2026-08-20
- **HEAD at investigation:** 1fb7baa
- **Status:** REPRODUCED（真实模型 + 真实 cohort，cuda；探针脚本见 §3）。
  下面所有数字都是在这个 HEAD 上亲自跑出来的，标【实测】。

## 1. Symptom

A 说话 → 静音 6 秒以上 → B 说话，**B 被打上 A 的 diart 标签**。
后果分三层：

1. `PredictionMessage.name` 把 A 的姓名回显到 B 的音频区间上
   （`session.py:_on_prediction` 读 `_last_match[speaker]`，而 `_last_match`
   **从来不会被清除**）。`/demo/live` 目前不渲染 `prediction`，所以演示页看不见，
   但这是 WS API 上的一条 V3/V4 红线泄漏，任何其他客户端都会中招。
2. 两个人在时间线上都显示成同一个「说话人 N」。
3. **这是 T4「簇级粘滞」的前置阻断项**：一旦按计划让姓名在标签上粘住，
   这条泄漏就会从「API 里有」变成「屏幕上写着吴泽」。

## 2. Root cause

### 2.1 窗级判决处在 genuine/impostor 分布重叠区

diart 每 0.5 秒用**一个 5s 窗**的嵌入判决「这是不是新人」。
【实测】同一对说话人、生产打分刻度（cohort 中心化 + `(cos+1)/2` 重映射）下，
同人 / 异人分数随切片时长的分离度：

| 切片 | genuine min | impostor max | gap |
|---|---|---|---|
| **0.5s** | 0.589 | 0.769 | **−0.180 重叠** |
| 1.0s | 0.726 | 0.715 | +0.011 |
| 1.5s | 0.732 | 0.703 | +0.029 |
| **2.0s** | 0.784 | 0.672 | **+0.113** |
| 2.5s | 0.865 | 0.664 | +0.202 |
| 4.0s | 0.907 | 0.642 | +0.265 |

窗级（每步 0.5s 新证据）落在**重叠区**——**没有任何 `delta_new` 能在那里工作**。
这不是调参问题。

### 2.2 `delta_new=1.057` 饱和，新人只能靠一对一指派约束被发现

【实测】同窗内两个确定不同的人的余弦距离：`wespeaker` max **1.047**、
`pyannote/embedding` max **1.065**、两个空间 p50 都 ≈0.90。
`delta_new=1.057` 压在整个跨说话人分布之上 ⇒
[`clustering.py:168`](../../src/diart/blocks/clustering.py) 的
`unmap_threshold(delta_new)` 分支实际从不触发。新质心只剩一条产生路径：
同一个 5s 窗里**同时活跃**的局部说话人多于现有质心数
（`identify()` 的一对一指派约束）。静音 ≥ `duration=5.0` 时没有任何窗同时装下两人 ⇒ 合并。

### 2.3 调低 `delta_new` 会在真实音频上把聚类打碎——**这条路是死的**

【实测】两侧扫描（`rho_update` 0.006/0.05/0.10 三档结果**完全相同**，无影响）：

| delta_new | handoff-6s/10s | alt-3s | **2news.mp3（真实）** | merge 失败 | split 失败 |
|---|---|---|---|---|---|
| 0.70 | 2 标签 ✅ | 3 标签 | **5 标签** ❌ | 0 | 3 |
| 0.75 | 2 标签 ✅ | 3 标签 | **5 标签** ❌ | 0 | 2 |
| 0.80 | 2 标签 ✅ | 2 标签 | **4 标签** ❌ | 0 | 2 |
| 0.85 | **1 标签** ❌ | 2 标签 | **4 标签** ❌ | 2 | 0 |
| **1.057（现状）** | **1 标签** ❌ | 2 标签 | **2 标签 ✅** | 2 | 0 |

`1.057` 是这组坏选项里最好的一个：真实音频上说话人数正确、纯度 99.7%。
调低它把「静音后合并」换成「真实音频上多出 2~3 个假说话人」，净损失更大。
**裁决：`delta_new` 不动。**

## 3. 为什么不能用「turn 级相似度阈值」

第一版设计是：每个标签记住上一条 turn 的嵌入，新 turn 与之比对，低于阈值就判易主。
**这个方案被自己的验证脚本否了。**

关键在 genuine 集合怎么取。拿**同一段连续话音的相邻切片**比，看起来干净可分
（2.0s 处 gap +0.113）——但那不是线上场景。线上场景是**跨段**：同一个人、
相隔几十秒的两条 turn。`scripts/verify_label_continuity.py` 只用**不同段落**
之间的配对构造 genuine：

| 切片 | genuine min | genuine p5 | impostor max | gap |
|---|---|---|---|---|
| 1.0s | 0.569 | 0.638 | 0.824 | **−0.255** |
| 1.5s | 0.577 | 0.623 | 0.839 | **−0.263** |
| 2.0s | 0.570 | 0.619 | 0.813 | **−0.242** |
| 3.0s | 0.587 | 0.619 | 0.823 | **−0.236** |

两个分布在每个 turn 长度上都**完全重叠**。可用阈值不存在：

| 阈值 | 0.62 | 0.65 | 0.68 | 0.70 | 0.72 |
|---|---|---|---|---|---|
| 2.0s 误判 / 检出 | 6% / 28% | 10% / 66% | 16% / 88% | **21% / 94%** | 24% / 97% |
| 1.0s 误判 / 检出 | 3% / 13% | 8% / 40% | 19% / 72% | **29% / 86%** | 41% / 94% |

想要 ≥80% 检出就得付 16~33% 的**误判**——把真人的姓名连续性打烂（V2）。
这与 survey §1 的物理一致，也与本仓库既有结论一致
（跨信道 genuine 本来就会掉到接受阈值以下）。

## 4. 实际修法：把回显**限期**，而不是判断说话人

先把 bug 的边界收窄。`_last_match[label]` 会被**任何** ≥ `min_identify_seconds`
的 turn 无条件重写（含写成 `(None, score)`），所以泄漏的精确范围是：

> 标签易主后，旧姓名一直回显，**直到新占用者产出一条 ≥2s 纯语音的 turn**。
> 若新占用者只说短句，就是永远。

对应的修法不需要任何新阈值——**复用已经标定好的 confirm/reject 判决**，只数次数：

```
每次跑完 embed+match（pure >= min_candidate_seconds）：
    pure >= min_identify_seconds ⇒ 照旧重写 _last_match，misses 清零
    否则                          ⇒ 本条 turn 若未 re-confirm 缓存的姓名，
                                     misses += 1；达到 identity_echo_max_misses
                                     就丢弃回显，并在该 turn 上置 contested=True
```

- 一条**重新确认**了缓存姓名的短 turn 会把计数清零 —— 真人的保护。
- 一条短 turn 即使以 ≥ 阈值确认了**别的**姓名，也只算**一次证据**，不短路：
  sub-2s 的确认本来就会误接受（2026-07-06），这与 `min_identify_seconds` 的
  存在理由一致。
- 完全没跑识别的 turn（多数重叠 / 无 cohort / 太短）**不计数**——没有证据。
- **不改本 turn 自己的 `name`**：那来自本 turn 音频对 roster 的匹配，与标签无关。

`identity_echo_max_misses=3`。它**限定泄漏时长**，不声称识别出了新的人。

## 5. 契约

- `settings.identity_echo_max_misses`（`DIART_IDENTITY_ECHO_MAX_MISSES`，默认 3）
- `TranscriptMessage.contested: bool = False`（追加字段，前向兼容）——
  语义是「本条 turn 上服务端丢弃了该标签的陈旧回显」，
  **不是**「本条 turn 是谁」。客户端必须丢掉为该标签缓存的姓名，且不得推断新姓名。
- `delta_new` / `tau_active` / `rho_update` **一律不动**，理由见 §2.2/§2.3，
  由 `scripts/eval/diart_cluster_sweep.py` 守住。

## 6. 不在本次范围

- 被命名率 14% → 不由本 bug 引起，见
  [选型计划](../superpowers/plans/2026-08-06-voiceprint-selection-plan.md) T3/T4。
- 合并后两人仍显示同一个「说话人 N」（显示层标签重编号）——改动面大，
  且 roster 命名正确后收益有限，留待 T4。

## 7. 门禁

| 脚本 / 测试 | 守什么 |
|---|---|
| `scripts/eval/diart_cluster_sweep.py` | 2news 说话人数 == 2、split 失败 == 0；`--sweep` 出 §2.3 那张表 |
| `scripts/verify_label_continuity.py` | §3 那两张表；若某天阈值方案变得可用则**主动 FAIL**，提示回头重议 |
| `tests/diart_server/test_label_continuity.py` | 计数规则的 11 条行为 |
| `tests/diart_server/test_diart_param_invariants.py` | 二值分割 / tau_active 失效 / 权重三取值 / 三个默认值不被顺手改掉 |

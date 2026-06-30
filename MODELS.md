# 模型清单（声纹 embedding / 分割 segmentation）

所有模型下载在 `./models`，运行时 `DIART_MODELS_DIR=./models`。WebSocket 连接用
`?embedding_model=<name>&segmentation_model=<name>` 选择；前端 `/demo/live` 提供下拉切换。

> ⚠️ 切换 embedding 模型会切换声纹库命名空间——不同模型的向量不可互比，
> 注册过的人只在"注册时所用的模型"下能被识别。

## Embedding（声纹）模型

| name (传给 `embedding_model`) | 后端 | 维度 | 速度 | 质量 | 备注 |
|---|---|---|---|---|---|
| `onnx/speaker-embedding.onnx`* | onnx | 256 | ★★★ | ★★☆ | WeSpeaker，纯离线最快 |
| `pyannote/wespeaker` | pyannote | 256 | ★★☆ | ★★★ | WeSpeaker fp32，第一梯队 |
| `speechbrain/ecapa` | speechbrain | 192 | ★★☆ | ★★★ | ECAPA-TDNN，SV 最强之一 |
| `speechbrain/ecapa-mel` | speechbrain | 192 | ★★☆ | ★★★ | ECAPA mel-spec 变体 |
| `speechbrain/resnet` | speechbrain | 256 | ★★☆ | ★★☆ | ResNet 声纹 |
| `speechbrain/xvect` | speechbrain | 512 | ★★☆ | ★★☆ | 经典 x-vector |
| `pyannote/embedding` | pyannote | 512 | ★★☆ | ★★☆ | diart 默认，精度偏弱 |
| `nvidia/titanet` | nemo | 192 | ★☆☆ | ★★★ | 需 `pip install nemo_toolkit[asr]`，默认隐藏 |

\* `name` 是相对 `DIART_MODELS_DIR` 的 `local_path`（见 `docker/models.json`）。

## Segmentation 模型
| name | 后端 | 备注 |
|---|---|---|
| `pyannote/segmentation-3.0` | pyannote | 默认 |
| `pyannote/segmentation` | pyannote | 旧版 |

## 切换 / 注册注意
- 推荐默认 `speechbrain/ecapa` 或 `pyannote/wespeaker`。
- 注册（`POST /v1/voiceprints`）必须带当前 `embedding_model`。
- titanet 默认在前端隐藏；要用先装 nemo 并从前端代码取消隐藏。

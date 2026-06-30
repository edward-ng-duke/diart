# 模型清单（声纹 embedding / 分割 segmentation）

所有模型下载在 `./models`，运行时 `DIART_MODELS_DIR=./models`。WebSocket 连接用
`?embedding_model=<name>&segmentation_model=<name>` 选择；前端 `/demo/live` 提供下拉切换。

> ⚠️ 切换 embedding 模型会切换声纹库命名空间——不同模型的向量不可互比，
> 注册过的人只在"注册时所用的模型"下能被识别。

## Embedding（声纹）模型

| name (传给 `embedding_model`) | 本地可用 | 后端 | 维度 | 速度 | 质量 | 备注 |
|---|---|---|---|---|---|---|
| `pyannote/wespeaker-voxceleb-resnet34-LM` | ✅ | pyannote | 256 | ★★☆ | ★★★ | **推荐默认**，本地验证：注册+流式均正常 |
| `pyannote/embedding` | ✅ | pyannote | 512 | ★★☆ | ★★☆ | diart 内置默认，本地验证正常 |
| `speechbrain/spkrec-ecapa-voxceleb` | ❌ | speechbrain | 192 | ★★☆ | ★★★ | 本地路径加载失败（`HFValidationError`）；diart 将 `models/speechbrain/...` 多斜杠路径传给 pyannote `Model.from_pretrained`，后者拒绝非 Hub 格式路径 |
| `speechbrain/spkrec-ecapa-voxceleb-mel-spec` | ❌ | speechbrain | 192 | ★★☆ | ★★★ | 同上，本地不可用 |
| `speechbrain/spkrec-resnet-voxceleb` | ❌ | speechbrain | 256 | ★★☆ | ★★☆ | 同上，本地不可用 |
| `speechbrain/spkrec-xvect-voxceleb` | ❌ | speechbrain | 512 | ★★☆ | ★★☆ | 同上，本地不可用 |
| `hbredin/wespeaker-voxceleb-resnet34-LM` | ❌ | onnx | 256 | ★★★ | ★★☆ | onnx 模型 forward pass 返回 None → `AttributeError: 'NoneType' object has no attribute 'cpu'`，注册时崩溃 |
| `nvidia/speakerverification_en_titanet_large` | ❌ | nemo | 192 | ★☆☆ | ★★★ | 需 `pip install nemo_toolkit[asr]`，且本地路径加载同样失败，前端默认隐藏 |

> ⚠️ **speechbrain / onnx / nemo 模型均不可从本地 `./models` 目录加载**（此构建版本）。
> 原因：diart 将本地路径（如 `models/speechbrain/spkrec-ecapa-voxceleb`）直接传给
> pyannote `Model.from_pretrained`，后者仅接受 Hub ID 或单文件路径，多级斜杠路径报
> `HFValidationError`；onnx 模型 forward pass 返回 None。修复需改动 diart-core，
> 超出本项目范围。**失败请求现在返回 HTTP 400（而非 500），错误信息明确说明原因。**

\* `name` 是相对 `DIART_MODELS_DIR` 的 `local_path`（见 `docker/models.json`）。

## Segmentation 模型
| name | 后端 | 备注 |
|---|---|---|
| `pyannote/segmentation-3.0` | pyannote | 默认 |
| `pyannote/segmentation` | pyannote | 旧版 |

## 切换 / 注册注意
- **推荐默认** `pyannote/wespeaker-voxceleb-resnet34-LM`（已验证）或 `pyannote/embedding`。
- speechbrain / onnx / nemo 模型当前不可用（见上表），注册这些模型会返回 HTTP 400。
- 注册（`POST /v1/voiceprints`）必须带当前 `embedding_model`；不同模型的声纹库互不兼容。
- titanet 默认在前端隐藏；要用先装 nemo 并从前端代码取消隐藏。

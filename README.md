# EC YOLACT 实验代码说明



## 模型说明与EGA接入位置

EC-YOLACT以YOLACT实例分割结构为基础，完整模型采用ResNet101 + CA作为主干，FPN生成P3–P7；共享预测头输出类别、检测框和掩膜系数。EGA对P3的原型分支输入进行重标定，ProtoNet生成32个原型掩膜，之后与掩膜系数组合并完成实例掩膜恢复。

| 路径 | 数据流 |
|---|---|
| 特征提取 | 图像 → ResNet101各残差阶段 + CA → C3/C4/C5 → FPN |
| 目标预测 | FPN的P3–P7 → 共享PredictionModule → 框、类别、32维掩膜系数 |
| 原型生成 | FPN的P3 → **EGA** → ProtoNet → 32个原型掩膜 |
| 实例输出 | 原型与系数组合 → Sigmoid → 框裁剪和尺寸恢复；NMS筛选实例 |

EGA按每张图像独立计算，不在batch维度归一化：

```text
E[b,1,i,j] = sum_c X[b,c,i,j]^2
A[b,1,i,j] = E[b,1,i,j] / (sum_mn E[b,1,m,n] + eps)
Y = X + gamma * X * A
```

X/Y形状均为[B,C,H,W]；A沿通道广播；eps=1e-6。gamma是可学习残差系数，对应论文EGA公式的beta，初始化为0。初始化时模块是恒等映射，训练后由任务损失调整增强幅度。EGA采用平方能量，不使用熵、softmax或对数；高能量不是缺陷的确定判据，改善效果仍需实验验证。

`utils/EGA.py`的标准类名是`EGA`，同时保留`EnergyAttention`别名，兼容上一版导入与权重键。半精度输入的能量求和使用FP32；模块保持输出尺寸和输入浮点类型。

`nets/yolact.py`中已接入以下调用，并非仅提供独立模块：

```python
from utils.EGA import EGA

# __init__: prototype_attention == "ega" 时
self.proto_attention = EGA()

# forward: FPN之后、ProtoNet之前
prototype_features = self.proto_attention(features[0])  # P3
pred_proto = self.proto_net(prototype_features)
```

这段重标定只改变原型支路，不修改共享预测头使用的P3。`--variant ec`开启Backbone CA + Prototype EGA；`--variant b`用于仅开启Prototype EGA的实验；`--variant a`把EGA放在主干阶段后进行位置消融。

## 快速运行

以下命令均在解压后的项目根目录执行。默认设备是CUDA；仅进行CPU调试可加`--device cpu`。先安装可用的PyTorch/torchvision配对版本，再安装项目依赖。

```powershell
python -m pip install -r requirements.txt
python utils/EGA.py
python -m unittest discover -s tests -v
```

先把WSD图像补齐到`datasets/wsd/images/`，或修改`configs/wsd.json`的图像目录；包内已有训练和验证JSON标注。检查通过后训练：

```powershell
python check_dataset.py --config configs/wsd.json
python train.py --config configs/wsd.json --variant ec --scratch --output runs/wsd_ec_scratch
```

上面的`--scratch`命令从零训练，不需要预训练文件，但不等价于预训练初始化实验。若已准备兼容的ResNet101权重，使用：

```powershell
python train.py --config configs/wsd.json --variant ec --init model_data/resnet101_imagenet.pth --output runs/wsd_ec
python evaluate.py --weights runs/wsd_ec/best_loss.pt --output results/wsd_ec
python predict.py --weights runs/wsd_ec/best_loss.pt --image D:/wsd/example.jpg --output results/example
python benchmark.py --weights runs/wsd_ec/best_loss.pt --image D:/wsd/example.jpg --output results/wsd_ec_speed.json
```

`resnet101_imagenet.pth`和`D:/wsd/example.jpg`是需要替换为实际文件的路径；包内没有该R101预训练权重和真实图像。从零训练后评估时，请把路径中的`wsd_ec`改为`wsd_ec_scratch`。训练输出目录必须为新目录，以免覆盖已有结果。

没有数据时，可验证独立模块和运行`python tests/smoke_cli.py`完成合成数据流程测试；合成测试结果不属于论文实验结果。

## 1 文件入口

| 路径 | 用途 |
|---|---|
| `configs/wsd.json` | WSD 七类实验设置与数据路径 |
| `configs/coco.json` | COCO 2017 八十类实验设置与数据路径 |
| `nets/resnet.py` | ResNet50 / ResNet101 及四阶段注意力开关 |
| `nets/yolact.py` | FPN、共享预测头、ProtoNet 和原型分支注意力 |
| `utils/EGA.py` | 能量注意力，已替换原来的熵注意力 |
| `utils/CA.py` | 门控响应注意力，保留论文 CA 名称 |
| `nets/yolact_training.py` | 框、分类、实例掩膜与语义辅助损失 |
| `train.py` | 统一训练入口，保存配置、日志和模型 |
| `evaluate.py` | COCO bbox / segm 评估及各类别 AP50 |
| `predict.py` | 单图可视化、实例二值掩膜与JSON导出 |
| `benchmark.py` | 同步计时，分别报告前向和前向加后处理 |
| `check_dataset.py` | 类别、图像路径和划分交叉检查 |
| `tests/` | 模块测试与合成数据端到端测试 |
| `reports/` | 原始代码与数据审查、实际测试记录 |
| `legacy/` | 旧入口和关键旧模块，仅用于对照，不运行 |

相对路径均以项目根目录为基准；先进入解压后的 `EC-YOLACT-reviewed` 文件夹。原压缩包保持不变。本包不重复携带原 R50 权重、历史预测结果、缓存和编辑器配置。

## 2 环境与安装

建议为实验建立独立环境。使用与显卡驱动匹配的 PyTorch / torchvision 配对版本；如果原有 Python 3.11 + PyTorch 2.9.0 + CUDA 12.8 环境已经可用，可先沿用，并执行以下命令验证。不要直接把 CPU 测试环境的版本当作论文训练环境。

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

本次功能测试使用 Linux、Python 3.12、PyTorch 2.15.0+cu132，实际运行设备为 CPU，不代表在 Windows / RTX 5060 Ti 上已经验证。训练入口保持 `if __name__ == '__main__'`，适配 Windows 多进程启动；若 DataLoader 报多进程问题，加 `--workers 0` 排查。

## 3 准备 WSD 数据

本包保留原训练标注 3672 张、3708 个实例，以及原验证标注 408 张、411 个实例；两者按文件名无交叉，共 4080 张，为约 90% / 10% 划分。包内没有图像，也没有独立测试集。将对应图像放入 `datasets/wsd/images/`，文件名与 JSON 的 `file_name` 完全一致；也可在配置中分别填写训练和验证图像的绝对目录。Windows 路径建议写成 `D:/datasets/wsd/images`。

```powershell
python check_dataset.py --config configs/wsd.json --annotations-only
python check_dataset.py --config configs/wsd.json
```

第一条可在没有图片时检查类别和划分；第二条要求全部图像存在。按文件名检查不能发现改名副本或同晶圆相邻帧泄漏。正式实验还应按晶圆或采集批次分组，并排查近重复图像；不得将验证集反复调参后的分数写作独立测试集结果。

| 配置顺序 / 标签值 | 原标注名称 | 论文简称 |
|---|---|---|
| 0 | SCRATCH | Sc |
| 1 | PARTICLE | Pa |
| 2 | BLOCK ETCH | BE |
| 3 | COATING BAD | CD |
| 4 | PIQ PARTICLE | PP |
| 5 | PO CONTAMINATION | PC |
| 6 | SEZ BURNT | SB |

模型额外包含背景类 0，前景类别在分类头内部偏移一位；评估时恢复原 COCO category_id。不要把其他项目的七类顺序直接复制过来。PO CONTAMINATION 是否可明确表述为 Phosphate Contamination，需要数据来源支持，代码保留原始名称。

## 4 参数与消融实验

默认输入 544×544，anchors 为 24、48、96、192、384；SGD，200 epochs，batch size 16，初始学习率 0.002，余弦衰减，无隐式学习率缩放。补充实现设置为 momentum=0.9、weight decay=0.0005、Nesterov=True、min_lr=0.00002、seed=11，无冻结阶段、无 warmup、FP32。表1未明确的参数应在论文或附录补充。

| --variant | Backbone | Prototype | 对应实验 |
|---|---|---|---|
| baseline | 无注意力 | 无注意力 | YOLACT 基线 |
| a | EGA | 无注意力 | Backbone + EGA |
| b | 无注意力 | EGA | Prototype + EGA |
| c | CA | 无注意力 | Backbone + CA |
| d | 无注意力 | CA | Prototype + CA |
| ec | CA | EGA | 完整 EC-YOLACT |

所有组默认使用 R101；主干模块位于四个残差阶段输出后，向 FPN 提供最后三个阶段特征。原型模块仅作用于 P3→ProtoNet 支路，不直接修改送往共享预测头的 P3。注意力的残差系数初始化为 0，相同随机种子下各组共有层的初始权重一致。

## 5 初始化和训练

原包 `model_data/yolact_weights_coco.pth` 是 R50 权重，不能作为默认 R101 实验的等价初始化。应提供 R101 的 torchvision ResNet 权重字典，或本项目相同结构的完整 YOLACT 权重。不同来源的 YOLACT 实现可能使用不同键名，不能仅凭“R101”文件名判断兼容。加载器检查主干深度与主干键完整性；允许替换类别头，具体未加载键保存在 `initialization.json`。所有对照组采用相同初始化来源。

```powershell
python train.py --config configs/wsd.json --variant ec --init model_data/resnet101_imagenet.pth --output runs/wsd_ec
python train.py --config configs/wsd.json --variant baseline --init model_data/resnet101_imagenet.pth --output runs/wsd_baseline
```

上述权重文件需自行提供，本包没有下载或伪造该文件。若明确要从零训练：

```powershell
python train.py --config configs/wsd.json --variant ec --scratch --output runs/wsd_ec_scratch
```

PowerShell 顺序运行六组，注意输出目录不得已有训练结果：

```powershell
foreach ($v in @('baseline','a','b','c','d','ec')) {
  python train.py --config configs/wsd.json --variant $v --init model_data/resnet101_imagenet.pth --output "runs/wsd_$v"
  if ($LASTEXITCODE -ne 0) { break }
}
```



训练保存 `last.pt`、`best_loss.pt`、`training.csv`、`config.json`、`initialization.json` 和 `environment.json`。best_loss 按验证总损失选择，**不是按最高 AP 选择**。建议固定选模策略，所有实验一致。训练采用原来的随机增强管线；验证采用固定缩放，不进行随机裁剪、翻转和颜色扰动。

## 6 精度评估

```powershell
python evaluate.py --weights runs/wsd_ec/best_loss.pt --output results/wsd_ec
```

评估读取权重中保存的配置；移动数据后可加 `--config configs/wsd.json` 覆盖路径，类别、输入尺寸、anchors、主干和 CA 归一化必须一致。输出 `metrics.csv`、`metrics.json`、`per_class.csv` 和 bbox / mask 预测 JSON。覆盖验证集全部图片，包括无目标图片。

| 论文栏目 | 导出字段 |
|---|---|
| 表2 MS COCO AP / AP50 / AP75 / APS / APM / APL | `type=segm` 对应字段，明确为实例掩膜指标 |
| 表3 Evaluating BBoxes | `type=bbox` 对应字段 |
| 表3 Evaluating Masks | `type=segm` 对应字段 |
| 表4 mAP@0.5 | `type=segm` 的 AP50 |

AP 使用 COCO IoU=0.50:0.05:0.95；AP50 仅 IoU=0.50；APS / APM / APL 使用 COCO 原图尺度定义。另输出 AR100 与 Recall50：前者跨 IoU 平均，后者 IoU=0.50、每图最多100个预测的类别平均召回，二者不能混用。无有效样本的指标写为 null / CSV 空值，不填0。置信度截断0.001，Fast NMS IoU0.5，最多100个实例，掩膜阈值0.5；这些设置须对全部组固定。

MS COCO 实验将配置改为 `configs/coco.json`，提供 train2017、val2017 和对应 instances 标注，再用同一训练及评估命令。该配置明确采用 COCO 2017 val2017；如果原文比较的是 test-dev 或其他年份，应改为相同划分并重新评估，不能直接混用。

## 7 速度测试

```powershell
python benchmark.py --weights runs/wsd_ec/best_loss.pt --image datasets/wsd/images/实际文件名.jpg --warmup 30 --runs 100 --output results/wsd_ec_speed.json
```

使用真实验证图像；batch=1，FP32，544×544，GPU 同步后计时。分别报告网络前向、网络前向加 NMS 和掩膜恢复；均不包含读盘、预处理、CPU→GPU传输及绘图。FPS=1000/平均毫秒耗时。论文必须明确采用哪一项，不应把不同硬件、精度模式或计时边界的数字放在同一比较表中。单张图重复计时是基准入口；严谨速度报告应补充多张代表性图像的统计。

## 8 验证与限制

```powershell
python -m unittest discover -s tests -v
python tests/smoke_cli.py
```

模块测试覆盖六组开关、R101完整输入、空目标、crowd、能量公式、梯度、共有层初始化一致性、错误深度权重拒绝和空预测评估。合成测试运行小尺寸 R50 的一次训练、严格重载、bbox/segm评估及计时，临时数据自动删除。`reports/smoke_cli.txt` 的 AP / FPS 仅是流程输出，绝不能写进论文。

修改了网络深度、模块、损失及评估划分后，旧结果不再自动对应此版本。请重新训练全部六组，再更新论文表格；本包没有内置目标准确率，也不保证达到96.5%、76.5%或29.4 FPS。不能用本项目的基线代替外部实现的 Mask R-CNN、PANet、HTC、CondInst、MaskFormer、YOLO 分割等模型实验；这些模型仍需各自代码及相同数据协议。

保留原 LICENSE 和 `legacy/README_original.md` 中的来源信息。COCO指标实现参考 https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py 。

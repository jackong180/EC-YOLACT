# EGA与模型接入更新

- utils/EGA.py：标准类名EGA，保留EnergyAttention别名；平方能量、逐样本空间归一化、可学习残差。保持gamma权重键，兼容上一整理版。
- nets/yolact.py：明确P3→EGA→ProtoNet调用；共享预测头继续读取原FPN特征。
- nets/resnet.py：主干EGA位置消融使用同一个EGA实现。
- README.md：新增模型概述、模块公式、接入代码及快速运行命令。
- tests/test_pipeline.py：新增输入约定和原型支路实际调用验证，共8项测试通过。

本次不改变上一整理版EGA的数学计算，主要统一模块命名、补充输入检查和文档，并验证实际接入。docs中的Word文档记录上一轮6项基础测试；当前测试记录以reports/tests.txt和本说明为准。

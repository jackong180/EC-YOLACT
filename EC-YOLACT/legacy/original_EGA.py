import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyAttention(nn.Module):
    """
    Entropy-driven Adaptive Attention (EAA)

    用于:
        YOLACT Prototype分支增强

    输入:
        x: [B,C,H,W]

    输出:
        y: [B,C,H,W]

    特点:
        1. 无Q/K/V
        2. 无N*N矩阵
        3. 参数量极低
        4. 适合晶圆缺陷分割
    """

    def __init__(self,
                 temperature=1.0,
                 eps=1e-6):

        super(
            EntropyAttention,
            self
        ).__init__()


        self.temperature = temperature

        self.eps = eps


        # 可学习残差比例
        self.gamma = nn.Parameter(
            torch.zeros(1)
        )


    def forward(self, x):

        """
        x:
        B,C,H,W
        """

        B,C,H,W = x.size()


        # =================================
        # 1. 特征空间概率建模
        #
        # p_i=e^xi/sum(e^x)
        # =================================

        feature = torch.mean(
            x,
            dim=1,
            keepdim=True
        )


        probability = F.softmax(
            feature.flatten(2)
            /
            self.temperature,
            dim=-1
        )


        probability = probability.reshape(
            B,
            1,
            H,
            W
        )


        # =================================
        # 2. 信息熵计算
        #
        # H=-sum(p log p)
        # =================================


        entropy = -(
            probability *
            torch.log(
                probability+self.eps
            )
        )


        # =================================
        # 3. 熵转注意力权重
        #
        # A=1/(1+H)
        # =================================

        attention = 1 / (
            1 + entropy
        )


        # =================================
        # 4. 特征增强
        # =================================

        out = x * attention


        # =================================
        # 5. 残差学习
        # =================================

        out = (
            x
            +
            self.gamma*out
        )


        return out
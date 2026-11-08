import torch
import torch.nn as nn
import torch.nn.functional as F



class CausalAttention(nn.Module):
    """
    Causal Attention (CA)

    用于:
        ResNet Backbone增强

    输入:
        B,C,H,W

    输出:
        B,C,H,W


    特点:
        1. 无QKV
        2. 无N*N矩阵
        3. 推理可用
        4. 适合YOLACT Backbone
    """

    def __init__(
            self,
            channels,
            reduction=16):

        super(
            CausalAttention,
            self
        ).__init__()


        # 因果贡献预测器

        self.causal_gate = nn.Sequential(

            nn.Conv2d(
                channels,
                channels//reduction,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),


            nn.Conv2d(
                channels//reduction,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )


        # 可学习残差比例

        self.gamma = nn.Parameter(
            torch.zeros(1)
        )


    def forward(self,x):

        """
        x:
        B,C,H,W
        """

        # ===========================
        # 1. 学习因果贡献
        # ===========================

        gate = self.causal_gate(x)


        # gate:
        # B,1,H,W


        # ===========================
        # 2. 特征响应贡献
        # ===========================

        response = torch.mean(
            torch.abs(x),
            dim=1,
            keepdim=True
        )


        # ===========================
        # 3. 因果权重
        # ===========================

        causal_weight = (
            gate *
            response
        )


        causal_weight = (
            causal_weight /
            (
                torch.mean(
                    causal_weight,
                    dim=(2,3),
                    keepdim=True
                )
                +1e-6
            )
        )


        # ===========================
        # 4. 特征增强
        # ===========================

        out = (
            x *
            causal_weight
        )


        # ===========================
        # 5. Residual
        # ===========================

        out = (
            x
            +
            self.gamma*out
        )


        return out
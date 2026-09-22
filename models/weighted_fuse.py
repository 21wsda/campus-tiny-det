"""
WeightedFuse: 可学习加权融合模块
继承自 ultralytics 的 Concat，这样 parse_model 能正确计算输出通道数
F = sum(w_i * X_i), w_i >= 0, sum(w_i) = 1
内部用 1x1 conv 把各输入投影到相同通道数，输出通道数 = 输入通道数之和
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules import Concat


class WeightedFuse(Concat):
    """
    输入: 多个已经对齐高宽的特征图（列表）
    输出: 加权融合后的特征图，通道数等于各输入通道数之和
    继承 Concat 是为了让 ultralytics 的 parse_model 正确计算输出通道数
    """

    def __init__(self, dimension=1):
        super().__init__(dimension)
        # 可学习权重和投影层在第一次 forward 时根据实际通道数创建
        self.weights = None
        self.projections = None
        self.out_channels = None
        self._initialized = False

    def _build(self, in_channels_list, device):
        """根据实际输入通道数创建参数"""
        total_out = sum(in_channels_list)
        self.out_channels = total_out
        num_inputs = len(in_channels_list)

        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32)).to(device)
        self.projections = nn.ModuleList()
        for c in in_channels_list:
            proj = nn.Conv2d(c, total_out, kernel_size=1, bias=False).to(device)
            nn.init.kaiming_normal_(proj.weight, mode='fan_out', nonlinearity='relu')
            self.projections.append(proj)

        self._initialized = True

    def forward(self, x):
        # x 是一个列表，包含多个输入特征图
        if not isinstance(x, (list, tuple)):
            x = [x]

        if not self._initialized:
            in_channels_list = [t.shape[1] for t in x]
            self._build(in_channels_list, x[0].device)

        # softmax 保证 w_i >= 0 且和为 1
        w = torch.softmax(self.weights, dim=0)

        out = w[0] * self.projections[0](x[0])
        for i in range(1, len(x)):
            out = out + w[i] * self.projections[i](x[i])
        return out

    def extra_repr(self):
        return f"dimension={self.d}, out_channels={self.out_channels}"

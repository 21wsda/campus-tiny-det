"""
WeightedFuse: 可学习加权融合模块
F = sum(w_i * X_i), w_i >= 0, sum(w_i) = 1
内部用 1x1 conv 把各输入投影到相同通道数，输出通道数 = 输入通道数之和
（和原来 Concat 的输出通道数保持一致，后面的 C2f 不用改）
"""
import torch
import torch.nn as nn


class WeightedFuse(nn.Module):
    """
    输入: 多个已经对齐高宽的特征图
    args: num_inputs (int) = 输入特征图个数
    输出: 加权融合后的特征图，通道数等于各输入通道数之和
    """

    def __init__(self, num_inputs: int = 2):
        super().__init__()
        self.num_inputs = num_inputs
        # 可学习权重，初始化为均匀
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        # 投影层在 forward 第一次时根据实际输入通道数创建
        self.projections = None

    def _build_projections(self, in_channels_list, device):
        """根据实际输入通道数创建 1x1 conv 投影层"""
        total_out = sum(in_channels_list)
        self.projections = nn.ModuleList()
        for c in in_channels_list:
            # 1x1 conv: c -> total_out
            proj = nn.Conv2d(c, total_out, kernel_size=1, bias=False).to(device)
            self.projections.append(proj)

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert len(inputs) == self.num_inputs, \
            f"WeightedFuse 期望 {self.num_inputs} 个输入，实际收到 {len(inputs)}"

        # 第一次前向时根据实际通道数建投影层
        if self.projections is None:
            in_channels_list = [x.shape[1] for x in inputs]
            self._build_projections(in_channels_list, inputs[0].device)

        # softmax 保证 w_i >= 0 且和为 1
        w = torch.softmax(self.weights, dim=0)

        out = w[0] * self.projections[0](inputs[0])
        for i in range(1, self.num_inputs):
            out = out + w[i] * self.projections[i](inputs[i])
        return out

    def extra_repr(self):
        return f"num_inputs={self.num_inputs}"

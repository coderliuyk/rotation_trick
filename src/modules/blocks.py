"""Adapted from taming transformers: https://github.com/CompVis/taming-transformers."""

import torch
import torch.nn as nn

# 定义非线性激活函数，使用 Swish 激活函数
def nonlinearity(x):
    return x * torch.sigmoid(x)

# 定义归一化层，使用 Group Normalization
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# 定义上采样模块
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv  # 是否使用卷积层
        if self.with_conv:
            # 如果使用卷积，定义卷积层
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # 使用最近邻插值进行上采样
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)  # 如果使用卷积，进行卷积操作
        return x

# 定义下采样模块
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv  # 是否使用卷积层
        if self.with_conv:
            # 定义卷积层，注意这里没有使用非对称填充
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)  # 定义填充
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)  # 填充操作
            x = self.conv(x)  # 通过卷积层
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)  # 使用平均池化进行下采样
        return x

# 定义残差块
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels  # 确定输出通道数
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut  # 是否使用卷积短路连接
        self.norm1 = Normalize(in_channels)  # 第一层归一化
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)  # 第一层卷积
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)  # 时间嵌入投影
        self.norm2 = Normalize(out_channels)  # 第二层归一化
        self.dropout = torch.nn.Dropout(dropout)  # Dropout 层
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)  # 第二层卷积
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)  # 卷积短路连接
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)  # 1x1卷积短路连接

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)  # 先进行归一化
        h = nonlinearity(h)  # 激活
        h = self.conv1(h)  # 通过第一层卷积
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]  # 添加时间嵌入
        h = self.norm2(h)  # 进行第二次归一化
        h = nonlinearity(h)  # 激活
        h = self.dropout(h)  # Dropout
        h = self.conv2(h)  # 通过第二层卷积
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)  # 如果需要，使用卷积短路连接
            else:
                x = self.nin_shortcut(x)  # 否则使用1x1卷积短路连接
        return x + h  # 返回残差连接的结果

# 定义注意力块
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)  # 归一化层
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)  # 查询卷积层
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)  # 键卷积层
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)  # 值卷积层
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)  # 输出投影层

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)  # 归一化
        q = self.q(h_)  # 查询
        k = self.k(h_)  # 键
        v = self.v(h_)  # 值
        # 计算注意力
        b, c, h, w = q.shape  # 获取形状
        q = q.reshape(b, c, h * w)  # 重塑查询
        q = q.permute(0, 2, 1)  # 变换维度为 (b, hw, c)
        k = k.reshape(b, c, h * w)  # 重塑键
        w_ = torch.bmm(q, k)  # 计算注意力权重
        w_ = w_ * (int(c) ** (-0.5))  # 缩放权重
        w_ = torch.nn.functional.softmax(w_, dim=2)  # 计算 softmax

        # 计算值的加权和
        v = v.reshape(b, c, h * w)  # 重塑值
        w_ = w_.permute(0, 2, 1)  # 转换为 (b, hw, hw)
        h_ = torch.bmm(v, w_)  # 计算加权和
        h_ = h_.reshape(b, c, h, w)  # 重塑为原始形状
        h_ = self.proj_out(h_)  # 通过输出投影层
        return x + h_  # 返回残差连接的结果

    h_ = self.proj_out(h_)

    return x + h_

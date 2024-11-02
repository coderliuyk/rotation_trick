"""Adapted from taming transformers: https://github.com/CompVis/taming-transformers"""
"""Adapted from taming transformers: https://github.com/CompVis/taming-transformers"""
import torch
import sys
import numpy as np
import torch.nn as nn
from pyprojroot import here as project_root

# 将项目根目录添加到系统路径
sys.path.insert(0, str(project_root()))

from src.modules.blocks import ResnetBlock, AttnBlock, nonlinearity, Normalize, Upsample


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        
        # 初始化参数
        self.ch = ch  # 基础通道数
        self.temb_ch = 0  # 时间嵌入通道数
        self.num_resolutions = len(ch_mult)  # 分辨率数量
        self.num_res_blocks = num_res_blocks  # 每个分辨率的残差块数量
        self.resolution = resolution  # 输出分辨率
        self.in_channels = in_channels  # 输入通道数
        self.give_pre_end = give_pre_end  # 是否返回中间结果

        # 计算最低分辨率的输入通道数和当前分辨率
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z到block_in的卷积层
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # 中间模块
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)  # 注意力块
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)

        # 上采样模块
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]  # 当前层输出的通道数
            for i_block in range(self.num_res_blocks + 1):
                # 添加残差块
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out  # 更新输入通道数
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))  # 添加注意力块
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)  # 上采样
                curr_res = curr_res * 2  # 更新当前分辨率
            self.up.insert(0, up)  # 前置以保持顺序一致

        # 最后的规范化和输出卷积层
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        # 断言检查输入z的形状是否符合预期
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # 时间步嵌入（目前未使用）
        temb = None

        # z到block_in
        h = self.conv_in(z)

        # 中间处理
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)  # 残差块
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)  # 注意力块
            if i_level != 0:
                h = self.up[i_level].upsample(h)  # 上采样

        # 最后的处理
        if self.give_pre_end:
            return h  # 返回中间结果

        h = self.norm_out(h)  # 规范化
        h = nonlinearity(h)  # 激活函数
        h = self.conv_out(h)  # 最后卷积层
        return h  # 返回输出

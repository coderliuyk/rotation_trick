import sys
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from pyprojroot import here as project_root

# 将项目根目录添加到系统路径
sys.path.insert(0, str(project_root()))

# 导入编码器和解码器
from diffusers.models.autoencoders.vae import Encoder, Decoder
from taming.modules.diffusionmodules.model import Encoder, Decoder
from src.models.model_utils import get_model_params
from src.local_vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 获取模型参数
        channels, resolution, z_channels, embed_dim, n_embed = get_model_params(args.dataset, args.f)
        self.args = args
        self.num_codes = n_embed  # 嵌入的数量
        self.cosine = (args.codebook == 'cosine')  # 是否使用余弦相似度
        decay = 0.8  # 代码本衰减的默认值

        # 根据输入的参数选择编码器和解码器的配置
        if args.f == 8:
            self.encoder = Encoder(double_z=False, z_channels=z_channels, resolution=resolution, in_channels=channels,
                                   out_ch=channels, ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2,
                                   attn_resolutions=[16], dropout=0.0)
            self.decoder = Decoder(double_z=False, z_channels=z_channels, resolution=resolution, in_channels=channels,
                                   out_ch=channels, ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2,
                                   attn_resolutions=[16], dropout=0.0)
        elif args.f == 4:
            self.encoder = Encoder(double_z=False, z_channels=z_channels, resolution=resolution, in_channels=channels,
                                   out_ch=channels, ch=128, ch_mult=[1, 2, 4], num_res_blocks=2,
                                   attn_resolutions=[], dropout=0.0)
            self.decoder = Decoder(double_z=False, z_channels=z_channels, resolution=resolution, in_channels=channels,
                                   out_ch=channels, ch=128, ch_mult=[1, 2, 4], num_res_blocks=2,
                                   attn_resolutions=[], dropout=0.0)

        # 初始化向量量化模块
        if args.codebook == 'cosine' or args.codebook == 'euclidean':
            # 使用EMA更新学习代码本向量
            self.vq = VectorQuantize(dim=embed_dim, codebook_size=n_embed, commitment_weight=args.commit_weight,
                                     decay=decay, accept_image_fmap=True, use_cosine_sim=(args.codebook == 'cosine'),
                                     threshold_ema_dead_code=0)
        else:
            raise Exception(f'codebook: {args.codebook} is not supported.')

        # 设置进入和退出代码本的投影
        if args.codebook == 'cosine':
            self.pre_quant_proj = nn.Sequential(nn.Linear(z_channels, embed_dim),
                                                nn.LayerNorm(embed_dim)) if embed_dim != z_channels else nn.Identity()
        else:
            self.pre_quant_proj = nn.Sequential(
                nn.Linear(z_channels, embed_dim)) if embed_dim != z_channels else nn.Identity()
        self.post_quant_proj = nn.Linear(embed_dim, z_channels) if embed_dim != z_channels else nn.Identity()

    def get_codes(self, x):
        # 编码过程
        x = self.encoder(x)  # 使用编码器编码输入
        x = rearrange(x, 'b c h w -> b h w c')  # 调整维度
        x = self.pre_quant_proj(x)  # 投影到嵌入维度
        x = rearrange(x, 'b h w c -> b c h w')  # 调整维度

        # VQ查找
        quantized, indices, _ = self.vq(x)  # 量化
        return indices  # 返回索引

    def decode(self, indices):
        # 从索引获取量化的代码
        q = self.vq.get_codes_from_indices(indices)
        if self.cosine:
            q = q / torch.norm(q, dim=1, keepdim=True)  # 进行L2归一化

        # 解码过程
        x = self.post_quant_proj(q)  # 投影回z通道
        x = rearrange(x, 'b (h w) c -> b c h w', h=16)  # 调整维度
        x = self.decoder(x)  # 使用解码器解码
        return x  # 返回解码结果

    def encode_forward(self, x):
        # 编码前向过程
        x = self.encoder(x)  # 编码输入
        x = rearrange(x, 'b c h w -> b h w c')  # 调整维度
        x = self.pre_quant_proj(x)  # 投影
        x = rearrange(x, 'b h w c -> b c h w')  # 调整维度

        # VQ查找
        quantized, indices, _ = self.vq(x)
        return quantized  # 返回量化结果

    def decoder_forward(self, q):
        if self.cosine:
            q = q / torch.norm(q, dim=1, keepdim=True)  # 进行L2归一化

        # 解码过程
        x = rearrange(q, 'b c h w -> b h w c')  # 调整维度
        x = self.post_quant_proj(x)  # 投影
        x = rearrange(x, 'b h w c -> b c h w')  # 调整维度
        x = self.decoder(x)  # 使用解码器解码
        return x  # 返回解码结果

    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        # 计算高效的旋转
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()  # 计算旋转向量
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
            torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e  # 返回更新后的e

    def forward(self, x, vhp=False, return_rec=False, double_fp=False, rot=False, loss_scale=1.0):
        init_x = x  # 保存原始输入
        # 编码过程
        x = self.encoder(x)
        x = rearrange(x, 'b c h w -> b h w c')  # 调整维度
        x = self.pre_quant_proj(x)  # 投影
        x = rearrange(x, 'b h w c -> b c h w')  # 调整维度

        # 进行L2归一化
        if self.cosine:
            x = x / torch.norm(x, dim=1, keepdim=True)

        e = x  # 保存编码后的向量
        # VQ查找
        quantized, indices, commit_loss = self.vq(x)
        q = quantized  # 量化后的结果

        # 如果使用旋转技巧
        if rot:
            b, c, h, w = x.shape
            x = rearrange(x, 'b c h w -> (b h w) c')  # 扁平化
            quantized = rearrange(quantized, 'b c h w -> (b h w) c')  # 扁平化
            pre_norm_q = self.get_very_efficient_rotation(
                x / (torch.norm(x, dim=1, keepdim=True) + 1e-6),
                quantized / (torch.norm(quantized, dim=1, keepdim=True) + 1e-6),
                x.unsqueeze(1)).squeeze()
            quantized = pre_norm_q * (
                torch.norm(quantized, dim=1, keepdim=True) / (torch.norm(x, dim=1, keepdim=True) + 1e-6)).detach()
            quantized = rearrange(quantized, '(b h w) c -> b c h w', b=b, h=h, w=w)  # 恢复形状

        # 如果进行双前向传播以获取精确梯度
        if double_fp:
            quantized = quantized.detach()  # 移除STE估计
        if self.cosine:
            quantized = quantized / torch.norm(quantized, dim=1, keepdim=True)  # L2归一化

        # 解码过程
        x = rearrange(quantized, 'b c h w -> b h w c')
        x = self.post_quant_proj(x)  # 投影
        x = rearrange(x, 'b h w c -> b c h w')  # 调整维度
        x = self.decoder(x)  #

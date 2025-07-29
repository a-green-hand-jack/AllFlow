"""简单UNet网络实现用于Flow Matching测试

本模块实现了一个简化的UNet架构，专门用于Flow Matching算法的测试验证。
这个实现专注于核心功能而非最优性能，确保测试的快速执行和可靠性。

架构特点:
- 编码器-解码器结构，适合处理空间数据
- 时间条件输入，支持Flow Matching的时间参数
- 残差连接，提升训练稳定性
- 批量归一化，加速收敛

设计目标:
- 简单易懂，便于测试验证
- 快速训练，减少测试时间
- 稳定收敛，确保测试可靠性
- 模块化设计，易于扩展

Author: AllFlow Test Suite Contributors
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """时间条件嵌入层.
    
    将标量时间参数转换为高维特征向量，用于条件化UNet的各个层。
    使用正弦位置编码确保时间的连续性和周期性。
    """
    
    def __init__(self, dim: int):
        """初始化时间嵌入层.
        
        Args:
            dim: 嵌入维度，必须是偶数
        """
        super().__init__()
        self.dim = dim
        
        # 创建正弦位置编码的频率
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))
        
        # 时间嵌入的MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """前向传播.
        
        Args:
            t: 时间参数, shape: (batch_size,), 范围 [0, 1]
            
        Returns:
            时间嵌入, shape: (batch_size, dim)
        """
        # 扩展时间维度: (batch_size,) -> (batch_size, half_dim)  
        emb = t[:, None] * self.emb[None, :]  # type: ignore
        
        # 使用sin和cos创建位置编码
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # 通过MLP变换
        return self.mlp(emb)


class ResBlock(nn.Module):
    """残差块，带时间条件.
    
    基础的残差连接块，集成时间嵌入来实现条件化生成。
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        """初始化残差块.
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数  
            time_emb_dim: 时间嵌入维度
        """
        super().__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 时间条件投影
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接的投影
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """前向传播.
        
        Args:
            x: 输入特征, shape: (batch_size, in_channels, H, W)
            time_emb: 时间嵌入, shape: (batch_size, time_emb_dim)
            
        Returns:
            输出特征, shape: (batch_size, out_channels, H, W)
        """
        # 保存残差连接
        residual = self.residual_conv(x)
        
        # 第一个卷积 + 批归一化 + 激活
        h = F.relu(self.bn1(self.conv1(x)))
        
        # 加入时间条件
        time_out = self.time_mlp(time_emb)[:, :, None, None]  # (batch_size, out_channels, 1, 1)
        h = h + time_out
        
        # 第二个卷积 + 批归一化
        h = self.bn2(self.conv2(h))
        
        # 残差连接 + 激活
        return F.relu(h + residual)


class SimpleUNet(nn.Module):
    """简化的UNet网络，用于Flow Matching测试.
    
    实现编码器-解码器架构，支持时间条件输入。专门设计用于
    Flow Matching算法的速度场预测任务。
    
    网络结构:
    - 编码器: 逐步下采样，提取多尺度特征
    - 瓶颈层: 最深层特征处理
    - 解码器: 逐步上采样，恢复原始分辨率
    - 跳跃连接: 保留细节信息
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        num_res_blocks: int = 2
    ):
        """初始化UNet网络.
        
        Args:
            in_channels: 输入通道数（MNIST为1）
            out_channels: 输出通道数（速度场维度）
            base_channels: 基础通道数
            time_emb_dim: 时间嵌入维度
            num_res_blocks: 每个分辨率级别的残差块数量
        """
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 编码器路径
        self.encoder_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels = [base_channels, base_channels * 2, base_channels * 4]
        for i in range(len(channels) - 1):
            # 残差块
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(channels[i], channels[i], time_emb_dim))
            self.encoder_blocks.append(blocks)
            
            # 下采样
            self.down_samples.append(
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1)
            )
        
        # 瓶颈层
        self.bottleneck = nn.ModuleList([
            ResBlock(channels[-1], channels[-1], time_emb_dim)
            for _ in range(num_res_blocks)
        ])
        
        # 解码器路径
        self.up_samples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        channels_reversed = channels[::-1]
        for i in range(len(channels_reversed) - 1):
            # 上采样
            self.up_samples.append(
                nn.ConvTranspose2d(
                    channels_reversed[i], 
                    channels_reversed[i + 1], 
                    3, stride=2, padding=1, output_padding=1
                )
            )
            
            # 残差块（输入是上采样+跳跃连接的concatenation）
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                in_ch = channels_reversed[i + 1] * 2 if i == 0 else channels_reversed[i + 1]
                blocks.append(ResBlock(in_ch, channels_reversed[i + 1], time_emb_dim))
            self.decoder_blocks.append(blocks)
        
        # 输出层
        self.output_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """前向传播.
        
        Args:
            x: 输入图像, shape: (batch_size, in_channels, H, W)
            t: 时间参数, shape: (batch_size,), 范围 [0, 1]
            
        Returns:
            预测的速度场, shape: (batch_size, out_channels, H, W)
        """
        # 时间嵌入
        time_emb = self.time_embedding(t)
        
        # 初始卷积
        x = self.init_conv(x)
        
        # 编码器路径 - 收集跳跃连接
        skip_connections = []
        
        for blocks, down_sample in zip(self.encoder_blocks, self.down_samples):
            # 残差块
            for block in blocks:  # type: ignore
                x = block(x, time_emb)
            
            # 保存跳跃连接
            skip_connections.append(x)
            
            # 下采样
            x = down_sample(x)
        
        # 瓶颈层
        for block in self.bottleneck:
            x = block(x, time_emb)
        
        # 解码器路径
        for up_sample, blocks, skip in zip(
            self.up_samples, self.decoder_blocks, reversed(skip_connections)
        ):
            # 上采样
            x = up_sample(x)
            
            # 跳跃连接
            x = torch.cat([x, skip], dim=1)
            
            # 残差块
            for block in blocks:  # type: ignore
                x = block(x, time_emb)
        
        # 输出层
        return self.output_conv(x)


def create_test_unet(image_size: int = 28, channels: int = 1) -> SimpleUNet:
    """创建用于测试的UNet网络实例.
    
    Args:
        image_size: 图像尺寸（MNIST默认28x28）
        channels: 图像通道数（MNIST为1）
        
    Returns:
        配置好的UNet网络实例
    """
    return SimpleUNet(
        in_channels=channels,
        out_channels=channels,  # 速度场与输入相同维度
        base_channels=32,       # 较小的通道数，加速测试
        time_emb_dim=64,        # 适中的时间嵌入维度
        num_res_blocks=1        # 减少块数量，加速训练
    ) 
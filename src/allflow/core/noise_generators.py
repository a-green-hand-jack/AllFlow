"""噪声生成器模块

提供不同几何空间中的噪声生成方法，包括欧几里得空间和SO(3)旋转群。
用于Flow Matching中当源分布x_0未提供时的自动噪声生成。

Author: AllFlow Contributors  
License: MIT
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class NoiseGeneratorBase(ABC):
    """噪声生成器的抽象基类.
    
    定义了所有噪声生成器必须实现的接口。不同的几何空间
    需要不同的噪声生成策略以确保生成的噪声在对应空间中有效。
    """
    
    @abstractmethod
    def sample_like(self, target: torch.Tensor) -> torch.Tensor:
        """生成与目标张量相同形状的噪声.
        
        Args:
            target: 目标张量，用于确定形状、设备和数据类型
            
        Returns:
            噪声张量，与target相同形状
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(
        self, 
        shape: torch.Size, 
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """生成指定形状的噪声.
        
        Args:
            shape: 输出张量形状
            device: 计算设备
            dtype: 数据类型
            
        Returns:
            噪声张量
        """
        raise NotImplementedError


class GaussianNoiseGenerator(NoiseGeneratorBase):
    """高斯噪声生成器.
    
    生成标准的高斯分布噪声，适用于欧几里得空间。
    这是最常用的噪声生成器，用于向量空间中的数据。
    
    特点：
    - 标准正态分布 N(0, σ²I)
    - 可配置的标准差
    - 支持各种张量形状
    - 数值稳定
    """
    
    def __init__(self, std: float = 1.0, mean: float = 0.0):
        """初始化高斯噪声生成器.
        
        Args:
            std: 标准差，控制噪声强度
            mean: 均值，通常为0
        """
        self.std = std
        self.mean = mean
        logger.debug(f"GaussianNoiseGenerator 初始化完成: mean={mean}, std={std}")
    
    def sample_like(self, target: torch.Tensor) -> torch.Tensor:
        """生成与目标张量相同形状的高斯噪声.
        
        Args:
            target: 目标张量
            
        Returns:
            高斯噪声张量
        """
        return torch.randn_like(target) * self.std + self.mean
    
    def sample(
        self, 
        shape: torch.Size, 
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """生成指定形状的高斯噪声.
        
        Args:
            shape: 输出张量形状
            device: 计算设备  
            dtype: 数据类型
            
        Returns:
            高斯噪声张量
        """
        return torch.randn(shape, device=device, dtype=dtype) * self.std + self.mean


class SO3NoiseGenerator(NoiseGeneratorBase):
    """SO(3)旋转群噪声生成器.
    
    生成在SO(3)旋转群上均匀分布的随机旋转，使用四元数表示。
    确保生成的"噪声"是有效的单位四元数。
    
    特点：
    - 在SO(3)上均匀分布
    - 四元数表示 [w, x, y, z]
    - 自动归一化为单位四元数
    - 数值稳定的采样
    
    数学原理：
    - 使用Marsaglia方法生成均匀分布的单位四元数
    - 确保在球面S³上的均匀分布对应SO(3)的均匀分布
    """
    
    def __init__(self, eps: float = 1e-8):
        """初始化SO(3)噪声生成器.
        
        Args:
            eps: 数值稳定性常数
        """
        self.eps = eps
        logger.debug("SO3NoiseGenerator 初始化完成")
    
    def sample_uniform_quaternion(
        self, 
        batch_size: int, 
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """使用Marsaglia方法生成均匀分布的单位四元数.
        
        Args:
            batch_size: 批量大小
            device: 计算设备
            dtype: 数据类型
            
        Returns:
            单位四元数张量, shape: (batch_size, 4)
        """
        # Marsaglia方法：生成在S³球面上均匀分布的四元数
        # 1. 生成两对独立的均匀随机数
        u1 = torch.rand(batch_size, device=device, dtype=dtype)
        u2 = torch.rand(batch_size, device=device, dtype=dtype)
        u3 = torch.rand(batch_size, device=device, dtype=dtype)
        
        # 2. 计算四元数分量
        sqrt_1_u1 = torch.sqrt(1 - u1)
        sqrt_u1 = torch.sqrt(u1)
        
        theta1 = 2 * torch.pi * u2
        theta2 = 2 * torch.pi * u3
        
        w = sqrt_1_u1 * torch.cos(theta1)
        x = sqrt_u1 * torch.sin(theta2)
        y = sqrt_u1 * torch.cos(theta2)
        z = sqrt_1_u1 * torch.sin(theta1)
        
        # 3. 组合为四元数张量
        q = torch.stack([w, x, y, z], dim=1)
        
        # 4. 归一化确保单位长度
        norm = torch.norm(q, dim=1, keepdim=True)
        q = q / torch.clamp(norm, min=self.eps)
        
        return q
    
    def sample_like(self, target: torch.Tensor) -> torch.Tensor:
        """生成与目标张量相同形状的SO(3)噪声.
        
        Args:
            target: 目标四元数张量, shape: (..., 4)
            
        Returns:
            随机四元数张量, shape: (..., 4)
            
        Raises:
            ValueError: 当目标张量最后一维不是4时
        """
        if target.shape[-1] != 4:
            raise ValueError(
                f"SO(3)噪声生成需要四元数格式（最后一维为4），"
                f"得到形状: {target.shape}"
            )
        
        # 计算批量大小（将除最后一维外的所有维度展平）
        batch_size = target.numel() // 4
        
        # 生成随机四元数
        random_q = self.sample_uniform_quaternion(
            batch_size, target.device, target.dtype
        )
        
        # 重塑为目标形状
        return random_q.view(target.shape)
    
    def sample(
        self, 
        shape: torch.Size, 
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """生成指定形状的SO(3)噪声.
        
        Args:
            shape: 输出张量形状，最后一维必须为4
            device: 计算设备
            dtype: 数据类型
            
        Returns:
            随机四元数张量
            
        Raises:
            ValueError: 当形状的最后一维不是4时
        """
        if shape[-1] != 4:
            raise ValueError(
                f"SO(3)噪声生成需要四元数格式（最后一维为4），"
                f"得到形状: {shape}"
            )
        
        # 计算批量大小
        batch_size = torch.Size(shape).numel() // 4
        
        # 生成随机四元数并重塑
        random_q = self.sample_uniform_quaternion(batch_size, device, dtype)
        return random_q.view(shape)


class UniformNoiseGenerator(NoiseGeneratorBase):
    """均匀分布噪声生成器.
    
    生成指定范围内的均匀分布噪声，适用于有界的欧几里得空间。
    
    特点：
    - 均匀分布 U(min_val, max_val)
    - 可配置的值范围
    - 适用于有界数据
    """
    
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        """初始化均匀噪声生成器.
        
        Args:
            min_val: 最小值
            max_val: 最大值
        """
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) 必须小于 max_val ({max_val})")
            
        self.min_val = min_val
        self.max_val = max_val
        logger.debug(f"UniformNoiseGenerator 初始化完成: [{min_val}, {max_val}]")
    
    def sample_like(self, target: torch.Tensor) -> torch.Tensor:
        """生成与目标张量相同形状的均匀噪声.
        
        Args:
            target: 目标张量
            
        Returns:
            均匀噪声张量
        """
        return torch.rand_like(target) * (self.max_val - self.min_val) + self.min_val
    
    def sample(
        self, 
        shape: torch.Size, 
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """生成指定形状的均匀噪声.
        
        Args:
            shape: 输出张量形状
            device: 计算设备
            dtype: 数据类型
            
        Returns:
            均匀噪声张量
        """
        return torch.rand(shape, device=device, dtype=dtype) * (self.max_val - self.min_val) + self.min_val


def create_noise_generator(
    generator_type: str, **kwargs
) -> NoiseGeneratorBase:
    """便利函数：创建噪声生成器实例.
    
    Args:
        generator_type: 生成器类型，'gaussian', 'so3', 或 'uniform'
        **kwargs: 传递给生成器构造函数的参数
        
    Returns:
        噪声生成器实例
        
    Example:
        >>> # 创建高斯噪声生成器
        >>> gaussian = create_noise_generator('gaussian', std=0.5)
        >>> 
        >>> # 创建SO(3)噪声生成器
        >>> so3 = create_noise_generator('so3')
        >>> 
        >>> # 创建均匀噪声生成器
        >>> uniform = create_noise_generator('uniform', min_val=-2, max_val=2)
    """
    generator_type = generator_type.lower()
    
    if generator_type == 'gaussian':
        return GaussianNoiseGenerator(**kwargs)
    elif generator_type == 'so3':
        return SO3NoiseGenerator(**kwargs)
    elif generator_type == 'uniform':
        return UniformNoiseGenerator(**kwargs)
    else:
        raise ValueError(
            f"不支持的噪声生成器类型: {generator_type}. "
            f"支持的类型: 'gaussian', 'so3', 'uniform'"
        ) 
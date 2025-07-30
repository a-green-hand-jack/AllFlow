"""速度场实现模块

提供不同几何空间中的速度场计算方法，包括欧几里得空间和SO(3)旋转群。
用于Flow Matching中计算流的瞬时速度场。

Author: AllFlow Contributors
License: MIT
"""

import logging
from typing import Optional

import torch

from .base import VectorField

logger = logging.getLogger(__name__)


class EuclideanVectorField(VectorField):
    """欧几里得空间中的速度场计算.
    
    在欧几里得空间中，Flow Matching的速度场就是目标点与源点的简单差值。
    这是标准Flow Matching算法的速度场计算方式。
    
    特点：
    - 简单直接：v = x_1 - x_0
    - 不依赖时间或当前位置
    - 计算高效
    - 数值稳定
    """
    
    def __init__(self, eps: float = 1e-8):
        """初始化欧几里得速度场计算器.
        
        Args:
            eps: 数值稳定性常数
        """
        self.eps = eps
        logger.debug("EuclideanVectorField 初始化完成")
    
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """计算欧几里得空间速度场.
        
        Args:
            x: 当前位置张量, shape: (batch_size, *data_shape)
            t: 时间张量, shape: (batch_size,)
            
        Returns:
            速度场值, shape: (batch_size, *data_shape)
            
        Note:
            这个方法需要在调用前设置x_0和x_1，通常通过set_endpoints方法
        """
        if not hasattr(self, '_x_0') or not hasattr(self, '_x_1'):
            raise RuntimeError(
                "必须先调用set_endpoints方法设置源点和目标点"
            )
        
        # Flow Matching速度场：v = x_1 - x_0
        return self._x_1 - self._x_0
    
    def set_endpoints(self, x_0: torch.Tensor, x_1: torch.Tensor) -> None:
        """设置速度场计算的端点.
        
        Args:
            x_0: 源点, shape: (batch_size, *data_shape)
            x_1: 目标点, shape: (batch_size, *data_shape)
        """
        if x_0.shape != x_1.shape:
            raise ValueError(f"x_0和x_1形状不匹配: {x_0.shape} vs {x_1.shape}")
            
        self._x_0 = x_0
        self._x_1 = x_1


class SO3VectorField(VectorField):
    """SO(3)旋转群中的速度场计算.
    
    在SO(3)空间中，速度场需要在李代数so(3)中计算，然后通过指数映射
    转换到切空间。这确保了旋转的几何约束得到保持。
    
    特点：
    - 保持旋转群的几何约束
    - 使用李代数进行计算
    - 支持四元数表示
    - 数值稳定的对数映射
    
    数学原理：
    - 四元数 q_1 = q_0 * exp(ω*t)
    - 速度场 ω = log(q_1 * q_0^{-1})
    """
    
    def __init__(self, eps: float = 1e-6):
        """初始化SO(3)速度场计算器.
        
        Args:
            eps: 数值稳定性常数
        """
        self.eps = eps
        logger.debug("SO3VectorField 初始化完成")
    
    def quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """计算四元数共轭.
        
        Args:
            q: 四元数张量, shape: (..., 4) - [w, x, y, z]格式
            
        Returns:
            共轭四元数, shape: (..., 4) - [w, -x, -y, -z]格式
        """
        conjugate = q.clone()
        conjugate[..., 1:] = -conjugate[..., 1:]  # 翻转虚部符号
        return conjugate
    
    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """四元数乘法.
        
        Args:
            q1, q2: 四元数张量, shape: (..., 4)
            
        Returns:
            乘积四元数, shape: (..., 4)
        """
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=-1)
    
    def quaternion_log(self, q: torch.Tensor) -> torch.Tensor:
        """四元数对数映射到李代数so(3).
        
        将单位四元数映射到对应的李代数元素（角速度向量）。
        
        Args:
            q: 单位四元数, shape: (..., 4) - [w, x, y, z]格式
            
        Returns:
            李代数元素, shape: (..., 3) - 角速度向量
        """
        # 确保四元数是单位的
        q = q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=self.eps)
        
        # 提取标量和向量部分
        w = q[..., 0:1]  # shape: (..., 1)
        v = q[..., 1:4]  # shape: (..., 3)
        
        # 计算向量部分的模长
        v_norm = torch.norm(v, dim=-1, keepdim=True)  # shape: (..., 1)
        
        # 处理接近恒等四元数的情况（w ≈ ±1, v ≈ 0）
        small_angle_mask = v_norm < self.eps
        
        # 对于小角度，使用泰勒展开：log(q) ≈ v/w
        small_angle_result = v / torch.clamp(torch.abs(w), min=self.eps)
        
        # 对于一般情况：log(q) = v * atan2(||v||, |w|) / ||v||
        w_abs = torch.abs(w)
        angle = torch.atan2(v_norm, w_abs)
        general_result = v * angle / torch.clamp(v_norm, min=self.eps)
        
        # 根据情况选择结果
        result = torch.where(small_angle_mask, small_angle_result, general_result)
        
        return result
    
    def compute_angular_velocity(self, q_0: torch.Tensor, q_1: torch.Tensor) -> torch.Tensor:
        """计算从q_0到q_1的角速度.
        
        Args:
            q_0: 起始四元数, shape: (batch_size, 4)
            q_1: 目标四元数, shape: (batch_size, 4)
            
        Returns:
            角速度向量, shape: (batch_size, 3)
        """
        # 计算相对旋转：q_rel = q_1 * q_0^{-1}
        q_0_conj = self.quaternion_conjugate(q_0)
        q_rel = self.quaternion_multiply(q_1, q_0_conj)
        
        # 计算角速度：ω = log(q_rel)
        omega = self.quaternion_log(q_rel)
        
        return omega
    
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """计算SO(3)空间速度场.
        
        Args:
            x: 当前四元数, shape: (batch_size, 4)
            t: 时间张量, shape: (batch_size,)
            
        Returns:
            速度场值, shape: (batch_size, 3) - 角速度向量
        """
        if not hasattr(self, '_q_0') or not hasattr(self, '_q_1'):
            raise RuntimeError(
                "必须先调用set_endpoints方法设置源四元数和目标四元数"
            )
        
        # 验证输入是四元数
        if x.shape[-1] != 4:
            raise ValueError(f"SO(3)速度场需要四元数输入，得到形状: {x.shape}")
        
        # 计算角速度
        omega = self.compute_angular_velocity(self._q_0, self._q_1)
        
        # 在Flow Matching中，速度场是常数，不依赖于当前位置或时间
        return omega
    
    def set_endpoints(self, q_0: torch.Tensor, q_1: torch.Tensor) -> None:
        """设置速度场计算的端点四元数.
        
        Args:
            q_0: 源四元数, shape: (batch_size, 4)
            q_1: 目标四元数, shape: (batch_size, 4)
        """
        if q_0.shape != q_1.shape:
            raise ValueError(f"q_0和q_1形状不匹配: {q_0.shape} vs {q_1.shape}")
            
        if q_0.shape[-1] != 4 or q_1.shape[-1] != 4:
            raise ValueError(
                f"SO(3)速度场需要四元数输入，得到形状: q_0={q_0.shape}, q_1={q_1.shape}"
            )
        
        # 归一化四元数
        q_0_norm = q_0 / torch.clamp(torch.norm(q_0, dim=-1, keepdim=True), min=self.eps)
        q_1_norm = q_1 / torch.clamp(torch.norm(q_1, dim=-1, keepdim=True), min=self.eps)
        
        self._q_0 = q_0_norm
        self._q_1 = q_1_norm


def create_vector_field(
    vector_field_type: str, **kwargs
) -> VectorField:
    """便利函数：创建速度场实例.
    
    Args:
        vector_field_type: 速度场类型，'euclidean' 或 'so3'
        **kwargs: 传递给速度场构造函数的参数
        
    Returns:
        速度场实例
        
    Example:
        >>> # 创建欧几里得速度场
        >>> euclidean_vf = create_vector_field('euclidean')
        >>> 
        >>> # 创建SO(3)速度场
        >>> so3_vf = create_vector_field('so3', eps=1e-7)
    """
    if vector_field_type.lower() == 'euclidean':
        return EuclideanVectorField(**kwargs)
    elif vector_field_type.lower() == 'so3':
        return SO3VectorField(**kwargs)
    else:
        raise ValueError(
            f"不支持的速度场类型: {vector_field_type}. "
            f"支持的类型: 'euclidean', 'so3'"
        ) 
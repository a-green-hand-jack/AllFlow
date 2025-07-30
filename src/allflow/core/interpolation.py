"""路径插值实现模块

提供不同几何空间中的路径插值方法，包括欧几里得空间和SO(3)旋转群。
用于Flow Matching中从源分布到目标分布的插值轨迹计算。

Author: AllFlow Contributors
License: MIT
"""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

from .base import PathInterpolation

logger = logging.getLogger(__name__)


class EuclideanInterpolation(PathInterpolation):
    """欧几里得空间中的线性插值.
    
    实现标准的线性插值：x_t = (1-t)*x_0 + t*x_1
    这是最基础和最常用的插值方法，适用于向量空间中的所有数据。
    
    特点：
    - 数值稳定
    - 计算高效
    - 路径为直线
    - 保持凸组合性质
    """
    
    def __init__(self, eps: float = 1e-8):
        """初始化欧几里得插值器.
        
        Args:
            eps: 数值稳定性常数，防止除零错误
        """
        self.eps = eps
        logger.debug("EuclideanInterpolation 初始化完成")
    
    def interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """执行线性插值.
        
        Args:
            x_0: 源点, shape: (batch_size, *data_shape)
            x_1: 目标点, shape: (batch_size, *data_shape)  
            t: 插值参数, shape: (batch_size,), 范围 [0, 1]
            
        Returns:
            插值结果, shape: (batch_size, *data_shape)
            
        Note:
            使用torch.lerp确保数值稳定性，避免直接计算 (1-t)*x_0 + t*x_1
        """
        # 验证输入
        if x_0.shape != x_1.shape:
            raise ValueError(f"x_0和x_1形状不匹配: {x_0.shape} vs {x_1.shape}")
        
        if t.shape[0] != x_0.shape[0]:
            raise ValueError(f"批量大小不匹配: t={t.shape[0]}, x_0={x_0.shape[0]}")
            
        # 扩展时间维度以支持广播
        t_expanded = t.view(-1, *([1] * (x_0.dim() - 1)))
        
        # 使用torch.lerp进行数值稳定的线性插值
        return torch.lerp(x_0, x_1, t_expanded)


class SO3Interpolation(PathInterpolation):
    """SO(3)旋转群中的球面线性插值(SLERP).
    
    实现四元数的球面线性插值，用于旋转空间中的平滑插值。
    SO(3)是3D旋转群，每个元素可以用单位四元数表示。
    
    特点：
    - 保持旋转的几何性质
    - 常角速度旋转
    - 最短路径插值
    - 数值稳定的SLERP实现
    
    四元数格式: [w, x, y, z] 其中w是标量部分，[x,y,z]是向量部分
    """
    
    def __init__(self, eps: float = 1e-6, dot_threshold: float = 0.9995):
        """初始化SO(3)插值器.
        
        Args:
            eps: 数值稳定性常数
            dot_threshold: 当两个四元数点积大于此值时使用线性插值
        """
        self.eps = eps
        self.dot_threshold = dot_threshold
        logger.debug("SO3Interpolation 初始化完成")
    
    def normalize_quaternion(self, q: torch.Tensor) -> torch.Tensor:
        """归一化四元数.
        
        Args:
            q: 四元数张量, shape: (..., 4)
            
        Returns:
            归一化的单位四元数, shape: (..., 4)
        """
        norm = torch.norm(q, dim=-1, keepdim=True)
        return q / torch.clamp(norm, min=self.eps)
    
    def quaternion_dot(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """计算四元数点积.
        
        Args:
            q1, q2: 四元数张量, shape: (..., 4)
            
        Returns:
            点积结果, shape: (...)
        """
        return torch.sum(q1 * q2, dim=-1)
    
    def quaternion_slerp(
        self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """四元数球面线性插值(SLERP).
        
        Args:
            q0: 起始四元数, shape: (batch_size, 4)
            q1: 结束四元数, shape: (batch_size, 4)
            t: 插值参数, shape: (batch_size,)
            
        Returns:
            插值四元数, shape: (batch_size, 4)
        """
        # 归一化输入四元数
        q0 = self.normalize_quaternion(q0)
        q1 = self.normalize_quaternion(q1)
        
        # 计算点积
        dot = self.quaternion_dot(q0, q1)
        
        # 选择最短路径：如果点积为负，翻转其中一个四元数
        q1 = torch.where(dot.unsqueeze(-1) < 0, -q1, q1)
        dot = torch.abs(dot)
        
        # 对于非常接近的四元数，使用线性插值避免数值不稳定
        close_mask = dot > self.dot_threshold
        
        # 扩展维度用于广播
        t_expanded = t.unsqueeze(-1)
        close_mask_expanded = close_mask.unsqueeze(-1)
        
        # 线性插值结果
        linear_result = (1 - t_expanded) * q0 + t_expanded * q1
        linear_result = self.normalize_quaternion(linear_result)
        
        # SLERP插值结果
        # 计算角度
        dot_clamped = torch.clamp(dot, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(dot_clamped)
        sin_theta = torch.sin(theta)
        
        # 避免除以零
        sin_theta = torch.clamp(sin_theta, min=self.eps)
        
        # SLERP公式
        sin_t_theta = torch.sin(t * theta).unsqueeze(-1)
        sin_1mt_theta = torch.sin((1 - t) * theta).unsqueeze(-1)
        sin_theta_expanded = sin_theta.unsqueeze(-1)
        
        slerp_result = (sin_1mt_theta * q0 + sin_t_theta * q1) / sin_theta_expanded
        
        # 根据情况选择线性插值或SLERP
        result = torch.where(close_mask_expanded, linear_result, slerp_result)
        
        return self.normalize_quaternion(result)
    
    def interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """执行SO(3)插值.
        
        Args:
            x_0: 源四元数, shape: (batch_size, 4) - [w, x, y, z]格式
            x_1: 目标四元数, shape: (batch_size, 4) - [w, x, y, z]格式
            t: 插值参数, shape: (batch_size,), 范围 [0, 1]
            
        Returns:
            插值四元数, shape: (batch_size, 4)
            
        Raises:
            ValueError: 当输入不是四元数格式时
        """
        # 验证输入是四元数格式
        if x_0.shape[-1] != 4 or x_1.shape[-1] != 4:
            raise ValueError(
                f"SO(3)插值需要四元数输入 (最后一维为4), "
                f"得到: x_0.shape={x_0.shape}, x_1.shape={x_1.shape}"
            )
            
        if x_0.shape != x_1.shape:
            raise ValueError(f"x_0和x_1形状不匹配: {x_0.shape} vs {x_1.shape}")
            
        if t.shape[0] != x_0.shape[0]:
            raise ValueError(f"批量大小不匹配: t={t.shape[0]}, x_0={x_0.shape[0]}")
        
        # 执行四元数SLERP
        return self.quaternion_slerp(x_0, x_1, t)


def create_interpolation(
    interpolation_type: str, **kwargs
) -> PathInterpolation:
    """便利函数：创建插值器实例.
    
    Args:
        interpolation_type: 插值器类型，'euclidean' 或 'so3'
        **kwargs: 传递给插值器构造函数的参数
        
    Returns:
        插值器实例
        
    Example:
        >>> # 创建欧几里得插值器
        >>> euclidean = create_interpolation('euclidean')
        >>> 
        >>> # 创建SO(3)插值器
        >>> so3 = create_interpolation('so3', eps=1e-7)
    """
    if interpolation_type.lower() == 'euclidean':
        return EuclideanInterpolation(**kwargs)
    elif interpolation_type.lower() == 'so3':
        return SO3Interpolation(**kwargs)
    else:
        raise ValueError(
            f"不支持的插值类型: {interpolation_type}. "
            f"支持的类型: 'euclidean', 'so3'"
        ) 
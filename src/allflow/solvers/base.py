"""ODE求解器抽象基类定义

本模块定义了AllFlow中所有ODE求解器的统一接口，确保不同求解器实现
之间的一致性和可互换性。

核心类:
- ODESolverBase: 所有ODE求解器的抽象基类
- VectorFieldWrapper: 速度场函数的包装器
- SolverConfig: 求解器配置参数

设计原则:
- 统一的求解器接口，支持多种数值方法
- 高效的批量处理和GPU加速
- 灵活的配置和错误处理
- 与torchdiffeq等专业库的无缝集成

Author: AllFlow Contributors
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Any, Dict
from dataclasses import dataclass
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """ODE求解器配置参数.
    
    Attributes:
        method: 数值积分方法名称
        
        rtol: 相对误差容限
        atol: 绝对误差容限
        max_num_steps: 最大步数
        step_size: 固定步长（可选）
        adaptive: 是否使用自适应步长
    """
    method: str = "euler"
    rtol: float = 1e-5
    atol: float = 1e-6
    max_num_steps: int = 1000
    step_size: Optional[float] = None
    adaptive: bool = True


class VectorFieldWrapper:
    """速度场函数的包装器.
    
    将Flow Matching的速度场函数包装为符合ODE求解器要求的格式。
    处理时间参数的正确传递和批量操作。
    """
    
    def __init__(
        self, 
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reverse_time: bool = False
    ):
        """初始化速度场包装器.
        
        Args:
            vector_field: 速度场函数 u(x, t)
            reverse_time: 是否反向时间积分（用于逆过程）
        """
        self.vector_field = vector_field
        self.reverse_time = reverse_time
    
    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """调用速度场函数.
        
        Args:
            t: 时间参数，shape: () 或 (1,)
            x: 状态向量，shape: (batch_size, *state_shape)
            
        Returns:
            速度向量，shape: (batch_size, *state_shape)
        """
        # 确保时间参数具有正确的形状
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # 扩展时间参数以匹配批量大小
        batch_size = x.shape[0]
        t_expanded = t.expand(batch_size)
        
        # 反向时间积分：t -> 1-t
        if self.reverse_time:
            t_expanded = 1.0 - t_expanded
        
        return self.vector_field(x, t_expanded)


class ODESolverBase(ABC):
    """ODE求解器的抽象基类.
    
    定义所有ODE求解器必须实现的核心接口。提供统一的API来求解
    形式为 dx/dt = f(x, t) 的常微分方程。
    
    Attributes:
        config: 求解器配置参数
        device: 计算设备
        dtype: 数据类型
    """
    
    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """初始化ODE求解器.
        
        Args:
            config: 求解器配置，默认使用默认配置
            device: 计算设备，默认自动检测
            dtype: 数据类型，默认为torch.float32
        """
        self.config = config or SolverConfig()
        self.dtype = dtype or torch.float32
        
        # 设备检测
        if device is None:
            device = self._detect_device()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        logger.info(f"ODE求解器初始化: {self.__class__.__name__}, device={self.device}")
    
    def _detect_device(self) -> torch.device:
        """自动检测最优计算设备."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    @abstractmethod
    def solve(
        self,
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        t_span: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """求解ODE系统.
        
        Args:
            vector_field: 速度场函数 f(x, t)
            x0: 初始条件，shape: (batch_size, *state_shape)
            t_span: 时间区间，shape: (num_times,)，单调递增
            **kwargs: 额外的求解器参数
            
        Returns:
            解轨迹，shape: (num_times, batch_size, *state_shape)
            
        Raises:
            ValueError: 当输入参数不合法时
            RuntimeError: 当求解过程失败时
        """
        raise NotImplementedError("子类必须实现solve方法")
    
    def integrate_forward(
        self,
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        num_steps: int = 100,
        **kwargs: Any
    ) -> torch.Tensor:
        """前向积分：从t=0到t=1.
        
        Args:
            vector_field: 速度场函数
            x0: 初始状态
            num_steps: 积分步数
            **kwargs: 额外参数
            
        Returns:
            最终状态，shape: (batch_size, *state_shape)
        """
        t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device, dtype=self.dtype)
        trajectory = self.solve(vector_field, x0, t_span, **kwargs)
        return trajectory[-1]  # 返回最终时刻的状态
    
    def integrate_backward(
        self,
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x1: torch.Tensor,
        num_steps: int = 100,
        **kwargs: Any
    ) -> torch.Tensor:
        """反向积分：从t=1到t=0.
        
        Args:
            vector_field: 速度场函数
            x1: 最终状态
            num_steps: 积分步数
            **kwargs: 额外参数
            
        Returns:
            初始状态，shape: (batch_size, *state_shape)
        """
        # 使用反向时间的速度场包装器
        reverse_field = VectorFieldWrapper(vector_field, reverse_time=True)
        
        t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device, dtype=self.dtype)
        trajectory = self.solve(reverse_field, x1, t_span, **kwargs)
        return trajectory[-1]  # 返回最终时刻的状态
    
    def validate_inputs(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor
    ) -> None:
        """验证求解器输入参数.
        
        Args:
            x0: 初始条件
            t_span: 时间区间
            
        Raises:
            ValueError: 当输入不合法时
        """
        # 检查初始条件
        if x0.dim() < 1:
            raise ValueError(f"初始条件x0至少需要1维，得到: {x0.dim()}")
        
        # 检查时间区间
        if t_span.dim() != 1:
            raise ValueError(f"时间区间t_span必须是1维张量，得到: {t_span.shape}")
        
        if len(t_span) < 2:
            raise ValueError(f"时间区间至少需要2个时间点，得到: {len(t_span)}")
        
        # 检查时间区间单调性
        if not torch.all(t_span[1:] >= t_span[:-1]):
            raise ValueError("时间区间必须单调非递减")
        
        # 检查时间范围
        if t_span[0] < 0 or t_span[-1] > 1:
            logger.warning(f"时间区间超出[0,1]范围: [{t_span[0]:.4f}, {t_span[-1]:.4f}]")
    
    def to_device(self, *tensors: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """将张量移动到求解器设备."""
        moved_tensors = [t.to(device=self.device, dtype=self.dtype) for t in tensors]
        
        if len(moved_tensors) == 1:
            return moved_tensors[0]
        return tuple(moved_tensors) 
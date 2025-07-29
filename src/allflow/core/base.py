"""Flow Matching核心基类定义

本模块定义了所有Flow Matching算法的抽象基类和核心接口，确保不同
算法变体之间的一致性和互操作性。

核心类：
- FlowMatchingBase: 所有Flow Matching算法的抽象基类
- VectorField: 速度场的抽象表示
- PathInterpolation: 路径插值的抽象接口

设计要求：
- 所有方法必须支持批量处理
- 强制使用PyTorch张量操作，禁止Python循环
- 支持任意设备(CPU/GPU)和精度(float32/float16)
- 数值稳定性优先，包含边界条件处理

数学基础：
Flow Matching的核心是学习概率路径 p_t(x) 从简单分布 p_0 到复杂分布 p_1，
通过最小化流匹配损失来训练速度场 u_t(x)。

Author: AllFlow Contributors
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)


class FlowMatchingBase(ABC):
    """Flow Matching算法的抽象基类.

    定义所有Flow Matching变体必须实现的核心接口。子类必须实现
    compute_vector_field、sample_trajectory和compute_loss方法。

    Attributes:
        device: 计算设备 (CPU或GPU)
        dtype: 张量数据类型
        sigma_min: 最小噪声水平，用于数值稳定性

    Note:
        所有实现必须避免Python循环，使用纯PyTorch张量操作
        以获得最佳性能。
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        sigma_min: float = 1e-5,
    ) -> None:
        """初始化Flow Matching基类.

        Args:
            device: 计算设备，如'cuda'、'mps'或'cpu'，None时自动检测
            dtype: 张量数据类型，默认为torch.float32
            sigma_min: 最小噪声水平，防止数值不稳定

        Raises:
            ValueError: 当sigma_min不是正数时
        """
        if sigma_min <= 0:
            raise ValueError(f"sigma_min必须是正数，得到: {sigma_min}")

        self.sigma_min = sigma_min
        self.dtype = dtype or torch.float32

        # 智能设备检测
        if device is None:
            device = self._detect_device()

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        logger.info(f"Flow Matching初始化: device={self.device}, dtype={self.dtype}")

    def _detect_device(self) -> torch.device:
        """智能检测最优计算设备.

        按优先级检测：CUDA > MPS > CPU

        Returns:
            检测到的最优设备
        """
        if torch.cuda.is_available():
            logger.debug("检测到CUDA设备，使用GPU加速")
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.debug("检测到MPS设备，使用Apple Silicon加速")
            return torch.device("mps")
        else:
            logger.debug("使用CPU设备")
            return torch.device("cpu")

    def to_device(
        self, *tensors: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """将张量移动到指定设备.

        Args:
            *tensors: 要移动的张量

        Returns:
            移动后的张量，如果只有一个张量则直接返回，否则返回元组
        """
        moved_tensors = [t.to(device=self.device, dtype=self.dtype) for t in tensors]

        if len(moved_tensors) == 1:
            return moved_tensors[0]
        return tuple(moved_tensors)

    @abstractmethod
    def compute_vector_field(
        self, x_t: torch.Tensor, t: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """计算Flow Matching在时间t处的速度场.

        这是Flow Matching的核心方法，计算给定时间和位置处的流速度。
        不同的Flow Matching变体在此方法中实现其特定的速度场计算逻辑。

        Args:
            x_t: 当前位置张量, shape: (batch_size, *data_shape)
            t: 时间参数张量, shape: (batch_size,), 取值范围 [0, 1]
            **kwargs: 算法特定的额外参数

        Returns:
            速度场张量, shape: (batch_size, *data_shape)

        Raises:
            ValueError: 当输入张量形状不匹配或时间参数超出[0,1]范围时
            RuntimeError: 当计算过程中出现数值问题时
        """
        raise NotImplementedError("子类必须实现compute_vector_field方法")

    @abstractmethod
    def sample_trajectory(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """根据插值路径采样轨迹点.

        根据Flow Matching的理论，在给定的源点x_0、目标点x_1和时间t的情况下，
        计算插值轨迹上的点。这是训练时计算损失函数的基础。

        Args:
            x_0: 源分布采样点, shape: (batch_size, *data_shape)
            x_1: 目标分布采样点, shape: (batch_size, *data_shape)
            t: 时间参数, shape: (batch_size,), 取值范围 [0, 1]
            **kwargs: 算法特定的插值参数

        Returns:
            轨迹点张量, shape: (batch_size, *data_shape)

        Note:
            标准的线性插值为: x_t = (1-t)*x_0 + t*x_1
            某些变体可能包含额外的随机项或条件依赖
        """
        raise NotImplementedError("子类必须实现sample_trajectory方法")

    @abstractmethod
    def compute_loss(
        self, x_0: torch.Tensor, x_1: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """计算Flow Matching训练损失.

        计算用于训练神经网络的Flow Matching损失函数。损失函数度量
        预测的速度场与真实速度场之间的差异。

        Args:
            x_0: 源分布采样, shape: (batch_size, *data_shape)
            x_1: 目标分布采样, shape: (batch_size, *data_shape)
            **kwargs: 损失计算的额外参数

        Returns:
            标量损失值, shape: ()

        Note:
            损失计算通常涉及在随机时间点t采样轨迹，然后计算
            预测速度场与真实速度场的L2距离。
        """
        raise NotImplementedError("子类必须实现compute_loss方法")

    def validate_inputs(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> None:
        """验证输入张量的有效性.

        检查输入张量的形状、设备、数据类型是否符合要求，
        以及时间参数是否在有效范围内。

        Args:
            x_0: 源分布张量
            x_1: 目标分布张量
            t: 时间参数张量

        Raises:
            ValueError: 当输入不符合要求时
        """
        # 检查张量形状一致性
        if x_0.shape != x_1.shape:
            raise ValueError(
                f"x_0和x_1的形状必须相同，得到: x_0={x_0.shape}, x_1={x_1.shape}"
            )

        # 检查批量维度一致性
        batch_size = x_0.shape[0]
        if t.shape[0] != batch_size:
            raise ValueError(
                f"时间参数t的批量大小必须与x_0/x_1一致，得到: "
                f"t.shape[0]={t.shape[0]}, batch_size={batch_size}"
            )

        # 检查时间参数维度
        if t.dim() != 1:
            raise ValueError(f"时间参数t必须是1维张量，得到形状: {t.shape}")

        # 检查时间参数范围
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(
                f"时间参数t必须在[0,1]范围内，得到范围: [{t.min():.4f}, {t.max():.4f}]"
            )

        # 检查张量是否包含NaN或Inf
        for name, tensor in [("x_0", x_0), ("x_1", x_1), ("t", t)]:
            if torch.isnan(tensor).any():
                raise ValueError(f"输入张量{name}包含NaN值")
            if torch.isinf(tensor).any():
                raise ValueError(f"输入张量{name}包含Inf值")

    def validate_vector_field_output(
        self, velocity: torch.Tensor, expected_shape: torch.Size
    ) -> None:
        """验证速度场输出的有效性.

        Args:
            velocity: 速度场张量
            expected_shape: 期望的输出形状

        Raises:
            ValueError: 当输出不符合要求时
        """
        if velocity.shape != expected_shape:
            raise ValueError(
                f"速度场输出形状不正确，期望: {expected_shape}, 得到: {velocity.shape}"
            )

        if torch.isnan(velocity).any():
            raise ValueError("速度场输出包含NaN值")

        if torch.isinf(velocity).any():
            raise ValueError("速度场输出包含Inf值")


class VectorField(ABC):
    """速度场的抽象表示.

    封装Flow Matching中的速度场概念，提供统一的接口来计算
    不同位置和时间处的流速度。
    """

    @abstractmethod
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """计算指定位置和时间的速度场值.

        Args:
            x: 位置张量, shape: (batch_size, *data_shape)
            t: 时间张量, shape: (batch_size,)

        Returns:
            速度场值, shape: (batch_size, *data_shape)
        """
        raise NotImplementedError


class PathInterpolation(ABC):
    """路径插值的抽象接口.

    定义从源分布到目标分布的插值路径。不同的Flow Matching
    变体可能使用不同的插值策略。
    """

    @abstractmethod
    def interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """执行路径插值.

        Args:
            x_0: 源点, shape: (batch_size, *data_shape)
            x_1: 目标点, shape: (batch_size, *data_shape)
            t: 插值参数, shape: (batch_size,), 范围 [0, 1]

        Returns:
            插值结果, shape: (batch_size, *data_shape)
        """
        raise NotImplementedError

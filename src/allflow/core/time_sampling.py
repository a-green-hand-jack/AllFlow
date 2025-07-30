"""时间采样器模块

本模块定义了Flow Matching中时间参数t的采样策略。
不同的Flow Matching变体可能需要不同的时间采样分布来优化训练效果。

核心组件：
- TimeSamplerBase: 时间采样器的抽象基类
- UniformTimeSampler: 均匀分布采样器（标准Flow Matching）
- NormalTimeSampler: 正态分布采样器
- ExponentialTimeSampler: 指数分布采样器
- ImportanceSampler: 重要性采样器

Author: AllFlow Contributors
License: MIT
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import torch

logger = logging.getLogger(__name__)


class TimeSamplerBase(ABC):
    """时间采样器抽象基类.

    定义了Flow Matching中时间参数t的采样接口。
    所有时间采样器都必须实现sample方法，返回[0,1]范围内的时间参数。

    Args:
        device: 计算设备
        dtype: 数据类型
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """初始化时间采样器.

        Args:
            device: 计算设备，默认使用cpu
            dtype: 数据类型，默认使用float32
        """
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32

    @abstractmethod
    def sample(self, batch_size: int) -> torch.Tensor:
        """采样时间参数.

        Args:
            batch_size: 批量大小

        Returns:
            时间参数张量, shape: (batch_size,), 取值范围 [0, 1]
        """
        pass

    def __repr__(self) -> str:
        """返回采样器的字符串表示."""
        return f"{self.__class__.__name__}(device={self.device}, dtype={self.dtype})"


class UniformTimeSampler(TimeSamplerBase):
    """均匀分布时间采样器.

    在[0, 1]区间内均匀采样时间参数，这是标准Flow Matching的默认选择。

    数学形式: t ~ Uniform(0, 1)

    Example:
        >>> sampler = UniformTimeSampler()
        >>> t = sampler.sample(batch_size=32)
        >>> assert t.shape == (32,) and 0 <= t.min() and t.max() <= 1
    """

    def sample(self, batch_size: int) -> torch.Tensor:
        """均匀采样时间参数.

        Args:
            batch_size: 批量大小

        Returns:
            均匀分布的时间参数, shape: (batch_size,)
        """
        return torch.rand(batch_size, device=self.device, dtype=self.dtype)


class NormalTimeSampler(TimeSamplerBase):
    """正态分布时间采样器.

    使用截断正态分布采样时间参数，可以在特定时间点附近集中采样。
    适用于需要在某个时间区域重点训练的场景。

    数学形式: t ~ TruncatedNormal(mean, std, [0, 1])

    Args:
        mean: 正态分布均值，默认0.5
        std: 正态分布标准差，默认0.2

    Example:
        >>> sampler = NormalTimeSampler(mean=0.3, std=0.1)
        >>> t = sampler.sample(batch_size=32)
        >>> # 大部分采样点会集中在0.3附近
    """

    def __init__(
        self,
        mean: float = 0.5,
        std: float = 0.2,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """初始化正态分布采样器.

        Args:
            mean: 正态分布均值
            std: 正态分布标准差
            device: 计算设备
            dtype: 数据类型
        """
        super().__init__(device=device, dtype=dtype)
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int) -> torch.Tensor:
        """正态分布采样时间参数.

        Args:
            batch_size: 批量大小

        Returns:
            截断正态分布的时间参数, shape: (batch_size,)
        """
        t = torch.normal(
            mean=self.mean,
            std=self.std,
            size=(batch_size,),
            device=self.device,
            dtype=self.dtype,
        )
        # 截断到[0, 1]区间
        return torch.clamp(t, min=0.0, max=1.0)


class ExponentialTimeSampler(TimeSamplerBase):
    """指数分布时间采样器.

    使用指数分布采样时间参数，可以偏向早期（rate>1）或晚期（rate<1）时间。
    适用于需要重点关注轨迹起始或结束阶段的场景。

    数学形式: t = F^(-1)(u) where u ~ Uniform(0,1), F是指数分布CDF

    Args:
        rate: 指数分布参数，rate>1偏向早期，rate<1偏向晚期

    Example:
        >>> sampler = ExponentialTimeSampler(rate=2.0)  # 偏向早期时间
        >>> t = sampler.sample(batch_size=32)
    """

    def __init__(
        self,
        rate: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """初始化指数分布采样器.

        Args:
            rate: 指数分布参数
            device: 计算设备
            dtype: 数据类型
        """
        super().__init__(device=device, dtype=dtype)
        if rate <= 0:
            raise ValueError(f"rate必须为正数，得到: {rate}")
        self.rate = rate

    def sample(self, batch_size: int) -> torch.Tensor:
        """指数分布采样时间参数.

        Args:
            batch_size: 批量大小

        Returns:
            指数分布的时间参数, shape: (batch_size,)
        """
        # 使用逆变换采样: t = -ln(1-u)/rate，其中u ~ Uniform(0,1)
        u = torch.rand(batch_size, device=self.device, dtype=self.dtype)
        t = -torch.log(1 - u + 1e-8) / self.rate  # 添加小常数避免log(0)

        # 截断到[0, 1]区间
        return torch.clamp(t, min=0.0, max=1.0)


class ImportanceTimeSampler(TimeSamplerBase):
    """重要性采样器.

    根据预定义的重要性权重函数采样时间参数。
    可以根据训练需求在不同时间点分配不同的采样概率。

    Args:
        importance_fn: 重要性权重函数，输入时间t，输出权重
        num_samples: 用于构建采样分布的离散点数

    Example:
        >>> # 在中间时间点重点采样
        >>> importance_fn = lambda t: torch.exp(-4 * (t - 0.5)**2)
        >>> sampler = ImportanceTimeSampler(importance_fn=importance_fn)
        >>> t = sampler.sample(batch_size=32)
    """

    def __init__(
        self,
        importance_fn: Callable[[torch.Tensor], torch.Tensor],
        num_samples: int = 1000,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """初始化重要性采样器.

        Args:
            importance_fn: 重要性权重函数
            num_samples: 离散化采样点数
            device: 计算设备
            dtype: 数据类型
        """
        super().__init__(device=device, dtype=dtype)
        self.importance_fn = importance_fn
        self.num_samples = num_samples

        # 预计算累积分布函数
        self._precompute_cdf()

    def _precompute_cdf(self) -> None:
        """预计算累积分布函数用于逆变换采样."""
        # 创建均匀网格
        t_grid = torch.linspace(
            0, 1, self.num_samples, device=self.device, dtype=self.dtype
        )

        # 计算重要性权重
        weights = self.importance_fn(t_grid)
        weights = torch.clamp(weights, min=1e-8)  # 避免零权重

        # 归一化得到概率密度
        pdf = weights / weights.sum()

        # 计算累积分布函数
        self.cdf_values = torch.cumsum(pdf, dim=0)
        self.t_grid = t_grid

    def sample(self, batch_size: int) -> torch.Tensor:
        """重要性采样时间参数.

        Args:
            batch_size: 批量大小

        Returns:
            重要性采样的时间参数, shape: (batch_size,)
        """
        # 生成均匀随机数
        u = torch.rand(batch_size, device=self.device, dtype=self.dtype)

        # 使用逆变换采样
        # 对每个u值，找到对应的t值
        indices = torch.searchsorted(self.cdf_values, u, right=False)
        indices = torch.clamp(indices, 0, len(self.t_grid) - 1)

        return self.t_grid[indices]


def create_time_sampler(
    sampler_type: str = "uniform",
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> TimeSamplerBase:
    """创建时间采样器的便捷函数.

    Args:
        sampler_type: 采样器类型，支持'uniform', 'normal', 'exponential', 'importance'
        device: 计算设备
        dtype: 数据类型
        **kwargs: 采样器特定参数

    Returns:
        时间采样器实例

    Example:
        >>> sampler = create_time_sampler('normal', mean=0.3, std=0.1)
        >>> t = sampler.sample(32)
    """
    samplers = {
        "uniform": UniformTimeSampler,
        "normal": NormalTimeSampler,
        "exponential": ExponentialTimeSampler,
        "importance": ImportanceTimeSampler,
    }

    if sampler_type not in samplers:
        raise ValueError(
            f"不支持的采样器类型: {sampler_type}，支持的类型: {list(samplers.keys())}"
        )

    sampler_class = samplers[sampler_type]
    return sampler_class(device=device, dtype=dtype, **kwargs)

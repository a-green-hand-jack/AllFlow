"""基础Flow Matching算法实现

本模块实现标准的Flow Matching算法，这是所有Flow Matching变体的基础。
基于Lipman et al. (2023)的"Flow Matching for Generative Modeling"论文。

核心功能：
- 实现标准的线性插值路径
- 计算Flow Matching速度场 u_t(x) = (x_1 - x_0) / (1 - σ_min)
- 提供训练损失计算
- 支持批量处理和GPU加速

数学基础：
给定源分布p_0(x)和目标分布p_1(x)，Flow Matching学习一个速度场u_t(x)，
使得遵循ODE dx/dt = u_t(x)的轨迹能够将p_0变换为p_1。

性能特点：
- 零Python循环，纯张量操作实现
- 数值稳定的边界条件处理
- 支持混合精度训练
- 内存优化的梯度计算

Author: AllFlow Contributors
License: MIT

Reference:
    Lipman, Y., et al. (2023). Flow Matching for Generative Modeling.
    arXiv preprint arXiv:2210.02747.
"""

import logging
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from allflow.core.base import FlowMatchingBase
from allflow.core.time_sampling import TimeSamplerBase, UniformTimeSampler

logger = logging.getLogger(__name__)


class FlowMatching(FlowMatchingBase):
    """标准Flow Matching算法实现.

    实现基础的Flow Matching算法，包括线性插值路径和标准速度场计算。
    这是所有其他Flow Matching变体的基础实现。

    算法核心：
    1. 线性插值路径：x_t = (1-t)*x_0 + t*x_1
    2. 速度场计算：u_t(x) = (x_1 - x_0)
    3. 损失函数：E[||u_θ(x_t, t) - u_t(x)||²]

    新特性（2024-07-29改进）：
    - 灵活的时间采样策略：支持均匀、正态、指数、重要性采样
    - 统一的模型接口：支持条件模型和额外参数
    - 简化的API：移除了不必要的多时间步参数

    Args:
        device: 计算设备，默认自动检测
        dtype: 数据类型，默认float32
        sigma_min: 最小噪声水平，用于数值稳定性
        time_sampler: 时间采样器，默认使用均匀分布

    Example:
        >>> # 基础用法
        >>> flow = FlowMatching(device='cuda')
        >>> x_0 = torch.randn(32, 128, device='cuda')
        >>> x_1 = torch.randn(32, 128, device='cuda')
        >>> loss = flow.compute_loss(x_0, x_1, model=my_velocity_model)

        >>> # 使用自定义时间采样器
        >>> from allflow.core.time_sampling import NormalTimeSampler
        >>> sampler = NormalTimeSampler(mean=0.3, std=0.1)
        >>> flow = FlowMatching(time_sampler=sampler)

        >>> # 使用条件模型
        >>> from allflow.core.model_interface import ConditionalModelWrapper
        >>> model_wrapper = ConditionalModelWrapper(model, condition=class_labels)
        >>> loss = flow.compute_loss(x_0, x_1, model=model_wrapper)

    Note:
        所有操作都使用批量化的PyTorch张量操作，避免了Python循环，
        确保最佳的计算性能和GPU利用率。支持灵活的模型输入和时间采样策略。
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        sigma_min: float = 1e-5,
        time_sampler: Optional[TimeSamplerBase] = None,
    ) -> None:
        """初始化Flow Matching算法.

        Args:
            device: 计算设备，如'cuda'、'mps'或'cpu'
            dtype: 张量数据类型，默认为torch.float32
            sigma_min: 最小噪声水平，防止数值不稳定
            time_sampler: 时间采样器，默认使用均匀分布采样
        """
        super().__init__(device=device, dtype=dtype, sigma_min=sigma_min)

        # 设置时间采样器
        if time_sampler is None:
            self.time_sampler = UniformTimeSampler(device=self.device, dtype=self.dtype)
        else:
            self.time_sampler = time_sampler
            # 确保采样器使用正确的设备和数据类型
            self.time_sampler.device = self.device
            self.time_sampler.dtype = self.dtype

        logger.info(f"标准Flow Matching算法初始化完成，时间采样器: {self.time_sampler}")

    def compute_vector_field(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        x_1: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """计算Flow Matching速度场.

        实现标准的Flow Matching速度场计算。对于Flow Matching，
        速度场就是目标点与源点的差值：u_t(x) = x_1 - x_0

        Args:
            x_t: 当前位置张量, shape: (batch_size, *data_shape)
            t: 时间参数张量, shape: (batch_size,), 取值范围 [0, 1]
            x_0: 源分布采样点, shape: (batch_size, *data_shape)
            x_1: 目标分布采样点, shape: (batch_size, *data_shape)
            **kwargs: 额外参数（保留用于子类扩展）

        Returns:
            速度场张量, shape: (batch_size, *data_shape)

        Raises:
            ValueError: 当x_0或x_1为None时，或输入形状不匹配时

        Note:
            Flow Matching的速度场不依赖于当前位置x_t，只依赖于源点和目标点。
            这使得Flow Matching相比其他方法更加简单和高效。
        """
        if x_0 is None or x_1 is None:
            raise ValueError("Flow Matching需要提供x_0和x_1来计算速度场")

        # 验证输入
        self.validate_inputs(x_0, x_1, t)

        # 确保所有张量在正确的设备上
        device_result = self.to_device(x_0, x_1, x_t, t)
        if isinstance(device_result, tuple):
            x_0, x_1, x_t, t = device_result
        else:
            # 这种情况不应该发生，因为我们传入了多个张量
            raise RuntimeError("to_device应该返回多个张量的元组")

        # Flow Matching的速度场：u_t(x) = x_1 - x_0
        # 这是一个常数向量场，不依赖于时间t或当前位置x_t
        velocity = x_1 - x_0

        # 验证输出形状
        self.validate_vector_field_output(velocity, x_t.shape)

        return velocity

    def sample_trajectory(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """采样线性插值轨迹.

        实现标准的线性插值路径：x_t = (1-t)*x_0 + t*x_1
        这是Flow Matching中最基础和最重要的操作。

        Args:
            x_0: 源点, shape: (batch_size, *data_shape)
            x_1: 目标点, shape: (batch_size, *data_shape)
            t: 插值参数, shape: (batch_size,), 范围 [0, 1]
            **kwargs: 额外参数（保留用于子类扩展）

        Returns:
            插值轨迹点, shape: (batch_size, *data_shape)

        Note:
            线性插值确保：
            - 当t=0时，返回x_0（源分布）
            - 当t=1时，返回x_1（目标分布）
            - 轨迹是从x_0到x_1的直线
        """
        # 验证输入
        self.validate_inputs(x_0, x_1, t)

        # 确保所有张量在正确的设备上
        device_result = self.to_device(x_0, x_1, t)
        if isinstance(device_result, tuple):
            x_0, x_1, t = device_result
        else:
            # 这种情况不应该发生，因为我们传入了多个张量
            raise RuntimeError("to_device应该返回多个张量的元组")

        # 将时间参数扩展到正确的形状以支持广播
        # t: (batch_size,) -> (batch_size, 1, 1, ...) 匹配x_0的维度
        t_expanded = t.view(-1, *([1] * (x_0.dim() - 1)))

        # 线性插值：x_t = (1-t)*x_0 + t*x_1
        # 使用torch.lerp实现数值稳定的插值
        x_t = torch.lerp(x_0, x_1, t_expanded)

        return x_t

    def prepare_training_data(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为训练准备数据：采样时间并计算轨迹点.

        这是一个便利方法，帮助用户准备compute_loss所需的数据。
        它使用内置的时间采样器采样时间点，并计算相应的轨迹点。

        Args:
            x_0: 源分布采样, shape: (batch_size, *data_shape)
            x_1: 目标分布采样, shape: (batch_size, *data_shape)
            batch_size: 批量大小，如果为None则使用x_0的批量大小

        Returns:
            Tuple包含:
            - x_t: 轨迹点, shape: (batch_size, *data_shape)
            - t: 采样的时间点, shape: (batch_size,)
            - true_velocity: 真实速度场, shape: (batch_size, *data_shape)

        Note:
            这个方法简化了训练循环的实现，用户可以：
            1. 调用此方法获取x_t, t, true_velocity
            2. 使用x_t和t调用模型获取predicted_velocity
            3. 调用compute_loss(x_0, x_1, t, predicted_velocity)
        """
        if batch_size is None:
            batch_size = x_0.shape[0]

        # 确保张量在正确的设备上
        device_result = self.to_device(x_0, x_1)
        if isinstance(device_result, tuple):
            x_0, x_1 = device_result
        else:
            raise RuntimeError("to_device应该返回两个张量的元组")

        # 使用时间采样器采样时间点
        t = self.time_sampler.sample(batch_size)

        # 采样轨迹点
        x_t = self.sample_trajectory(x_0, x_1, t)

        # 计算真实速度场
        true_velocity = self.compute_vector_field(x_t, t, x_0=x_0, x_1=x_1)

        return x_t, t, true_velocity

    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        predicted_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """计算Flow Matching训练损失.

        计算预测速度场与真实速度场之间的L2损失函数。
        损失函数为：L = ||predicted_velocity - true_velocity||²

        Args:
            x_0: 源分布采样, shape: (batch_size, *data_shape)
            x_1: 目标分布采样, shape: (batch_size, *data_shape)
            t: 时间参数, shape: (batch_size,), 范围 [0, 1]
            predicted_velocity: 模型预测的速度场, shape: (batch_size, *data_shape)

        Returns:
            标量损失值, shape: ()

        Note:
            这个方法现在专注于纯算法逻辑，不涉及具体的神经网络实现。
            模型调用和预测应该在外部完成，这里只计算损失。
        """
        # 验证输入
        self.validate_inputs(x_0, x_1, t)

        # 确保张量在正确的设备上
        device_result = self.to_device(x_0, x_1, t, predicted_velocity)
        if isinstance(device_result, tuple):
            x_0, x_1, t, predicted_velocity = device_result
        else:
            raise RuntimeError("to_device应该返回四个张量的元组")

        # 采样轨迹点
        x_t = self.sample_trajectory(x_0, x_1, t)

        # 计算真实速度场
        true_velocity = self.compute_vector_field(x_t, t, x_0=x_0, x_1=x_1)

        # 验证预测输出的形状
        self.validate_vector_field_output(predicted_velocity, x_t.shape)

        # 计算L2损失
        loss = F.mse_loss(predicted_velocity, true_velocity, reduction="mean")

        return loss

    def generate_sample(
        self,
        x_0: torch.Tensor,
        velocity_field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_steps: int = 100,
        method: str = "euler",
    ) -> torch.Tensor:
        """使用速度场函数生成样本.

        通过数值求解ODE dx/dt = v(x, t)来生成样本，
        从简单分布（如高斯噪声）生成复杂数据。

        Args:
            x_0: 初始噪声, shape: (batch_size, *data_shape)
            velocity_field_fn: 速度场函数，接收(x_t, t)返回速度场
            num_steps: 积分步数，更多步数通常得到更好的质量
            method: 数值积分方法，'euler'或'heun'

        Returns:
            生成的样本, shape: (batch_size, *data_shape)

        Note:
            这个方法现在专注于纯ODE积分逻辑，不涉及具体的神经网络。
            速度场函数应该在外部定义，通常是训练好的模型的包装。
            建议在实际应用中使用更高级的ODE求解器（如torchdiffeq）。
        """
        x_0_result = self.to_device(x_0)
        if isinstance(x_0_result, tuple):
            x_0 = x_0_result[0]
        else:
            x_0 = x_0_result

        x = x_0.clone()
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_current = step * dt
            t_tensor = torch.full(
                (x.shape[0],), t_current, device=self.device, dtype=self.dtype
            )

            velocity = velocity_field_fn(x, t_tensor)

            if method == "euler":
                x = x + dt * velocity
            elif method == "heun":
                # Heun方法（改进的Euler）
                k1 = velocity
                x_temp = x + dt * k1
                t_next = (step + 1) * dt
                t_next_tensor = torch.full(
                    (x.shape[0],), t_next, device=self.device, dtype=self.dtype
                )
                k2 = velocity_field_fn(x_temp, t_next_tensor)
                x = x + 0.5 * dt * (k1 + k2)
            else:
                raise ValueError(f"不支持的积分方法: {method}")

        return x

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
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F

from ..core.base import FlowMatchingBase

logger = logging.getLogger(__name__)


class FlowMatching(FlowMatchingBase):
    """标准Flow Matching算法实现.

    实现基础的Flow Matching算法，包括线性插值路径和标准速度场计算。
    这是所有其他Flow Matching变体的基础实现。

    算法核心：
    1. 线性插值路径：x_t = (1-t)*x_0 + t*x_1
    2. 速度场计算：u_t(x) = (x_1 - x_0)
    3. 损失函数：E[||u_θ(x_t, t) - u_t(x)||²]

    Args:
        device: 计算设备，默认自动检测
        dtype: 数据类型，默认float32
        sigma_min: 最小噪声水平，用于数值稳定性

    Example:
        >>> flow = FlowMatching(device='cuda')
        >>> x_0 = torch.randn(32, 128, device='cuda')
        >>> x_1 = torch.randn(32, 128, device='cuda')
        >>> t = torch.rand(32, device='cuda')
        >>> x_t = flow.sample_trajectory(x_0, x_1, t)
        >>> velocity = flow.compute_vector_field(x_t, t, x_0=x_0, x_1=x_1)
        >>> loss = flow.compute_loss(x_0, x_1, model=my_velocity_model)

    Note:
        所有操作都使用批量化的PyTorch张量操作，避免了Python循环，
        确保最佳的计算性能和GPU利用率。
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        sigma_min: float = 1e-5,
    ) -> None:
        """初始化Flow Matching算法.

        Args:
            device: 计算设备，如'cuda'、'mps'或'cpu'
            dtype: 张量数据类型，默认为torch.float32
            sigma_min: 最小噪声水平，防止数值不稳定
        """
        super().__init__(device=device, dtype=dtype, sigma_min=sigma_min)
        logger.info("标准Flow Matching算法初始化完成")

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

    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        model: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        num_timesteps: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor:
        """计算Flow Matching训练损失.

        计算Flow Matching的L2损失函数，用于训练神经网络速度场预测器。
        损失函数为：L = E_t[||u_θ(x_t, t) - u_t(x)||²]

        Args:
            x_0: 源分布采样, shape: (batch_size, *data_shape)
            x_1: 目标分布采样, shape: (batch_size, *data_shape)
            model: 神经网络速度场预测器，输入(x_t, t)，输出预测速度场
            num_timesteps: 用于损失计算的时间步数，默认为1
            **kwargs: 额外参数

        Returns:
            标量损失值, shape: ()

        Raises:
            ValueError: 当model为None时

        Note:
            训练过程中，我们在随机时间点t采样轨迹点x_t，
            然后计算神经网络预测的速度场与真实速度场的L2距离。
        """
        if model is None:
            raise ValueError("必须提供model来计算训练损失")

        # 验证输入（创建临时时间张量用于验证）
        temp_t = torch.zeros(x_0.shape[0], device=x_0.device, dtype=x_0.dtype)
        self.validate_inputs(x_0, x_1, temp_t)

        # 确保张量在正确的设备上
        device_result = self.to_device(x_0, x_1)
        if isinstance(device_result, tuple):
            x_0, x_1 = device_result
        else:
            # 这种情况不应该发生，因为我们传入了两个张量
            raise RuntimeError("to_device应该返回两个张量的元组")
        batch_size = x_0.shape[0]

        # 随机采样时间点，支持多个时间步的损失计算
        # TODO: 这里要修改,因为时间 t 的采样不一定是服从高斯分布的,可以是服从其他更加复杂的分布的
        t = torch.rand(batch_size * num_timesteps, device=self.device, dtype=self.dtype)

        # 如果使用多个时间步，需要扩展x_0和x_1
        if num_timesteps > 1:
            x_0_expanded = x_0.repeat(num_timesteps, *([1] * (x_0.dim() - 1)))
            x_1_expanded = x_1.repeat(num_timesteps, *([1] * (x_1.dim() - 1)))
        else:
            x_0_expanded, x_1_expanded = x_0, x_1

        # 采样轨迹点
        x_t = self.sample_trajectory(x_0_expanded, x_1_expanded, t)

        # 计算真实速度场
        true_velocity = self.compute_vector_field(
            x_t, t, x_0=x_0_expanded, x_1=x_1_expanded
        )

        # 获取模型预测的速度场
        # TODO:这里没有考虑 model 可能还会有其他的输入的参数, 需要修改
        # TODO: 在使用的时候最好不使用这种形式,要包装好 model, 比如使用一个类来包装 model,包装其他参数通过其他方式导入
        # TODO: 或者就是简单的,通过传入 Flow.input 这这种形式实现
        predicted_velocity = model(x_t, t)

        # 验证预测输出的形状
        self.validate_vector_field_output(predicted_velocity, x_t.shape)

        # 计算L2损失
        loss = F.mse_loss(predicted_velocity, true_velocity, reduction="mean")

        return loss

    def generate_sample(
        self,
        x_0: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_steps: int = 100,
        method: str = "euler",
        **kwargs: Any,
    ) -> torch.Tensor:
        """使用训练好的模型生成样本.

        通过数值求解ODE dx/dt = u_θ(x, t)来生成样本，
        从简单分布（如高斯噪声）生成复杂数据。

        Args:
            x_0: 初始噪声, shape: (batch_size, *data_shape)
            model: 训练好的速度场预测器
            num_steps: 积分步数，更多步数通常得到更好的质量
            method: 数值积分方法，'euler'或'heun'
            **kwargs: 额外参数

        Returns:
            生成的样本, shape: (batch_size, *data_shape)

        Note:
            这个方法实现了简单的Euler积分器。在实际应用中，
            建议使用更高级的ODE求解器（如torchdiffeq）获得更好的精度。
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

            velocity = model(x, t_tensor)

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
                k2 = model(x_temp, t_next_tensor)
                x = x + 0.5 * dt * (k1 + k2)
            else:
                raise ValueError(f"不支持的积分方法: {method}")

        return x

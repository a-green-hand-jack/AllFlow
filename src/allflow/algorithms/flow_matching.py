"""Flow Matching算法实现

Author: AllFlow Contributors
License: MIT
"""

import logging
from typing import Any, Callable, Optional, Union, Tuple

import torch
import torch.nn.functional as F

from allflow.core.base import FlowMatchingBase
from allflow.core.time_sampling import TimeSamplerBase, UniformTimeSampler
from allflow.core.interpolation import PathInterpolation, EuclideanInterpolation
from allflow.core.vector_field import VectorField, EuclideanVectorField
from allflow.core.noise_generators import NoiseGeneratorBase, GaussianNoiseGenerator

logger = logging.getLogger(__name__)


class FlowMatching(FlowMatchingBase):
    """标准Flow Matching算法实现.

    实现基础的Flow Matching算法，支持多种几何空间的插值和速度场计算。
    新版本支持插值器、速度场计算器和噪声生成器的灵活配置。

    算法核心：
    1. 路径插值：通过插值器计算 x_t = interpolate(x_0, x_1, t)  
    2. 速度场计算：通过速度场计算器计算 u_t(x)
    3. 损失函数：E[||u_θ(x_t, t) - u_t(x)||²]
    4. 智能噪声生成：x_0未提供时自动生成合适的噪声

    新特性（2024-07-29重大更新）：
    - **几何空间支持**: 欧几里得空间、SO(3)旋转群等
    - **灵活插值**: 线性插值、球面插值(SLERP)等  
    - **智能噪声**: 根据几何空间自动选择合适的噪声生成器
    - **解耦设计**: 算法逻辑与具体实现完全分离
    - **向后兼容**: 默认配置与原版API完全兼容

    Args:
        device: 计算设备，默认自动检测
        dtype: 数据类型，默认float32
        sigma_min: 最小噪声水平，用于数值稳定性
        time_sampler: 时间采样器，默认使用均匀分布
        path_interpolation: 路径插值器，默认使用欧几里得线性插值
        vector_field: 速度场计算器，默认使用欧几里得速度场
        noise_generator: 噪声生成器，默认使用高斯噪声

    Example:
        >>> # 标准欧几里得空间使用（向后兼容）
        >>> flow = FlowMatching(device='cuda')
        >>> x_t, t, true_velocity = flow.prepare_training_data(x_0, x_1)
        >>> 
        >>> # SO(3)旋转空间使用
        >>> from allflow.core.interpolation import SO3Interpolation
        >>> from allflow.core.vector_field import SO3VectorField  
        >>> from allflow.core.noise_generators import SO3NoiseGenerator
        >>> 
        >>> flow_so3 = FlowMatching(
        ...     path_interpolation=SO3Interpolation(),
        ...     vector_field=SO3VectorField(),
        ...     noise_generator=SO3NoiseGenerator()
        ... )
        >>> 
        >>> # 自动噪声生成（x_0可选）
        >>> x_t, t, true_velocity = flow.prepare_training_data(x_1=target_quaternions)
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        sigma_min: float = 1e-5,
        time_sampler: Optional[TimeSamplerBase] = None,
        path_interpolation: Optional[PathInterpolation] = None,
        vector_field: Optional[VectorField] = None,
        noise_generator: Optional[NoiseGeneratorBase] = None,
    ) -> None:
        """初始化Flow Matching算法.

        Args:
            device: 计算设备，如'cuda'、'mps'或'cpu'
            dtype: 张量数据类型，默认为torch.float32
            sigma_min: 最小噪声水平，防止数值不稳定
            time_sampler: 时间采样器，默认使用均匀分布采样
            path_interpolation: 路径插值器，默认使用欧几里得线性插值
            vector_field: 速度场计算器，默认使用欧几里得速度场
            noise_generator: 噪声生成器，默认使用高斯噪声
        """
        super().__init__(device=device, dtype=dtype, sigma_min=sigma_min)

        # 设置时间采样器
        if time_sampler is None:
            self.time_sampler = UniformTimeSampler(device=self.device, dtype=self.dtype)
        else:
            self.time_sampler = time_sampler
            self.time_sampler.device = self.device
            self.time_sampler.dtype = self.dtype

        # 设置路径插值器
        if path_interpolation is None:
            self.path_interpolation = EuclideanInterpolation()
        else:
            self.path_interpolation = path_interpolation

        # 设置速度场计算器
        if vector_field is None:
            self.vector_field = EuclideanVectorField()
        else:
            self.vector_field = vector_field

        # 设置噪声生成器
        if noise_generator is None:
            self.noise_generator = GaussianNoiseGenerator()
        else:
            self.noise_generator = noise_generator

        logger.info(
            f"Flow Matching算法初始化完成: "
            f"插值器={type(self.path_interpolation).__name__}, "
            f"速度场={type(self.vector_field).__name__}, "
            f"噪声生成器={type(self.noise_generator).__name__}, "
            f"时间采样器={type(self.time_sampler).__name__}"
        )

    def compute_vector_field(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        x_1: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """计算Flow Matching速度场.

        使用配置的速度场计算器计算在当前位置和时间的速度场值。
        不同的几何空间使用不同的速度场计算方法。

        Args:
            x_t: 当前位置张量, shape: (batch_size, *data_shape)
            t: 时间参数张量, shape: (batch_size,), 取值范围 [0, 1]
            x_0: 源分布采样点, shape: (batch_size, *data_shape)
            x_1: 目标分布采样点, shape: (batch_size, *data_shape)
            **kwargs: 额外参数（保留用于子类扩展）

        Returns:
            速度场张量, shape: (batch_size, *data_shape) 或 (batch_size, tangent_dim)

        Raises:
            ValueError: 当x_0或x_1为None时，或输入形状不匹配时
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
            raise RuntimeError("to_device应该返回多个张量的元组")

        # 设置速度场计算器的端点
        self.vector_field.set_endpoints(x_0, x_1)

        # 计算速度场
        velocity = self.vector_field(x_t, t)

        return velocity

    def sample_trajectory(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """采样插值轨迹.

        使用配置的路径插值器计算从源点到目标点的插值轨迹。
        不同的几何空间使用不同的插值方法。

        Args:
            x_0: 源点, shape: (batch_size, *data_shape)
            x_1: 目标点, shape: (batch_size, *data_shape)
            t: 插值参数, shape: (batch_size,), 范围 [0, 1]
            **kwargs: 额外参数（保留用于子类扩展）

        Returns:
            插值轨迹点, shape: (batch_size, *data_shape)
        """
        # 验证输入
        self.validate_inputs(x_0, x_1, t)

        # 确保所有张量在正确的设备上
        device_result = self.to_device(x_0, x_1, t)
        if isinstance(device_result, tuple):
            x_0, x_1, t = device_result
        else:
            raise RuntimeError("to_device应该返回多个张量的元组")

        # 使用配置的插值器
        x_t = self.path_interpolation.interpolate(x_0, x_1, t)

        return x_t

    def prepare_training_data(
        self,
        x_1: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为训练准备数据：采样时间并计算轨迹点.

        这是一个便利方法，支持智能噪声生成。当x_0未提供时，
        会自动使用配置的噪声生成器生成合适的源分布噪声。

        Args:
            x_1: 目标分布采样, shape: (batch_size, *data_shape) [必需]
            x_0: 源分布采样, shape: (batch_size, *data_shape) [可选]
            batch_size: 批量大小，如果为None则使用x_1的批量大小

        Returns:
            Tuple包含:
            - x_t: 轨迹点, shape: (batch_size, *data_shape)
            - t: 采样的时间点, shape: (batch_size,)
            - true_velocity: 真实速度场, shape: (batch_size, *data_shape或tangent_dim)

        Note:
            新特性：x_0参数现在可选！
            - 如果提供x_0：使用传统Flow Matching
            - 如果不提供x_0：自动使用噪声生成器创建源分布
        """
        if batch_size is None:
            batch_size = x_1.shape[0]

        # 确保x_1在正确的设备上
        x_1_result = self.to_device(x_1)
        if isinstance(x_1_result, tuple):
            x_1 = x_1_result[0]
        else:
            x_1 = x_1_result

        # 智能噪声生成：如果x_0未提供，自动生成
        if x_0 is None:
            logger.debug("x_0未提供，使用噪声生成器自动生成源分布")
            x_0 = self.noise_generator.sample_like(x_1)
            logger.debug(f"生成的x_0形状: {x_0.shape}, 设备: {x_0.device}")
        else:
            # 确保提供的x_0在正确设备上
            x_0_result = self.to_device(x_0)
            if isinstance(x_0_result, tuple):
                x_0 = x_0_result[0]
            else:
                x_0 = x_0_result

        # 使用时间采样器采样时间点
        t = self.time_sampler.sample(batch_size)

        # 采样轨迹点
        x_t = self.sample_trajectory(x_0, x_1, t)

        # 计算真实速度场
        true_velocity = self.compute_vector_field(x_t, t, x_0=x_0, x_1=x_1)

        return x_t, t, true_velocity

    def compute_loss(
        self,
        x_1: torch.Tensor,
        predicted_velocity: torch.Tensor,
        t: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算Flow Matching训练损失.

        计算预测速度场与真实速度场之间的L2损失函数。
        支持智能噪声生成，x_0参数现在可选。

        Args:
            x_1: 目标分布采样, shape: (batch_size, *data_shape) [必需]
            predicted_velocity: 模型预测的速度场, shape: (batch_size, *data_shape或tangent_dim)
            t: 时间参数, shape: (batch_size,), 范围 [0, 1]
            x_0: 源分布采样, shape: (batch_size, *data_shape) [可选]

        Returns:
            标量损失值, shape: ()

        Note:
            这个方法现在专注于纯算法逻辑，支持智能噪声生成。
            模型调用和预测应该在外部完成，这里只计算损失。
        """
        # 智能噪声生成：如果x_0未提供，自动生成
        if x_0 is None:
            x_0 = self.noise_generator.sample_like(x_1)

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
        if predicted_velocity.shape != true_velocity.shape:
            raise ValueError(
                f"预测速度场形状 {predicted_velocity.shape} "
                f"与真实速度场形状 {true_velocity.shape} 不匹配"
            )

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

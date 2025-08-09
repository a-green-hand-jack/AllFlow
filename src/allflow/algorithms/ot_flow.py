"""Optimal Transport Flow (OT-Flow) 算法实现

基于最优传输理论的Flow Matching变体，通过最优传输计划来改进路径插值，
减少传输成本并提高生成质量。

核心思想：
1. 使用最优传输 (Optimal Transport) 计算 x_0 和 x_1 之间的最优配对
2. 基于最优配对重新组织训练数据
3. 学习改进的条件速度场

参考文献：
- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- "Optimal Transport Flow" 相关工作
- Wasserstein距离和最优传输理论

Author: AllFlow Contributors
"""

import logging
from typing import Any, Optional, Tuple, Union

import torch

from ..core.optimal_transport import (
    EuclideanOptimalTransport,
    OptimalTransportBase,
    SO3OptimalTransport,
)
from .flow_matching import FlowMatching

logger = logging.getLogger(__name__)


class OptimalTransportFlow(FlowMatching):
    """Optimal Transport Flow算法实现.

    OT-Flow使用最优传输理论来改进标准Flow Matching：
    1. 计算源分布和目标分布之间的最优传输计划
    2. 基于最优配对重新组织训练数据
    3. 使用改进的路径插值和速度场计算

    相比标准Flow Matching的优势：
    - 减少传输成本（Wasserstein距离）
    - 更稳定的训练过程
    - 更好的生成质量
    - 支持大规模数据集的近似最优传输
    """

    def __init__(
        self,
        space_type: str = "euclidean",
        ot_method: str = "sinkhorn",
        distance_metric: str = "geodesic",
        reg_param: float = 0.1,
        max_iter: int = 1000,
        optimal_transport: Optional[OptimalTransportBase] = None,
        **kwargs: Any,
    ) -> None:
        """初始化OT-Flow算法.

        Args:
            space_type: 空间类型，'euclidean' 或 'so3'
            ot_method: 最优传输求解方法，'exact', 'sinkhorn', 'approx'
            distance_metric: SO3空间的距离度量，'geodesic', 'chordal', 'frobenius'
            reg_param: Sinkhorn正则化参数
            max_iter: Sinkhorn最大迭代次数
            optimal_transport: 自定义最优传输计算器，如果提供则忽略其他OT参数
            **kwargs: 传递给FlowMatching的其他参数
        """
        super().__init__(**kwargs)

        self.space_type = space_type.lower()

        # 创建或使用最优传输计算器
        if optimal_transport is not None:
            self.optimal_transport = optimal_transport
        else:
            if self.space_type == "so3":
                self.optimal_transport = SO3OptimalTransport(
                    method=ot_method,
                    distance_metric=distance_metric,
                    reg_param=reg_param,
                    max_iter=max_iter,
                    device=self.device,
                    dtype=self.dtype,
                )
            elif self.space_type == "euclidean":
                self.optimal_transport = EuclideanOptimalTransport(
                    method=ot_method,
                    reg_param=reg_param,
                    max_iter=max_iter,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                raise ValueError(
                    f"不支持的空间类型: {space_type}，支持: ['euclidean', 'so3']"
                )

        logger.info(
            f"OT-Flow初始化: 空间={self.space_type}, "
            f"方法={self.optimal_transport.method}, "
            f"传输器={type(self.optimal_transport).__name__}"
        )

    def compute_optimal_transport_plan(
        self, x_0: torch.Tensor, x_1: torch.Tensor, return_cost: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """计算最优传输计划.

        计算从x_0到x_1的最优传输计划T，使得传输成本最小。

        Args:
            x_0: 源分布样本，shape: (batch_size, *data_shape)
            x_1: 目标分布样本，shape: (batch_size, *data_shape)
            return_cost: 是否返回传输成本

        Returns:
            transport_plan: 传输计划矩阵，shape: (batch_size, batch_size)
            cost (可选): 传输成本，shape: ()
        """
        return self.optimal_transport.compute_transport_plan(x_0, x_1, return_cost)

    def reorder_by_transport_plan(
        self, x_0: torch.Tensor, x_1: torch.Tensor, transport_plan: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据传输计划重新排序数据.

        这是OT-Flow的核心操作：在采样轨迹点之前，先用最优传输计划重新配对x_0和x_1。

        Args:
            x_0: 源数据，shape: (batch_size, *data_shape)
            x_1: 目标数据，shape: (batch_size, *data_shape)
            transport_plan: 传输计划矩阵，shape: (batch_size, batch_size)

        Returns:
            重新排序的 (x_0_reordered, x_1_reordered)
        """
        return self.optimal_transport.reorder_by_transport_plan(
            x_0, x_1, transport_plan
        )

    def prepare_training_data(
        self,
        x_1: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        use_ot_reordering: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为OT-Flow训练准备数据.

        核心改进：在采样轨迹点之前，先进行最优传输重排序。

        Args:
            x_1: 目标分布样本
            x_0: 源分布样本（可选）
            batch_size: 批量大小
            use_ot_reordering: 是否使用最优传输重排序

        Returns:
            (x_t, t, true_velocity) 元组
        """
        if batch_size is None:
            batch_size = x_1.shape[0]

        # 确保x_1在正确设备上
        x_1_result = self.to_device(x_1)
        if isinstance(x_1_result, tuple):
            x_1 = x_1_result[0]
        else:
            x_1 = x_1_result

        # 生成或处理x_0
        if x_0 is None:
            x_0 = self.noise_generator.sample_like(x_1)
        else:
            x_0_result = self.to_device(x_0)
            if isinstance(x_0_result, tuple):
                x_0 = x_0_result[0]
            else:
                x_0 = x_0_result

        # 🎯 核心OT-Flow操作：最优传输重排序
        if use_ot_reordering and batch_size >= 2:
            # 计算最优传输计划（不返回成本）
            transport_plan = self.compute_optimal_transport_plan(
                x_0, x_1, return_cost=False
            )

            # 确保transport_plan是张量而不是元组
            if isinstance(transport_plan, tuple):
                transport_plan = transport_plan[0]

            # 根据传输计划重新配对x_0和x_1
            x_0, x_1 = self.reorder_by_transport_plan(x_0, x_1, transport_plan)

            logger.debug(f"应用最优传输重排序，批量大小: {batch_size}")

        # 继续标准Flow Matching流程
        # 采样时间点
        t = self.time_sampler.sample(batch_size)

        # 路径插值
        x_t = self.sample_trajectory(x_0, x_1, t)

        # 计算真实速度场
        true_velocity = self.compute_vector_field(x_t, t, x_0=x_0, x_1=x_1)

        return x_t, t, true_velocity

    def compute_ot_loss(
        self,
        x_1: torch.Tensor,
        predicted_velocity: torch.Tensor,
        t: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        ot_weight: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算OT-Flow损失.

        组合标准Flow Matching损失和最优传输正则化项。

        Args:
            x_1: 目标数据
            predicted_velocity: 预测速度场
            t: 时间参数
            x_0: 源数据（可选）
            ot_weight: 最优传输损失权重

        Returns:
            (total_loss, fm_loss, ot_loss) 元组
        """
        # 标准Flow Matching损失
        fm_loss = self.compute_loss(x_1, predicted_velocity, t, x_0)

        # 最优传输正则化
        if x_0 is None:
            x_0 = self.noise_generator.sample_like(x_1)

        # 计算传输成本作为正则化项
        _, ot_cost = self.compute_optimal_transport_plan(x_0, x_1, return_cost=True)

        # 归一化传输成本
        batch_size = x_0.shape[0]
        data_dim = x_0.numel() // batch_size
        ot_loss = ot_cost / (batch_size * data_dim)

        # 总损失
        total_loss = fm_loss + ot_weight * ot_loss

        return total_loss, fm_loss, ot_loss

    def get_algorithm_info(self) -> dict:
        """获取OT-Flow算法信息."""
        ot_info = self.optimal_transport.get_transport_info()
        return {
            "algorithm_type": "optimal_transport_flow",
            "algorithm_name": "OptimalTransportFlow",
            "space_type": self.space_type,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "optimal_transport": ot_info,
        }


# 便捷工厂函数
def create_ot_flow(
    space_type: str = "euclidean",
    ot_method: str = "sinkhorn",
    distance_metric: str = "geodesic",
    reg_param: float = 0.1,
    **kwargs: Any,
) -> OptimalTransportFlow:
    """创建OT-Flow的便捷函数.

    Args:
        space_type: 空间类型，'euclidean' 或 'so3'
        ot_method: 最优传输方法
        distance_metric: SO3空间的距离度量
        reg_param: 正则化参数
        **kwargs: 其他FlowMatching参数

    Returns:
        配置好的OT-Flow实例
    """
    return OptimalTransportFlow(
        space_type=space_type,
        ot_method=ot_method,
        distance_metric=distance_metric,
        reg_param=reg_param,
        **kwargs,
    )

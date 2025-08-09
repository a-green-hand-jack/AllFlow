"""最优传输核心实现

提供不同几何空间上的最优传输计算，支持欧几里得空间和SO3旋转群。

参考文献：
- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- "Optimal Transport on SO(3)" 相关工作
- "SE(3) Flow Matching" 实现

Author: AllFlow Contributors
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 条件导入POT库
try:
    import ot

    HAS_POT = True
    logger.debug("POT库可用，将使用优化的最优传输实现")
except ImportError:
    HAS_POT = False
    logger.debug("POT库不可用，将使用PyTorch近似实现")


class OptimalTransportBase(ABC):
    """最优传输基类.

    定义了最优传输计算的统一接口，支持不同几何空间的实现。
    """

    def __init__(
        self,
        method: str = "sinkhorn",
        reg_param: float = 0.1,
        max_iter: int = 1000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """初始化最优传输计算器.

        Args:
            method: 最优传输方法，'exact', 'sinkhorn', 'approx'
            reg_param: Sinkhorn正则化参数
            max_iter: 最大迭代次数
            device: 计算设备
            dtype: 数据类型
        """
        self.method = method
        self.reg_param = reg_param
        self.max_iter = max_iter
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32

        # 验证方法
        valid_methods = ["exact", "sinkhorn", "approx"]
        if method not in valid_methods:
            raise ValueError(f"不支持的最优传输方法: {method}，支持: {valid_methods}")

        # 检查依赖
        if method in ["exact", "sinkhorn"] and not HAS_POT:
            logger.warning(f"请求使用{method}方法但POT库不可用，将回退到近似方法")
            self.method = "approx"

    @abstractmethod
    def compute_distance_matrix(
        self, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        """计算距离矩阵.

        Args:
            x_0: 源分布样本，shape: (batch_size, *data_shape)
            x_1: 目标分布样本，shape: (batch_size, *data_shape)

        Returns:
            距离矩阵，shape: (batch_size, batch_size)
        """
        pass

    def compute_transport_plan(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        return_cost: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """计算最优传输计划.

        Args:
            x_0: 源分布样本，shape: (batch_size, *data_shape)
            x_1: 目标分布样本，shape: (batch_size, *data_shape)
            return_cost: 是否返回传输成本

        Returns:
            transport_plan: 传输计划矩阵，shape: (batch_size, batch_size)
            cost (可选): 传输成本
        """
        # 计算距离矩阵
        cost_matrix = self.compute_distance_matrix(x_0, x_1)

        # 根据方法选择求解器
        if self.method == "exact" and HAS_POT:
            transport_plan, cost = self._solve_exact_ot(cost_matrix)
        elif self.method == "sinkhorn" and HAS_POT:
            transport_plan, cost = self._solve_sinkhorn_ot(cost_matrix)
        else:
            transport_plan, cost = self._solve_approx_ot(cost_matrix)

        if return_cost:
            return transport_plan, cost
        else:
            return transport_plan

    def _solve_exact_ot(
        self, cost_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用POT库求解精确最优传输."""
        if not HAS_POT:
            raise RuntimeError("POT库不可用，无法计算精确最优传输")

        batch_size = cost_matrix.shape[0]

        # 转换为numpy
        cost_np = cost_matrix.detach().cpu().numpy().astype(np.float64)

        # 均匀分布权重
        a = ot.unif(batch_size)  # type: ignore
        b = ot.unif(batch_size)  # type: ignore

        # 计算最优传输计划
        transport_plan_np = ot.emd(a, b, cost_np)  # type: ignore

        # 转换回PyTorch
        transport_plan = torch.from_numpy(transport_plan_np).to(
            device=self.device, dtype=self.dtype
        )

        # 计算传输成本
        cost = torch.sum(transport_plan * cost_matrix)

        return transport_plan, cost

    def _solve_sinkhorn_ot(
        self, cost_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用POT库求解Sinkhorn近似最优传输."""
        if not HAS_POT:
            raise RuntimeError("POT库不可用，无法计算Sinkhorn最优传输")

        batch_size = cost_matrix.shape[0]

        # 转换为numpy
        cost_np = cost_matrix.detach().cpu().numpy().astype(np.float64)

        # 均匀分布权重
        a = ot.unif(batch_size)  # type: ignore
        b = ot.unif(batch_size)  # type: ignore

        # Sinkhorn算法
        transport_plan_np = ot.sinkhorn(  # type: ignore
            a,
            b,
            cost_np,
            reg=self.reg_param,
            numItermax=self.max_iter,
            stopThr=1e-9,
        )

        # 转换回PyTorch
        transport_plan = torch.from_numpy(transport_plan_np).to(
            device=self.device, dtype=self.dtype
        )

        # 计算传输成本
        cost = torch.sum(transport_plan * cost_matrix)

        return transport_plan, cost

    def _solve_approx_ot(
        self, cost_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用贪心算法求解近似最优传输."""
        batch_size = cost_matrix.shape[0]
        transport_plan = torch.zeros_like(cost_matrix)

        # 贪心最近邻匹配
        used_targets = set()

        for i in range(batch_size):
            # 找到未使用的最小成本目标
            costs = cost_matrix[i].clone()
            for used in used_targets:
                costs[used] = float("inf")

            j = torch.argmin(costs).item()
            transport_plan[i, j] = torch.tensor(
                1.0 / batch_size, dtype=transport_plan.dtype
            )
            used_targets.add(j)

        # 计算传输成本
        cost = torch.sum(transport_plan * cost_matrix)

        return transport_plan, cost

    def reorder_by_transport_plan(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        transport_plan: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据传输计划重新排序数据.

        Args:
            x_0: 源数据
            x_1: 目标数据
            transport_plan: 传输计划矩阵

        Returns:
            重新排序的 (x_0_reordered, x_1_reordered)
        """
        # 找到最优匹配
        _, matched_indices = torch.max(transport_plan, dim=1)

        # 重新排序x_1以匹配x_0
        x_1_reordered = x_1[matched_indices]

        return x_0, x_1_reordered

    def get_transport_info(self) -> dict:
        """获取传输计算器信息."""
        return {
            "transport_type": self.__class__.__name__,
            "method": self.method,
            "reg_param": self.reg_param,
            "max_iter": self.max_iter,
            "has_pot_library": HAS_POT,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }


class EuclideanOptimalTransport(OptimalTransportBase):
    """欧几里得空间的最优传输实现.

    使用L2距离作为成本函数，适用于欧几里得空间中的数据。
    """

    def compute_distance_matrix(
        self, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        """计算欧几里得距离矩阵.

        Args:
            x_0: 源分布样本，shape: (batch_size, data_dim)
            x_1: 目标分布样本，shape: (batch_size, data_dim)

        Returns:
            距离矩阵，shape: (batch_size, batch_size)，元素为L2距离的平方
        """
        batch_size = x_0.shape[0]

        # 展平数据以计算距离
        x_0_flat = x_0.view(batch_size, -1)
        x_1_flat = x_1.view(batch_size, -1)

        # 计算L2距离的平方
        cost_matrix = torch.cdist(x_0_flat, x_1_flat, p=2) ** 2

        return cost_matrix


class SO3OptimalTransport(OptimalTransportBase):
    """SO(3)旋转群的最优传输实现.

    使用四元数表示旋转，计算旋转之间的测地距离作为成本函数。
    适用于SO(3)旋转空间中的数据。
    """

    def __init__(
        self,
        distance_metric: str = "geodesic",
        **kwargs: Any,
    ) -> None:
        """初始化SO3最优传输.

        Args:
            distance_metric: 距离度量，'geodesic', 'chordal', 'frobenius'
            **kwargs: 传递给基类的参数
        """
        super().__init__(**kwargs)

        self.distance_metric = distance_metric
        valid_metrics = ["geodesic", "chordal", "frobenius"]
        if distance_metric not in valid_metrics:
            raise ValueError(
                f"不支持的距离度量: {distance_metric}，支持: {valid_metrics}"
            )

    def compute_distance_matrix(
        self, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        """计算SO(3)空间中的距离矩阵.

        Args:
            x_0: 源四元数，shape: (batch_size, 4)，假设已归一化
            x_1: 目标四元数，shape: (batch_size, 4)，假设已归一化

        Returns:
            距离矩阵，shape: (batch_size, batch_size)
        """
        # 验证输入形状
        if x_0.shape[-1] != 4 or x_1.shape[-1] != 4:
            raise ValueError(
                f"SO3输入必须是四元数，期望最后一维为4，得到: {x_0.shape}, {x_1.shape}"
            )

        batch_size = x_0.shape[0]

        # 展平四元数（如果有额外维度）
        x_0_flat = x_0.view(batch_size, 4)
        x_1_flat = x_1.view(batch_size, 4)

        # 归一化四元数
        x_0_normalized = F.normalize(x_0_flat, dim=1)
        x_1_normalized = F.normalize(x_1_flat, dim=1)

        if self.distance_metric == "geodesic":
            return self._compute_geodesic_distance_matrix(
                x_0_normalized, x_1_normalized
            )
        elif self.distance_metric == "chordal":
            return self._compute_chordal_distance_matrix(x_0_normalized, x_1_normalized)
        elif self.distance_metric == "frobenius":
            return self._compute_frobenius_distance_matrix(
                x_0_normalized, x_1_normalized
            )
        else:
            raise ValueError(f"未实现的距离度量: {self.distance_metric}")

    def _compute_geodesic_distance_matrix(
        self, q_0: torch.Tensor, q_1: torch.Tensor
    ) -> torch.Tensor:
        """计算四元数之间的测地距离矩阵.

        测地距离是SO(3)流形上的自然距离，定义为：
        d(q_i, q_j) = arccos(|⟨q_i, q_j⟩|)

        Args:
            q_0: 归一化四元数，shape: (batch_size, 4)
            q_1: 归一化四元数，shape: (batch_size, 4)

        Returns:
            测地距离矩阵，shape: (batch_size, batch_size)
        """
        # 计算内积矩阵
        dot_products = torch.mm(q_0, q_1.t())  # (batch_size, batch_size)

        # 取绝对值处理四元数的双重覆盖性质 (q 和 -q 表示同一旋转)
        abs_dot_products = torch.abs(dot_products)

        # 限制范围避免数值错误
        abs_dot_products = torch.clamp(abs_dot_products, 0.0, 1.0)

        # 计算测地距离
        geodesic_distances = torch.acos(abs_dot_products)

        return geodesic_distances**2  # 返回距离的平方保持与欧几里得实现一致

    def _compute_chordal_distance_matrix(
        self, q_0: torch.Tensor, q_1: torch.Tensor
    ) -> torch.Tensor:
        """计算四元数之间的弦距离矩阵.

        弦距离定义为：
        d(q_i, q_j) = min(||q_i - q_j||, ||q_i + q_j||)

        Args:
            q_0: 归一化四元数，shape: (batch_size, 4)
            q_1: 归一化四元数，shape: (batch_size, 4)

        Returns:
            弦距离矩阵，shape: (batch_size, batch_size)
        """
        batch_size = q_0.shape[0]

        # 扩展维度进行广播
        q_0_expanded = q_0.unsqueeze(1)  # (batch_size, 1, 4)
        q_1_expanded = q_1.unsqueeze(0)  # (1, batch_size, 4)

        # 计算两种可能的距离
        dist_pos = torch.norm(
            q_0_expanded - q_1_expanded, dim=2
        )  # (batch_size, batch_size)
        dist_neg = torch.norm(
            q_0_expanded + q_1_expanded, dim=2
        )  # (batch_size, batch_size)

        # 取最小距离
        chordal_distances = torch.min(dist_pos, dist_neg)

        return chordal_distances**2

    def _compute_frobenius_distance_matrix(
        self, q_0: torch.Tensor, q_1: torch.Tensor
    ) -> torch.Tensor:
        """计算旋转矩阵的Frobenius距离矩阵.

        将四元数转换为旋转矩阵，然后计算Frobenius范数。

        Args:
            q_0: 归一化四元数，shape: (batch_size, 4)
            q_1: 归一化四元数，shape: (batch_size, 4)

        Returns:
            Frobenius距离矩阵，shape: (batch_size, batch_size)
        """
        # 将四元数转换为旋转矩阵
        R_0 = self._quaternion_to_rotation_matrix(q_0)  # (batch_size, 3, 3)
        R_1 = self._quaternion_to_rotation_matrix(q_1)  # (batch_size, 3, 3)

        batch_size = R_0.shape[0]

        # 计算所有配对的距离矩阵
        distances = torch.zeros(
            batch_size, batch_size, device=R_0.device, dtype=R_0.dtype
        )

        for i in range(batch_size):
            for j in range(batch_size):
                # 计算旋转矩阵差的Frobenius范数
                diff = R_0[i] - R_1[j]  # (3, 3)
                distances[i, j] = torch.norm(diff, p="fro")

        return distances**2

    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """将四元数转换为旋转矩阵.

        Args:
            quaternions: 四元数，shape: (batch_size, 4)，格式为 [w, x, y, z]

        Returns:
            旋转矩阵，shape: (batch_size, 3, 3)
        """
        # 假设四元数格式为 [w, x, y, z]
        w, x, y, z = (
            quaternions[:, 0],
            quaternions[:, 1],
            quaternions[:, 2],
            quaternions[:, 3],
        )

        # 计算旋转矩阵元素
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        # 构建旋转矩阵
        batch_size = quaternions.shape[0]
        R = torch.zeros(
            batch_size, 3, 3, device=quaternions.device, dtype=quaternions.dtype
        )

        R[:, 0, 0] = 1 - 2 * (yy + zz)
        R[:, 0, 1] = 2 * (xy - wz)
        R[:, 0, 2] = 2 * (xz + wy)
        R[:, 1, 0] = 2 * (xy + wz)
        R[:, 1, 1] = 1 - 2 * (xx + zz)
        R[:, 1, 2] = 2 * (yz - wx)
        R[:, 2, 0] = 2 * (xz - wy)
        R[:, 2, 1] = 2 * (yz + wx)
        R[:, 2, 2] = 1 - 2 * (xx + yy)

        return R

    def get_transport_info(self) -> dict:
        """获取SO3传输计算器信息."""
        base_info = super().get_transport_info()
        base_info.update(
            {
                "space_type": "SO3",
                "distance_metric": self.distance_metric,
            }
        )
        return base_info


# 工厂函数
def create_optimal_transport(
    space_type: str = "euclidean",
    method: str = "sinkhorn",
    **kwargs: Any,
) -> OptimalTransportBase:
    """创建最优传输计算器的工厂函数.

    Args:
        space_type: 空间类型，'euclidean' 或 'so3'
        method: 最优传输方法
        **kwargs: 其他参数

    Returns:
        最优传输计算器实例
    """
    if space_type.lower() == "euclidean":
        return EuclideanOptimalTransport(method=method, **kwargs)
    elif space_type.lower() == "so3":
        return SO3OptimalTransport(method=method, **kwargs)
    else:
        raise ValueError(f"不支持的空间类型: {space_type}，支持: ['euclidean', 'so3']")

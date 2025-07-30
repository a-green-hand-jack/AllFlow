"""Optimal Transport Flow (OT-Flow) 算法实现

基于最优传输理论的Flow Matching变体，通过最优传输计划来改进路径插值，
减少传输成本并提高生成质量。

核心思想：
1. 使用最优传输 (Optimal Transport) 计算 x_0 和 x_1 之间的最优配对
2. 基于最优配对计算更优的插值路径
3. 学习改进的条件速度场

参考文献：
- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- "Optimal Transport Flow" 相关工作
- Wasserstein距离和最优传输理论

Author: AllFlow Contributors
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Any, Union
import logging

from .flow_matching import FlowMatching

logger = logging.getLogger(__name__)

# 条件导入POT库（Python Optimal Transport）
try:
    import ot
    HAS_POT = True
    logger.info("POT库可用，OT-Flow将使用优化的最优传输实现")
except ImportError:
    HAS_POT = False
    logger.warning(
        "POT库不可用，OT-Flow将使用简化的最优传输实现。"
        "建议安装POT库：pip install pot"
    )


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
        ot_method: str = "sinkhorn",
        reg_param: float = 0.1,
        max_iter: int = 1000,
        approx_threshold: int = 10000,
        **kwargs: Any
    ) -> None:
        """初始化OT-Flow算法.
        
        Args:
            ot_method: 最优传输求解方法，'exact', 'sinkhorn', 'approx'
            reg_param: Sinkhorn正则化参数
            max_iter: Sinkhorn最大迭代次数
            approx_threshold: 近似方法的阈值，当batch_size超过此值时使用近似方法
            **kwargs: 传递给FlowMatching的其他参数
        """
        super().__init__(**kwargs)
        
        self.ot_method = ot_method
        self.reg_param = reg_param
        self.max_iter = max_iter
        self.approx_threshold = approx_threshold
        
        # 验证最优传输方法
        valid_methods = ['exact', 'sinkhorn', 'approx']
        if ot_method not in valid_methods:
            raise ValueError(f"不支持的最优传输方法: {ot_method}，支持: {valid_methods}")
        
        # 检查依赖
        if ot_method in ['exact', 'sinkhorn'] and not HAS_POT:
            logger.warning(
                f"请求使用{ot_method}方法但POT库不可用，"
                f"将回退到近似方法"
            )
            self.ot_method = 'approx'
        
        logger.info(
            f"OT-Flow初始化: 方法={self.ot_method}, "
            f"正则化={self.reg_param}, 最大迭代={self.max_iter}"
        )
    
    def compute_optimal_transport_plan(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        return_cost: bool = False
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
        batch_size = x_0.shape[0]
        
        # 展平数据以计算距离
        x_0_flat = x_0.view(batch_size, -1)
        x_1_flat = x_1.view(batch_size, -1)
        
        if self.ot_method == 'exact' and HAS_POT:
            # 精确最优传输（适用于小批量）
            transport_plan, cost = self._compute_exact_ot(x_0_flat, x_1_flat)
            
        elif self.ot_method == 'sinkhorn' and HAS_POT:
            # Sinkhorn近似（适用于中等批量）
            transport_plan, cost = self._compute_sinkhorn_ot(x_0_flat, x_1_flat)
            
        else:
            # 近似方法（适用于大批量）
            transport_plan, cost = self._compute_approx_ot(x_0_flat, x_1_flat)
        
        if return_cost:
            return transport_plan, cost
        else:
            return transport_plan


# 便捷工厂函数
def create_ot_flow(
    ot_method: str = "sinkhorn",
    reg_param: float = 0.1,
    **kwargs: Any
) -> OptimalTransportFlow:
    """创建OT-Flow的便捷函数.
    
    Args:
        ot_method: 最优传输方法
        reg_param: 正则化参数
        **kwargs: 其他FlowMatching参数
        
    Returns:
        配置好的OT-Flow实例
    """
    return OptimalTransportFlow(
        ot_method=ot_method,
        reg_param=reg_param,
        **kwargs
    )

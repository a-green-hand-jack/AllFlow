"""AllFlow算法实现模块

本模块包含所有Flow Matching算法变体的具体实现，包括基础Flow Matching、
Mean Flow、Conditional Flow Matching、RectifiedFlow和Optimal Transport Flow。

支持的算法：
- FlowMatching: 基础Flow Matching算法实现
- MeanFlow: 期望流的高效计算方法  
- ConditionalFlowMatching: 支持条件生成的Flow Matching
- RectifiedFlow: 流线矫正和迭代优化算法
- OptimalTransportFlow: 最优传输引导的流方法

性能特点：
- 所有算法避免Python循环，使用纯张量操作
- 支持批量处理和GPU加速
- 数值稳定的边界条件处理
- 统一的API接口设计

Author: AllFlow Contributors
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 类型检查时的导入，避免运行时循环导入
    pass

# 算法实现类 - 将在实现后导入
# from .flow_matching import FlowMatching
# from .mean_flow import MeanFlow
# from .cfm import ConditionalFlowMatching
# from .rectified_flow import RectifiedFlow
# from .ot_flow import OptimalTransportFlow

# 导出的公共API
__all__ = [
    # Flow Matching算法实现 (将在实现后取消注释)
    # "FlowMatching",
    # "MeanFlow", 
    # "ConditionalFlowMatching",
    # "RectifiedFlow",
    # "OptimalTransportFlow",
] 
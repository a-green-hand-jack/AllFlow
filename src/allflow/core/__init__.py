"""AllFlow核心抽象模块

本模块包含AllFlow库的核心抽象类和接口定义，为所有Flow Matching算法
提供统一的基类和通用组件。

主要组件：
- FlowMatchingBase: 所有Flow Matching算法的抽象基类
- TrajectoryPath: 轨迹路径的抽象表示
- TimeScheduler: 时间调度和噪声管理
- VectorField: 速度场的抽象表示

设计原则：
- 提供统一的接口，确保所有算法实现的一致性
- 支持批量处理和GPU加速
- 强制类型注解，确保类型安全
- 模块化设计，便于扩展和测试

Author: AllFlow Contributors
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 避免循环导入，仅在类型检查时导入
    pass

# 核心抽象类 - 将在实现后导入
# from .base import FlowMatchingBase, VectorField
# from .trajectory import TrajectoryPath, InterpolationScheme  
# from .scheduler import TimeScheduler, NoiseScheduler

# 导出的公共API
__all__ = [
    # 核心基类 (将在实现后取消注释)
    # "FlowMatchingBase",
    # "VectorField",
    # "TrajectoryPath", 
    # "InterpolationScheme",
    # "TimeScheduler",
    # "NoiseScheduler",
] 
"""AllFlow ODE求解器模块

本模块包含用于Flow Matching采样的各种ODE积分器实现，支持不同精度
和性能需求的求解方案。

支持的求解器：
- EulerSolver: 一阶Euler方法，速度最快
- HeunSolver: 二阶Heun方法，精度与速度平衡
- AdaptiveSolver: 自适应步长求解器，精度最高

设计特点：
- 统一的求解器接口
- 高效的批量处理支持  
- GPU优化的数值积分
- 自动步长控制和错误估计

Author: AllFlow Contributors
"""

# 求解器实现类 - 将在实现后导入
# from .euler import EulerSolver
# from .heun import HeunSolver
# from .adaptive import AdaptiveSolver

# 导出的公共API
__all__ = [
    # ODE求解器 (将在实现后取消注释)
    # "EulerSolver",
    # "HeunSolver", 
    # "AdaptiveSolver",
] 
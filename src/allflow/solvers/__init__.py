"""AllFlow ODE求解器模块

本模块包含用于Flow Matching采样的各种ODE积分器实现，支持不同精度
和性能需求的求解方案。

支持的求解器：
- ODESolverBase: 抽象基类，定义统一接口
- TorchDiffEqSolver: torchdiffeq库的高性能包装器
- EulerSolver: 简单Euler方法的后备实现
- SolverConfig: 求解器配置参数
- VectorFieldWrapper: 速度场函数包装器

设计特点：
- 统一的求解器接口
- 高效的批量处理支持  
- GPU优化的数值积分
- 自动步长控制和错误估计

Author: AllFlow Contributors
"""

# 基础接口导入
from .base import ODESolverBase, SolverConfig, VectorFieldWrapper

# 条件导入torchdiffeq求解器
try:
    from .torchdiffeq_solver import TorchDiffEqSolver, EulerSolver
    HAS_TORCHDIFFEQ = True
except ImportError:
    TorchDiffEqSolver = None
    EulerSolver = None
    HAS_TORCHDIFFEQ = False

# 导出的公共API
__all__ = [
    # 基础接口
    "ODESolverBase",
    "SolverConfig", 
    "VectorFieldWrapper",
]

# 条件导出（如果依赖可用）
if HAS_TORCHDIFFEQ:
    __all__.extend([
        "TorchDiffEqSolver",
        "EulerSolver",
    ])

# 便捷函数
def create_solver(solver_type: str = "torchdiffeq", **kwargs) -> ODESolverBase:
    """创建ODE求解器的便捷函数.
    
    Args:
        solver_type: 求解器类型，'torchdiffeq'或'euler'
        **kwargs: 传递给求解器的参数
        
    Returns:
        配置好的求解器实例
        
    Raises:
        ImportError: 当请求的求解器不可用时
        ValueError: 当solver_type不支持时
    """
    if solver_type == "torchdiffeq":
        if not HAS_TORCHDIFFEQ:
            raise ImportError(
                "torchdiffeq不可用。请安装: pip install torchdiffeq"
            )
        return TorchDiffEqSolver(**kwargs)  # type: ignore
    
    elif solver_type == "euler":
        if not HAS_TORCHDIFFEQ:
            raise ImportError(
                "Euler求解器需要torchdiffeq。请安装: pip install torchdiffeq"
            )
        return EulerSolver(**kwargs)  # type: ignore
    
    else:
        raise ValueError(f"不支持的求解器类型: {solver_type}")

# 添加便捷函数到导出列表
__all__.append("create_solver") 
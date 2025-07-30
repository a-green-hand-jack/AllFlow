"""AllFlow - 高效的Flow Matching算法库

AllFlow是一个专注于Flow Matching核心算法的PyTorch库，提供：

- 纯算法实现，与神经网络架构解耦
- 极致性能优化，避免Python循环
- 跨设备兼容性（CPU/CUDA/MPS）
- 科学严谨的数学实现
- 灵活的时间采样策略
- 统一的模型接口系统

主要组件:
- FlowMatching: 标准Flow Matching算法
- TimeSampler: 灵活的时间采样器（均匀、正态、指数、重要性）
- ModelInterface: 统一的模型接口（支持条件模型和额外参数）
- ODESolver: 高精度ODE求解器
- 工具函数: 验证、设备管理等

快速开始:
    >>> import allflow
    >>> # 基础用法
    >>> flow = allflow.FlowMatching()
    >>> 
    >>> # 使用自定义时间采样器
    >>> sampler = allflow.NormalTimeSampler(mean=0.3, std=0.1)
    >>> flow = allflow.FlowMatching(time_sampler=sampler)
    >>> 
    >>> # 使用条件模型
    >>> model_wrapper = allflow.ConditionalModelWrapper(model, condition=labels)
    >>> loss = flow.compute_loss(x_0, x_1, model=model_wrapper)

Author: AllFlow Contributors
License: MIT
Version: 0.1.0
"""

import logging
from typing import Optional

# 核心算法导入
from .algorithms.flow_matching import FlowMatching
from .core.base import FlowMatchingBase, VectorField, PathInterpolation

# 时间采样器导入
from .core.time_sampling import (
    TimeSamplerBase,
    UniformTimeSampler,
    NormalTimeSampler,
    ExponentialTimeSampler,
    ImportanceTimeSampler,
    create_time_sampler,
)

# 模型接口导入
from .core.model_interface import (
    ModelInterface,
    SimpleModelWrapper,
    ConditionalModelWrapper,
    FlexibleModelWrapper,
    FunctionModelWrapper,
    create_model_wrapper,
)

# ODE求解器导入
from .solvers.base import ODESolverBase, SolverConfig, VectorFieldWrapper

# 尝试导入torchdiffeq求解器（可选依赖）
try:
    from .solvers.torchdiffeq_solver import TorchDiffEqSolver, EulerSolver
    HAS_TORCHDIFFEQ = True
except ImportError:
    TorchDiffEqSolver = None
    EulerSolver = None
    HAS_TORCHDIFFEQ = False

# 导入插值器相关类
from .core.interpolation import (
    EuclideanInterpolation,
    SO3Interpolation,
    create_interpolation,
)

# 导入速度场相关类
from .core.vector_field import (
    EuclideanVectorField,
    SO3VectorField,
    create_vector_field,
)

# 导入噪声生成器相关类
from .core.noise_generators import (
    NoiseGeneratorBase,
    GaussianNoiseGenerator,
    SO3NoiseGenerator,
    UniformNoiseGenerator,
    create_noise_generator,
)

# 版本信息
__version__ = "0.1.0"
__author__ = "AllFlow Contributors"
__email__ = "allflow@example.com"
__license__ = "MIT"

# 公共API导出
__all__ = [
    # 版本信息
    "__version__", "__author__", "__license__",
    
    # 核心算法
    "FlowMatching",
    "FlowMatchingBase",
    "VectorField", 
    "PathInterpolation",
    
    # 时间采样器
    "TimeSamplerBase",
    "UniformTimeSampler",
    "NormalTimeSampler",
    "ExponentialTimeSampler",
    "ImportanceTimeSampler",
    "create_time_sampler",
    
    # 模型接口
    "ModelInterface",
    "SimpleModelWrapper",
    "ConditionalModelWrapper",
    "FlexibleModelWrapper",
    "FunctionModelWrapper",
    "create_model_wrapper",
    
    # 路径插值 (新增)
    "EuclideanInterpolation",
    "SO3Interpolation",
    "create_interpolation",
    # 速度场计算 (新增)
    "EuclideanVectorField",
    "SO3VectorField",
    "create_vector_field",
    # 噪声生成器 (新增)
    "NoiseGeneratorBase",
    "GaussianNoiseGenerator",
    "SO3NoiseGenerator",
    "UniformNoiseGenerator",
    "create_noise_generator",
    # ODE求解器
    "ODESolverBase",
    "SolverConfig",
    "VectorFieldWrapper",
    
    # 工具函数
    "get_device_info",
    "set_global_seed",
    "validate_tensor_inputs",
    
    # 便捷函数
    "create_flow_matching",
    "create_solver",
]

# 条件导出（如果依赖可用）
if HAS_TORCHDIFFEQ:
    __all__.extend([
        "TorchDiffEqSolver",
        "EulerSolver",
    ])

# 设置日志
logger = logging.getLogger(__name__)


def get_device_info() -> dict:
    """获取当前设备信息.
    
    Returns:
        包含设备信息的字典
    """
    import torch
    
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    
    if torch.cuda.is_available():
        try:
            cuda_version = torch.version.cuda  # type: ignore
        except AttributeError:
            cuda_version = 'unknown'
        info.update({
            "cuda_version": cuda_version,
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
        })
    
    return info


def set_global_seed(seed: int) -> None:
    """设置全局随机种子.
    
    Args:
        seed: 随机种子值
    """
    import torch
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 确保确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"全局随机种子设置为: {seed}")


def validate_tensor_inputs(*tensors, allow_empty: bool = False) -> bool:
    """验证张量输入的有效性.
    
    Args:
        *tensors: 要验证的张量
        allow_empty: 是否允许空张量
        
    Returns:
        验证是否通过
        
    Raises:
        ValueError: 当张量无效时
    """
    import torch
    
    if not allow_empty and len(tensors) == 0:
        raise ValueError("至少需要一个张量进行验证")
    
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"第{i}个输入不是torch.Tensor类型: {type(tensor)}")
        
        if torch.isnan(tensor).any():
            raise ValueError(f"第{i}个张量包含NaN值")
        
        if torch.isinf(tensor).any():
            raise ValueError(f"第{i}个张量包含Inf值")
    
    return True


def create_flow_matching(
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    **kwargs
) -> FlowMatching:
    """创建Flow Matching实例的便捷函数.
    
    Args:
        device: 计算设备，如'cuda'、'mps'或'cpu'
        dtype: 数据类型，如'float32'
        **kwargs: 传递给FlowMatching的其他参数
        
    Returns:
        配置好的FlowMatching实例
        
    Example:
        >>> flow = allflow.create_flow_matching(device='cuda')
        >>> # 开始使用flow进行训练
    """
    import torch
    
    # 转换dtype字符串为torch类型
    torch_dtype = None
    if dtype is not None:
        if isinstance(dtype, str):
            torch_dtype = getattr(torch, dtype)
        else:
            torch_dtype = dtype
    
    return FlowMatching(device=device, dtype=torch_dtype, **kwargs)


def create_solver(
    solver_type: str = "torchdiffeq",
    method: str = "dopri5",
    **kwargs
) -> ODESolverBase:
    """创建ODE求解器的便捷函数.
    
    Args:
        solver_type: 求解器类型，'torchdiffeq'或'euler'
        method: 数值方法名称
        **kwargs: 传递给求解器的其他参数
        
    Returns:
        配置好的ODE求解器实例
        
    Raises:
        ImportError: 当请求的求解器不可用时
        ValueError: 当solver_type不支持时
    """
    if solver_type == "torchdiffeq":
        if not HAS_TORCHDIFFEQ:
            raise ImportError(
                "torchdiffeq不可用。请安装: pip install torchdiffeq"
            )
        return TorchDiffEqSolver(method=method, **kwargs)  # type: ignore
    
    elif solver_type == "euler":
        if not HAS_TORCHDIFFEQ:
            raise ImportError(
                "Euler求解器需要torchdiffeq。请安装: pip install torchdiffeq"
            )
        return EulerSolver(**kwargs)  # type: ignore
    
    else:
        raise ValueError(f"不支持的求解器类型: {solver_type}")


def _check_dependencies() -> None:
    """检查必要的依赖并推荐最优计算后端."""
    try:
        import torch
        
        # 检查设备可用性并给出建议
        available_devices = []
        performance_info = []
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            available_devices.append(f"CUDA (GPU: {gpu_count}个)")
            performance_info.append("✅ CUDA GPU可用，将获得最佳性能")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_devices.append("MPS (Apple Silicon)")
            performance_info.append("✅ Apple Silicon MPS可用，针对M系列芯片优化")
        
        available_devices.append("CPU")
        
        if len(available_devices) == 1:  # 只有CPU
            import warnings
            warnings.warn(
                "⚠️  只有CPU后端可用，性能可能受限。\n"
                "建议：\n"
                "• Linux: 安装CUDA版本的PyTorch\n"
                "• Mac: 确保使用支持MPS的PyTorch 2.0+",
                UserWarning
            )
        else:
            # 有GPU或MPS加速
            import logging
            logging.info(f"🚀 AllFlow已检测到加速计算后端: {', '.join(available_devices)}")
            for info in performance_info:
                logging.info(info)
                
    except ImportError:
        raise ImportError(
            "AllFlow需要PyTorch>=2.0.0。请根据您的平台安装：\n\n"
            "📱 Apple Silicon Mac:\n"
            "   pip install torch>=2.0.0\n\n"
            "🐧 Linux with CUDA:\n"
            "   pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118\n\n"
            "💻 CPU only:\n"
            "   pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu"
        )


# 在导入时检查依赖
_check_dependencies() 
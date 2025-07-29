"""AllFlow: 高效的Flow Matching算法库

AllFlow是一个专注于Flow Matching核心算法的PyTorch库，提供高效、可扩展的
Flow Matching变体实现。本模块导出了库的公共API，包括所有算法类和工具函数。

主要特色：
- 零Python循环，纯张量操作实现
- 统一的算法接口设计
- 完整的Flow Matching变体覆盖
- 高性能GPU优化

Example:
    基本使用方式：
    
    >>> import torch
    >>> from allflow import FlowMatching
    >>> 
    >>> flow = FlowMatching(device='cuda')
    >>> x_0 = torch.randn(32, 64, device='cuda')
    >>> x_1 = torch.randn(32, 64, device='cuda')
    >>> t = torch.rand(32, device='cuda')
    >>> 
    >>> velocity = flow.compute_vector_field(x_0, x_1, t)
    >>> loss = flow.compute_loss(x_0, x_1)

Author: AllFlow Contributors
License: MIT
"""

from typing import Any

# 版本信息
__version__ = "0.1.0"
__author__ = "AllFlow Contributors"

# 核心算法类 - 将在实现后导入
# from .algorithms import (
#     FlowMatching,
#     MeanFlow,
#     ConditionalFlowMatching,
#     RectifiedFlow,
#     OptimalTransportFlow,
# )

# ODE求解器 - 将在实现后导入
# from .solvers import (
#     EulerSolver,
#     HeunSolver,
#     AdaptiveSolver,
# )

# 工具函数 - 将在实现后导入
# from .utils import (
#     validate_tensor_shapes,
#     compute_flow_straightness,
# )

# 公共API列表 - 定义哪些符号可以被外部导入
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # 核心算法类 (将在实现后取消注释)
    # "FlowMatching",
    # "MeanFlow", 
    # "ConditionalFlowMatching",
    # "RectifiedFlow",
    # "OptimalTransportFlow",
    
    # ODE求解器 (将在实现后取消注释)
    # "EulerSolver",
    # "HeunSolver",
    # "AdaptiveSolver",
    
    # 工具函数 (将在实现后取消注释)
    # "validate_tensor_shapes",
    # "compute_flow_straightness",
]


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
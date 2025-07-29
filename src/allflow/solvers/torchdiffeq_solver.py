"""TorchDiffEq求解器包装器实现

本模块实现了torchdiffeq库的包装器，为Flow Matching提供高质量的ODE求解功能。
torchdiffeq是一个专业的神经常微分方程求解库，支持多种高精度数值方法。

支持的求解方法：
- euler: 一阶Euler方法，速度最快
- heun3: 三阶Heun方法，精度与速度平衡  
- rk4: 四阶Runge-Kutta方法，经典高精度
- dopri5: Dormand-Prince方法，自适应步长
- dopri8: 八阶Dormand-Prince方法，极高精度
- adaptive_heun: 自适应Heun方法

优势特点：
- 工业级数值稳定性和精度
- GPU优化的高性能实现
- 自适应步长控制
- 丰富的求解方法选择

Author: AllFlow Contributors

Reference:
    Chen, R. T. Q., et al. (2018). Neural ordinary differential equations.
    https://github.com/rtqichen/torchdiffeq
"""

from typing import Callable, Optional, Union, Any, List
import logging
import torch
from .base import ODESolverBase, SolverConfig, VectorFieldWrapper

# 条件导入torchdiffeq
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    odeint = None

logger = logging.getLogger(__name__)


class TorchDiffEqSolver(ODESolverBase):
    """TorchDiffEq库的包装器求解器.
    
    提供对torchdiffeq库的统一接口封装，支持多种高精度ODE求解方法。
    特别适用于需要高精度和稳定性的Flow Matching采样任务。
    
    支持的方法：
    - 'euler': 一阶显式Euler方法
    - 'heun3': 三阶Heun方法
    - 'rk4': 经典四阶Runge-Kutta方法
    - 'dopri5': 五阶Dormand-Prince自适应方法（推荐）
    - 'dopri8': 八阶Dormand-Prince方法（高精度）
    - 'adaptive_heun': 自适应Heun方法
    
    Example:
        >>> solver = TorchDiffEqSolver(method='dopri5', rtol=1e-5)
        >>> x0 = torch.randn(32, 128)
        >>> t_span = torch.linspace(0, 1, 101)
        >>> trajectory = solver.solve(vector_field, x0, t_span)
    """
    
    def __init__(
        self,
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-6,
        config: Optional[SolverConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        **solver_kwargs: Any
    ):
        """初始化TorchDiffEq求解器.
        
        Args:
            method: ODE求解方法名称
            rtol: 相对误差容限
            atol: 绝对误差容限
            config: 求解器配置（可选）
            device: 计算设备
            dtype: 数据类型
            **solver_kwargs: 传递给torchdiffeq的额外参数
            
        Raises:
            ImportError: 当torchdiffeq未安装时
            ValueError: 当method不受支持时
        """
        # 检查torchdiffeq可用性
        if not HAS_TORCHDIFFEQ:
            raise ImportError(
                "torchdiffeq未安装。请安装：pip install torchdiffeq\n"
                "或安装AllFlow时包含ODE求解器支持：pip install allflow[ode]"
            )
        
        # 验证求解方法
        self._validate_method(method)
        
        # 创建配置
        if config is None:
            config = SolverConfig(
                method=method,
                rtol=rtol, 
                atol=atol,
                adaptive=method in ['dopri5', 'dopri8', 'adaptive_heun']
            )
        
        super().__init__(config=config, device=device, dtype=dtype)
        
        # 存储torchdiffeq特定参数
        self.solver_kwargs = solver_kwargs
        
        logger.info(f"TorchDiffEq求解器初始化: method={method}, rtol={rtol}, atol={atol}")
    
    @staticmethod
    def _validate_method(method: str) -> None:
        """验证求解方法是否受支持."""
        supported_methods = {
            'euler', 'heun3', 'rk4', 'dopri5', 'dopri8', 
            'adaptive_heun', 'midpoint', 'implicit_adams'
        }
        
        if method not in supported_methods:
            raise ValueError(
                f"不支持的求解方法: {method}。"
                f"支持的方法: {sorted(supported_methods)}"
            )
    
    def solve(
        self,
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        t_span: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """使用torchdiffeq求解ODE系统.
        
        Args:
            vector_field: 速度场函数 f(x, t)，输入(x, t)，输出速度
            x0: 初始条件，shape: (batch_size, *state_shape)
            t_span: 时间点序列，shape: (num_times,)，必须单调
            **kwargs: 额外的求解器参数，会覆盖默认配置
            
        Returns:
            轨迹解，shape: (num_times, batch_size, *state_shape)
            
        Raises:
            ValueError: 当输入参数不合法时
            RuntimeError: 当求解过程失败时
            
        Note:
            torchdiffeq要求速度场函数的签名为f(t, x)，与我们的f(x, t)不同，
            因此需要使用VectorFieldWrapper进行转换。
        """
        # 验证输入
        self.validate_inputs(x0, t_span)
        
        # 确保张量在正确设备上
        device_result = self.to_device(x0, t_span)
        if isinstance(device_result, tuple):
            x0, t_span = device_result
        else:
            raise RuntimeError("to_device应该返回两个张量的元组")
        
        # 包装速度场函数以符合torchdiffeq的接口要求
        wrapped_field = VectorFieldWrapper(vector_field, reverse_time=False)
        
        # 合并求解器参数
        solver_options = {
            'method': self.config.method,
            'rtol': self.config.rtol,
            'atol': self.config.atol,
            **self.solver_kwargs,
            **kwargs  # 用户提供的参数具有最高优先级
        }
        
        try:
            # 调用torchdiffeq求解ODE
            assert odeint is not None, "odeint不应为None"
            trajectory: torch.Tensor = odeint(
                wrapped_field,
                x0,
                t_span,
                **solver_options
            )
            
            # 验证输出
            self._validate_solution(trajectory, x0, t_span)
            
            return trajectory
            
        except Exception as e:
            logger.error(f"TorchDiffEq求解失败: {e}")
            raise RuntimeError(f"ODE求解失败: {e}") from e
    
    def _validate_solution(
        self, 
        trajectory: torch.Tensor, 
        x0: torch.Tensor, 
        t_span: torch.Tensor
    ) -> None:
        """验证求解结果的有效性."""
        expected_shape = (len(t_span),) + x0.shape
        
        if trajectory.shape != expected_shape:
            raise RuntimeError(
                f"求解结果形状错误，期望: {expected_shape}, 得到: {trajectory.shape}"
            )
        
        if torch.isnan(trajectory).any():
            raise RuntimeError("求解结果包含NaN值")
        
        if torch.isinf(trajectory).any():
            raise RuntimeError("求解结果包含Inf值")
        
        # 检查初始条件是否保持
        initial_error = torch.norm(trajectory[0] - x0, dim=tuple(range(1, x0.dim())))
        if torch.any(initial_error > 1e-6):
            logger.warning(f"初始条件误差较大，最大误差: {initial_error.max():.2e}")
    
    def get_adaptive_info(self) -> dict[str, Any]:
        """获取自适应求解器的性能信息.
        
        Returns:
            包含求解器性能统计的字典
        """
        info = {
            'method': self.config.method,
            'adaptive': self.config.adaptive,
            'rtol': self.config.rtol,
            'atol': self.config.atol,
        }
        
        # 可以在未来添加更多性能统计信息
        
        return info
    
    @classmethod
    def recommended_config(cls, precision: str = "medium") -> "TorchDiffEqSolver":
        """创建推荐配置的求解器实例.
        
        Args:
            precision: 精度级别，'low'/'medium'/'high'
            
        Returns:
            配置好的求解器实例
        """
        configs = {
            'low': {
                'method': 'euler',
                'rtol': 1e-3,
                'atol': 1e-4,
            },
            'medium': {
                'method': 'dopri5',
                'rtol': 1e-5,
                'atol': 1e-6,
            },
            'high': {
                'method': 'dopri8',
                'rtol': 1e-7,
                'atol': 1e-8,
            }
        }
        
        if precision not in configs:
            raise ValueError(f"不支持的精度级别: {precision}")
        
        return cls(**configs[precision])


class EulerSolver(ODESolverBase):
    """简单的Euler求解器实现.
    
    当torchdiffeq不可用时的后备实现，提供基础的一阶Euler积分方法。
    性能较低但不依赖外部库。
    """
    
    def solve(
        self,
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        t_span: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """使用Euler方法求解ODE.
        
        实现简单的显式Euler积分：x_{n+1} = x_n + h * f(x_n, t_n)
        """
        self.validate_inputs(x0, t_span)
        
        device_result = self.to_device(x0, t_span)
        if isinstance(device_result, tuple):
            x0, t_span = device_result
        else:
            raise RuntimeError("to_device应该返回两个张量的元组")
        
        # 初始化轨迹存储
        trajectory = torch.zeros(
            (len(t_span),) + x0.shape, 
            device=self.device, 
            dtype=self.dtype
        )
        trajectory[0] = x0
        
        # Euler积分步进
        x = x0
        for i in range(len(t_span) - 1):
            dt = t_span[i + 1] - t_span[i]
            t_current = torch.full((x.shape[0],), t_span[i].item(), device=self.device, dtype=self.dtype)
            
            velocity = vector_field(x, t_current)
            x = x + dt * velocity
            trajectory[i + 1] = x
        
        return trajectory 
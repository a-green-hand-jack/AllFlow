"""模型接口模块

本模块定义了Flow Matching中神经网络模型的统一接口。
支持灵活的模型输入，包括条件信息、额外参数等。

核心组件：
- ModelInterface: 模型接口抽象基类
- SimpleModelWrapper: 简单模型包装器
- ConditionalModelWrapper: 条件模型包装器
- FlexibleModelWrapper: 灵活模型包装器

Author: AllFlow Contributors
License: MIT
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


class ModelInterface(ABC):
    """模型接口抽象基类.

    定义了Flow Matching中神经网络模型的统一调用接口。
    所有模型包装器都必须实现predict_velocity方法。

    这个接口的设计目标是：
    1. 统一不同类型的模型调用方式
    2. 支持条件信息和额外参数
    3. 保持向后兼容性
    4. 便于扩展和测试
    """

    @abstractmethod
    def predict_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """预测速度场.

        Args:
            x_t: 当前位置, shape: (batch_size, *data_shape)
            t: 时间参数, shape: (batch_size,)
            **kwargs: 额外参数

        Returns:
            预测的速度场, shape: (batch_size, *data_shape)
        """
        pass

    def __call__(
        self, x_t: torch.Tensor, t: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """使模型接口可调用."""
        return self.predict_velocity(x_t, t, **kwargs)


class SimpleModelWrapper(ModelInterface):
    """简单模型包装器.

    包装只接受(x_t, t)输入的标准神经网络模型。
    这是最基础的包装器，适用于标准的Flow Matching场景。

    Args:
        model: 神经网络模型，接受(x_t, t)输入

    Example:
        >>> model = MyVelocityNet()  # 接受(x_t, t)的神经网络
        >>> wrapper = SimpleModelWrapper(model)
        >>> velocity = wrapper(x_t, t)
    """

    def __init__(
        self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        """初始化简单模型包装器.

        Args:
            model: 神经网络模型
        """
        self.model = model

    def predict_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """预测速度场.

        Args:
            x_t: 当前位置
            t: 时间参数
            **kwargs: 忽略的额外参数

        Returns:
            预测的速度场
        """
        return self.model(x_t, t)


class ConditionalModelWrapper(ModelInterface):
    """条件模型包装器.

    包装接受条件信息的神经网络模型。
    支持类标签、文本embeddings等条件输入。

    Args:
        model: 神经网络模型，接受(x_t, t, condition)输入
        condition: 条件信息，可以是张量或字典
        condition_key: 条件参数在kwargs中的键名

    Example:
        >>> model = MyConditionalNet()  # 接受(x_t, t, class_label)
        >>> wrapper = ConditionalModelWrapper(
        ...     model=model,
        ...     condition=class_labels,
        ...     condition_key='class_label'
        ... )
        >>> velocity = wrapper(x_t, t)
    """

    def __init__(
        self,
        model: Callable,
        condition: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
        condition_key: str = "condition",
    ) -> None:
        """初始化条件模型包装器.

        Args:
            model: 神经网络模型
            condition: 条件信息
            condition_key: 条件参数的键名
        """
        self.model = model
        self.condition = condition
        self.condition_key = condition_key

    def predict_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """预测速度场.

        Args:
            x_t: 当前位置
            t: 时间参数
            **kwargs: 可能包含条件信息的额外参数

        Returns:
            预测的速度场
        """
        # 获取条件信息：优先使用kwargs中的，然后使用初始化时的
        condition = kwargs.get(self.condition_key, self.condition)

        if condition is not None:
            return self.model(x_t, t, condition)
        else:
            # 如果没有条件信息，降级为简单模型
            logger.warning("没有提供条件信息，使用简单模型调用")
            return self.model(x_t, t)


class FlexibleModelWrapper(ModelInterface):
    """灵活模型包装器.

    支持最灵活的模型调用方式，可以处理任意数量和类型的输入参数。
    适用于复杂的神经网络架构和高级应用场景。

    Args:
        model: 神经网络模型
        input_mapping: 输入参数映射规则
        default_inputs: 默认输入参数

    Example:
        >>> model = MyComplexNet()  # 接受多种参数
        >>> wrapper = FlexibleModelWrapper(
        ...     model=model,
        ...     input_mapping={'position': 'x_t', 'time': 't', 'scale': 'temperature'},
        ...     default_inputs={'temperature': 1.0}
        ... )
        >>> velocity = wrapper(x_t, t, temperature=0.8)
    """

    def __init__(
        self,
        model: Callable,
        input_mapping: Optional[Dict[str, str]] = None,
        default_inputs: Optional[Dict[str, Any]] = None,
        **fixed_inputs: Any,
    ) -> None:
        """初始化灵活模型包装器.

        Args:
            model: 神经网络模型
            input_mapping: 参数名映射，键为模型参数名，值为wrapper参数名
            default_inputs: 默认输入参数
            **fixed_inputs: 固定输入参数
        """
        self.model = model
        self.input_mapping = input_mapping or {"x_t": "x_t", "t": "t"}
        self.default_inputs = default_inputs or {}
        self.fixed_inputs = fixed_inputs

    def predict_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """预测速度场.

        Args:
            x_t: 当前位置
            t: 时间参数
            **kwargs: 额外参数

        Returns:
            预测的速度场
        """
        # 构建模型输入参数
        model_inputs = {}

        # 基础参数
        wrapper_inputs = {"x_t": x_t, "t": t}
        wrapper_inputs.update(kwargs)

        # 应用输入映射
        for model_param, wrapper_param in self.input_mapping.items():
            if wrapper_param in wrapper_inputs:
                model_inputs[model_param] = wrapper_inputs[wrapper_param]

        # 添加默认参数
        for param, value in self.default_inputs.items():
            if param not in model_inputs:
                model_inputs[param] = value

        # 添加固定参数
        model_inputs.update(self.fixed_inputs)

        # 调用模型
        try:
            return self.model(**model_inputs)
        except TypeError as e:
            # 如果参数不匹配，尝试位置参数调用
            logger.warning(f"关键字参数调用失败: {e}，尝试位置参数调用")
            return self.model(x_t, t)


class FunctionModelWrapper(ModelInterface):
    """函数式模型包装器.

    将普通函数包装为模型接口，支持lambda函数和自定义函数。
    适用于简单的测试场景和函数式编程风格。

    Args:
        model_fn: 模型函数

    Example:
        >>> # 使用lambda函数
        >>> wrapper = FunctionModelWrapper(lambda x, t: x * t)
        >>> velocity = wrapper(x_t, t)

        >>> # 使用自定义函数
        >>> def my_velocity_fn(x_t, t, scale=1.0):
        ...     return scale * (x_t - 0.5)
        >>> wrapper = FunctionModelWrapper(my_velocity_fn)
        >>> velocity = wrapper(x_t, t, scale=2.0)
    """

    def __init__(self, model_fn: Callable) -> None:
        """初始化函数式模型包装器.

        Args:
            model_fn: 模型函数
        """
        self.model_fn = model_fn

    def predict_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """预测速度场.

        Args:
            x_t: 当前位置
            t: 时间参数
            **kwargs: 额外参数

        Returns:
            预测的速度场
        """
        try:
            # 首先尝试传递所有参数
            return self.model_fn(x_t, t, **kwargs)
        except TypeError:
            try:
                # 如果失败，尝试只传递基础参数
                return self.model_fn(x_t, t)
            except TypeError as e:
                raise ValueError(f"模型函数调用失败: {e}")


def create_model_wrapper(
    model: Callable, wrapper_type: str = "simple", **wrapper_kwargs: Any
) -> ModelInterface:
    """创建模型包装器的便捷函数.

    Args:
        model: 神经网络模型或函数
        wrapper_type: 包装器类型，支持'simple', 'conditional', 'flexible', 'function'
        **wrapper_kwargs: 包装器特定参数

    Returns:
        模型包装器实例

    Example:
        >>> model = MyVelocityNet()
        >>> wrapper = create_model_wrapper(model, 'conditional', condition=labels)
        >>> velocity = wrapper(x_t, t)
    """
    wrappers = {
        "simple": SimpleModelWrapper,
        "conditional": ConditionalModelWrapper,
        "flexible": FlexibleModelWrapper,
        "function": FunctionModelWrapper,
    }

    if wrapper_type not in wrappers:
        raise ValueError(
            f"不支持的包装器类型: {wrapper_type}，支持的类型: {list(wrappers.keys())}"
        )

    wrapper_class = wrappers[wrapper_type]
    return wrapper_class(model, **wrapper_kwargs)

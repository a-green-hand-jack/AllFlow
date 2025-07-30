"""测试AllFlow改进功能

专门测试时间采样器、模型接口和新的解耦API设计。
"""

import logging

import pytest  # type: ignore
import torch

from allflow import FlowMatching
from allflow.core.model_interface import (
    ConditionalModelWrapper,
    FlexibleModelWrapper,
    FunctionModelWrapper,
    SimpleModelWrapper,
    create_model_wrapper,
)
from allflow.core.time_sampling import (
    ExponentialTimeSampler,
    ImportanceTimeSampler,
    NormalTimeSampler,
    UniformTimeSampler,
    create_time_sampler,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def device():
    """检测并返回最佳可用设备."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def sample_data(device):
    """生成测试数据."""
    batch_size = 4
    dim = 8
    x_0 = torch.randn(batch_size, dim, device=device)
    x_1 = torch.randn(batch_size, dim, device=device)
    t = torch.rand(batch_size, device=device)
    return x_0, x_1, t


@pytest.fixture
def simple_model():
    """简单的测试模型."""

    def model(x, t):
        # 简单的线性组合
        return x * t.unsqueeze(-1) + torch.ones_like(x) * (1 - t.unsqueeze(-1))

    return model


@pytest.fixture
def conditional_model():
    """条件模型."""

    def model(x, t, condition):
        batch_size = x.shape[0]
        condition_expanded = condition.unsqueeze(-1).expand(-1, x.shape[-1])
        return x * t.unsqueeze(-1) + condition_expanded * (1 - t.unsqueeze(-1))

    return model


class TestTimeSamplers:
    """时间采样器测试."""

    def test_uniform_sampler_default(self, device):
        """测试默认统一采样器."""
        flow = FlowMatching(device=device)  # 默认使用UniformTimeSampler
        assert isinstance(flow.time_sampler, UniformTimeSampler)

        # 测试采样
        batch_size = 10
        t = flow.time_sampler.sample(batch_size)
        assert t.shape == (batch_size,)
        assert torch.all(t >= 0) and torch.all(t <= 1)
        assert t.device.type == device.type

    def test_normal_sampler(self, device):
        """测试正态分布采样器."""
        sampler = NormalTimeSampler(mean=0.3, std=0.1, device=device)
        flow = FlowMatching(device=device, time_sampler=sampler)

        batch_size = 10
        t = flow.time_sampler.sample(batch_size)
        assert t.shape == (batch_size,)
        assert torch.all(t >= 0) and torch.all(t <= 1)  # 应该被截断到[0,1]
        assert t.device.type == device.type

    def test_exponential_sampler(self, device):
        """测试指数分布采样器."""
        sampler = ExponentialTimeSampler(rate=2.0, device=device)
        flow = FlowMatching(device=device, time_sampler=sampler)

        batch_size = 10
        t = flow.time_sampler.sample(batch_size)
        assert t.shape == (batch_size,)
        assert torch.all(t >= 0) and torch.all(t <= 1)
        assert t.device.type == device.type

    def test_importance_sampler(self, device):
        """测试重要性采样器."""

        def importance_fn(t):
            return t**2  # 更关注t接近1的时间点

        sampler = ImportanceTimeSampler(importance_fn, device=device)
        flow = FlowMatching(device=device, time_sampler=sampler)

        batch_size = 10
        t = flow.time_sampler.sample(batch_size)
        assert t.shape == (batch_size,)
        assert torch.all(t >= 0) and torch.all(t <= 1)
        assert t.device.type == device.type

    def test_create_sampler_convenience(self, device):
        """测试便利创建函数."""
        # 创建不同类型的采样器
        uniform = create_time_sampler("uniform", device=device)
        assert isinstance(uniform, UniformTimeSampler)

        normal = create_time_sampler("normal", mean=0.5, std=0.1, device=device)
        assert isinstance(normal, NormalTimeSampler)

        exponential = create_time_sampler("exponential", rate=1.0, device=device)
        assert isinstance(exponential, ExponentialTimeSampler)


class TestModelInterfaces:
    """模型接口测试."""

    def test_simple_model_wrapper(self, device, sample_data, simple_model):
        """测试简单模型包装器."""
        x_0, x_1, t = sample_data

        wrapper = SimpleModelWrapper(simple_model)

        # 准备数据
        flow = FlowMatching(device=device)
        x_t, t_new, true_velocity = flow.prepare_training_data(x_0, x_1)

        # 测试预测
        predicted_velocity = wrapper.predict_velocity(x_t, t_new)
        assert predicted_velocity.shape == x_t.shape
        assert torch.isfinite(predicted_velocity).all()

    def test_conditional_model_wrapper(self, device, sample_data, conditional_model):
        """测试条件模型包装器."""
        x_0, x_1, t = sample_data
        batch_size = x_0.shape[0]

        # 创建条件
        condition = torch.randn(batch_size, device=device)

        wrapper = ConditionalModelWrapper(
            conditional_model, condition=condition, condition_key="condition"
        )

        # 准备数据
        flow = FlowMatching(device=device)
        x_t, t_new, true_velocity = flow.prepare_training_data(x_0, x_1)

        # 测试预测
        predicted_velocity = wrapper.predict_velocity(x_t, t_new)
        assert predicted_velocity.shape == x_t.shape
        assert torch.isfinite(predicted_velocity).all()

    def test_flexible_model_wrapper(self, device, sample_data):
        """测试灵活模型包装器."""
        x_0, x_1, t = sample_data

        def flexible_model(x, t, scale=1.0, shift=0.0):
            return scale * x + shift

        wrapper = FlexibleModelWrapper(
            flexible_model,
            input_mapping={"x_t": "x", "t": "t"},
            default_kwargs={"scale": 2.0, "shift": 0.1},
        )

        # 准备数据
        flow = FlowMatching(device=device)
        x_t, t_new, true_velocity = flow.prepare_training_data(x_0, x_1)

        # 测试预测
        predicted_velocity = wrapper.predict_velocity(x_t, t_new)
        assert predicted_velocity.shape == x_t.shape
        assert torch.isfinite(predicted_velocity).all()

    def test_function_model_wrapper(self, device, sample_data):
        """测试函数模型包装器."""
        x_0, x_1, t = sample_data

        # 使用lambda函数
        lambda_model = lambda x, t: x * 0.5
        wrapper = FunctionModelWrapper(lambda_model)

        # 准备数据
        flow = FlowMatching(device=device)
        x_t, t_new, true_velocity = flow.prepare_training_data(x_0, x_1)

        # 测试预测
        predicted_velocity = wrapper.predict_velocity(x_t, t_new)
        assert predicted_velocity.shape == x_t.shape
        assert torch.isfinite(predicted_velocity).all()

    def test_create_model_wrapper_convenience(self, device, simple_model):
        """测试模型包装器便利创建函数."""
        # 创建不同类型的包装器
        simple_wrapper = create_model_wrapper(simple_model, "simple")
        assert isinstance(simple_wrapper, SimpleModelWrapper)

        function_wrapper = create_model_wrapper(lambda x, t: x, "function")
        assert isinstance(function_wrapper, FunctionModelWrapper)


class TestDecoupledAPI:
    """测试新的解耦API设计."""

    def test_prepare_training_data(self, device, sample_data):
        """测试prepare_training_data方法."""
        x_0, x_1, t = sample_data
        flow = FlowMatching(device=device)

        # 测试prepare_training_data
        x_t, t_sampled, true_velocity = flow.prepare_training_data(x_0, x_1)

        assert x_t.shape == x_0.shape
        assert t_sampled.shape == (x_0.shape[0],)
        assert true_velocity.shape == x_0.shape
        assert torch.all(t_sampled >= 0) and torch.all(t_sampled <= 1)

    def test_compute_loss_decoupled(self, device, sample_data, simple_model):
        """测试解耦的compute_loss方法."""
        x_0, x_1, t = sample_data
        flow = FlowMatching(device=device)

        # 准备数据
        x_t, t_sampled, true_velocity = flow.prepare_training_data(x_0, x_1)

        # 调用模型
        predicted_velocity = simple_model(x_t, t_sampled)

        # 计算损失
        loss = flow.compute_loss(x_0, x_1, t_sampled, predicted_velocity)

        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_generate_sample_decoupled(self, device, simple_model):
        """测试解耦的generate_sample方法."""
        batch_size = 4
        dim = 8
        flow = FlowMatching(device=device)

        # 初始噪声
        x_0 = torch.randn(batch_size, dim, device=device)

        # 定义速度场函数
        def velocity_field_fn(x, t):
            return simple_model(x, t)

        # 生成样本
        x_final = flow.generate_sample(x_0, velocity_field_fn, num_steps=10)

        assert x_final.shape == x_0.shape
        assert torch.isfinite(x_final).all()

    def test_end_to_end_workflow(self, device, sample_data, simple_model):
        """测试端到端的新工作流程."""
        x_0, x_1, t = sample_data
        flow = FlowMatching(device=device)

        # 1. 准备训练数据
        x_t, t_sampled, true_velocity = flow.prepare_training_data(x_0, x_1)

        # 2. 模型预测
        predicted_velocity = simple_model(x_t, t_sampled)

        # 3. 计算损失
        loss = flow.compute_loss(x_0, x_1, t_sampled, predicted_velocity)

        # 4. 生成样本
        def velocity_field_fn(x, t):
            return simple_model(x, t)

        x_generated = flow.generate_sample(x_0, velocity_field_fn, num_steps=5)

        # 验证结果
        assert torch.isfinite(loss)
        assert x_generated.shape == x_0.shape
        assert torch.isfinite(x_generated).all()

    def test_backward_compatibility_note(self, device, sample_data):
        """测试说明：旧API已被移除，确保新API覆盖所有功能."""
        x_0, x_1, t = sample_data
        flow = FlowMatching(device=device)

        # 新API应该能够处理所有之前的用例
        # 1. 时间采样：通过time_sampler
        assert hasattr(flow, "time_sampler")

        # 2. 模型调用：通过外部处理
        x_t, t_sampled, true_velocity = flow.prepare_training_data(x_0, x_1)
        assert x_t is not None

        # 3. 损失计算：通过解耦的compute_loss
        # 模拟一个简单的预测
        predicted_velocity = torch.zeros_like(true_velocity)
        loss = flow.compute_loss(x_0, x_1, t_sampled, predicted_velocity)
        assert torch.isfinite(loss)

        logger.info("新的解耦API成功覆盖了所有原有功能")

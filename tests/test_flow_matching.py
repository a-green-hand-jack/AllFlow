"""Flow Matching核心算法单元测试

本模块包含对Flow Matching算法核心功能的全面测试，确保数学正确性、
数值稳定性和跨设备兼容性。

测试内容:
- 数学正确性验证
- 边界条件测试
- 输入验证测试
- 设备兼容性测试
- 性能基准测试

Author: AllFlow Test Suite Contributors
"""

import logging
import sys
from pathlib import Path

import pytest  # type: ignore
import torch

# 将项目根目录下的 src 目录添加到 Python 解释器的搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from allflow.algorithms.flow_matching import FlowMatching
from allflow.core.base import FlowMatchingBase

logger = logging.getLogger(__name__)


class TestFlowMatchingCore:
    """Flow Matching核心功能测试类."""

    @pytest.fixture
    def device(self):
        """自动检测可用设备."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @pytest.fixture
    def flow_matching(self, device):
        """创建FlowMatching实例."""
        return FlowMatching(device=device, dtype=torch.float32)

    @pytest.fixture
    def sample_data(self, device):
        """创建测试用的样本数据."""
        batch_size = 8
        dim = 16

        x_0 = torch.randn(batch_size, dim, device=device)
        x_1 = torch.randn(batch_size, dim, device=device)
        t = torch.rand(batch_size, device=device)

        return x_0, x_1, t

    def test_initialization(self, device):
        """测试Flow Matching初始化."""
        # 测试默认初始化
        flow = FlowMatching()
        assert isinstance(flow, FlowMatchingBase)
        assert flow.sigma_min > 0

        # 测试带参数初始化
        flow = FlowMatching(device=device, sigma_min=1e-4)
        assert flow.device == device
        assert flow.sigma_min == 1e-4

        # 测试错误的sigma_min
        with pytest.raises(ValueError):
            FlowMatching(sigma_min=-1.0)

    def test_sample_trajectory(self, flow_matching, sample_data):
        """测试轨迹采样的数学正确性."""
        x_0, x_1, t = sample_data

        # 基本功能测试
        x_t = flow_matching.sample_trajectory(x_0, x_1, t)

        # 检查输出形状
        assert x_t.shape == x_0.shape
        assert x_t.device == x_0.device
        assert x_t.dtype == x_0.dtype

        # 验证边界条件
        t_zero = torch.zeros_like(t)
        t_one = torch.ones_like(t)

        x_t_0 = flow_matching.sample_trajectory(x_0, x_1, t_zero)
        x_t_1 = flow_matching.sample_trajectory(x_0, x_1, t_one)

        # t=0时应该等于x_0
        assert torch.allclose(x_t_0, x_0, atol=1e-6)

        # t=1时应该等于x_1
        assert torch.allclose(x_t_1, x_1, atol=1e-6)

        # 验证线性插值性质
        t_half = torch.full_like(t, 0.5)
        x_t_half = flow_matching.sample_trajectory(x_0, x_1, t_half)
        expected_half = 0.5 * x_0 + 0.5 * x_1

        assert torch.allclose(x_t_half, expected_half, atol=1e-6)

    def test_compute_vector_field(self, flow_matching, sample_data):
        """测试速度场计算的正确性."""
        x_0, x_1, t = sample_data
        x_t = flow_matching.sample_trajectory(x_0, x_1, t)

        # 计算速度场
        velocity = flow_matching.compute_vector_field(x_t, t, x_0=x_0, x_1=x_1)

        # 检查输出形状
        assert velocity.shape == x_t.shape
        assert velocity.device == x_t.device
        assert velocity.dtype == x_t.dtype

        # 验证Flow Matching的速度场公式：u_t(x) = x_1 - x_0
        expected_velocity = x_1 - x_0
        assert torch.allclose(velocity, expected_velocity, atol=1e-6)

        # 验证速度场不依赖于x_t和t（Flow Matching的特性）
        x_t_alt = torch.randn_like(x_t)
        t_alt = torch.rand_like(t)
        velocity_alt = flow_matching.compute_vector_field(
            x_t_alt, t_alt, x_0=x_0, x_1=x_1
        )

        assert torch.allclose(velocity, velocity_alt, atol=1e-6)

    def test_compute_loss(self, flow_matching, sample_data):
        """测试损失函数计算."""
        x_0, x_1, t = sample_data

        # 创建一个简单的模型用于测试
        def mock_model(x_t, t):
            # 返回正确的速度场用于测试
            return x_1 - x_0

        # 计算损失
        loss = flow_matching.compute_loss(x_0, x_1, model=mock_model)

        # 检查损失形状和值
        assert loss.dim() == 0  # 标量
        assert loss >= 0  # 损失应该非负
        assert torch.isfinite(loss)  # 损失应该是有限值

        # 对于完美模型，损失应该接近0
        assert loss < 1e-5

        # 测试错误的模型
        def bad_model(x_t, t):
            return torch.zeros_like(x_t)

        bad_loss = flow_matching.compute_loss(x_0, x_1, model=bad_model)
        assert bad_loss > loss  # 错误模型的损失应该更大

    def test_input_validation(self, flow_matching, device):
        """测试输入验证功能."""
        batch_size = 4
        dim = 8

        # 正确的输入
        x_0 = torch.randn(batch_size, dim, device=device)
        x_1 = torch.randn(batch_size, dim, device=device)
        t = torch.rand(batch_size, device=device)

        # 应该正常工作
        flow_matching.validate_inputs(x_0, x_1, t)

        # 测试形状不匹配
        x_1_wrong = torch.randn(batch_size, dim + 1, device=device)
        with pytest.raises(ValueError, match="形状必须相同"):
            flow_matching.validate_inputs(x_0, x_1_wrong, t)

        # 测试批量大小不匹配
        t_wrong = torch.rand(batch_size + 1, device=device)
        with pytest.raises(ValueError, match="批量大小必须"):
            flow_matching.validate_inputs(x_0, x_1, t_wrong)

        # 测试时间参数超出范围
        t_invalid = torch.tensor([-0.1, 0.5, 1.1, 0.8], device=device)
        with pytest.raises(ValueError, match="时间参数t必须在"):
            flow_matching.validate_inputs(x_0, x_1, t_invalid)

        # 测试NaN值
        x_0_nan = x_0.clone()
        x_0_nan[0, 0] = float("nan")
        with pytest.raises(ValueError, match="包含NaN值"):
            flow_matching.validate_inputs(x_0_nan, x_1, t)

        # 测试Inf值
        x_1_inf = x_1.clone()
        x_1_inf[0, 0] = float("inf")
        with pytest.raises(ValueError, match="包含Inf值"):
            flow_matching.validate_inputs(x_0, x_1_inf, t)

    def test_device_compatibility(self):
        """测试跨设备兼容性."""
        batch_size = 4
        dim = 8

        # 测试CPU
        flow_cpu = FlowMatching(device="cpu")
        x_0_cpu = torch.randn(batch_size, dim)
        x_1_cpu = torch.randn(batch_size, dim)
        t_cpu = torch.rand(batch_size)

        x_t_cpu = flow_cpu.sample_trajectory(x_0_cpu, x_1_cpu, t_cpu)
        assert x_t_cpu.device.type == "cpu"

        # 测试GPU（如果可用）
        if torch.cuda.is_available():
            flow_cuda = FlowMatching(device="cuda")
            x_0_cuda = torch.randn(batch_size, dim, device="cuda")
            x_1_cuda = torch.randn(batch_size, dim, device="cuda")
            t_cuda = torch.rand(batch_size, device="cuda")

            x_t_cuda = flow_cuda.sample_trajectory(x_0_cuda, x_1_cuda, t_cuda)
            assert x_t_cuda.device.type == "cuda"

        # 测试MPS（如果可用）
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            flow_mps = FlowMatching(device="mps")
            x_0_mps = torch.randn(batch_size, dim, device="mps")
            x_1_mps = torch.randn(batch_size, dim, device="mps")
            t_mps = torch.rand(batch_size, device="mps")

            x_t_mps = flow_mps.sample_trajectory(x_0_mps, x_1_mps, t_mps)
            assert x_t_mps.device.type == "mps"

    def test_batch_processing(self, flow_matching, device):
        """测试批量处理功能."""
        dims = [4, 16, 64]
        batch_sizes = [1, 8, 32]

        for dim in dims:
            for batch_size in batch_sizes:
                x_0 = torch.randn(batch_size, dim, device=device)
                x_1 = torch.randn(batch_size, dim, device=device)
                t = torch.rand(batch_size, device=device)

                # 测试轨迹采样
                x_t = flow_matching.sample_trajectory(x_0, x_1, t)
                assert x_t.shape == (batch_size, dim)

                # 测试速度场计算
                velocity = flow_matching.compute_vector_field(x_t, t, x_0=x_0, x_1=x_1)
                assert velocity.shape == (batch_size, dim)

    def test_numerical_stability(self, flow_matching, device):
        """测试数值稳定性."""
        batch_size = 4
        dim = 8

        # 测试极小的sigma_min
        flow_small = FlowMatching(device=device, sigma_min=1e-10)

        x_0 = torch.randn(batch_size, dim, device=device)
        x_1 = torch.randn(batch_size, dim, device=device)
        t = torch.rand(batch_size, device=device)

        # 应该仍然能正常工作
        x_t = flow_small.sample_trajectory(x_0, x_1, t)
        velocity = flow_small.compute_vector_field(x_t, t, x_0=x_0, x_1=x_1)

        # 检查没有NaN或Inf
        assert torch.isfinite(x_t).all()
        assert torch.isfinite(velocity).all()

        # 测试大数值
        x_0_large = torch.randn(batch_size, dim, device=device) * 1000
        x_1_large = torch.randn(batch_size, dim, device=device) * 1000

        x_t_large = flow_matching.sample_trajectory(x_0_large, x_1_large, t)
        velocity_large = flow_matching.compute_vector_field(
            x_t_large, t, x_0=x_0_large, x_1=x_1_large
        )

        assert torch.isfinite(x_t_large).all()
        assert torch.isfinite(velocity_large).all()

    def test_generate_sample_basic(self, flow_matching, device):
        """测试基础采样生成功能."""
        batch_size = 4
        dim = 8

        # 创建简单的测试模型
        def test_model(x, t):
            # 简单的线性模型
            return torch.zeros_like(x)

        x_0 = torch.randn(batch_size, dim, device=device)

        # 测试Euler积分
        x_final_euler = flow_matching.generate_sample(
            x_0, test_model, num_steps=10, method="euler"
        )
        assert x_final_euler.shape == x_0.shape
        assert torch.isfinite(x_final_euler).all()

        # 测试Heun积分
        x_final_heun = flow_matching.generate_sample(
            x_0, test_model, num_steps=10, method="heun"
        )
        assert x_final_heun.shape == x_0.shape
        assert torch.isfinite(x_final_heun).all()

        # 测试不支持的方法
        with pytest.raises(ValueError):
            flow_matching.generate_sample(x_0, test_model, method="unsupported")


@pytest.mark.benchmark
class TestFlowMatchingPerformance:
    """Flow Matching性能测试类."""

    def test_performance_scaling(self, benchmark):
        """测试性能随批量大小的扩展性."""
        device = torch.device("cpu")  # 使用CPU确保一致性
        flow = FlowMatching(device=device)

        def run_flow_matching():
            batch_size = 64
            dim = 128

            x_0 = torch.randn(batch_size, dim, device=device)
            x_1 = torch.randn(batch_size, dim, device=device)
            t = torch.rand(batch_size, device=device)

            x_t = flow.sample_trajectory(x_0, x_1, t)
            velocity = flow.compute_vector_field(x_t, t, x_0=x_0, x_1=x_1)

            return x_t, velocity

        # 运行基准测试
        result = benchmark(run_flow_matching)
        x_t, velocity = result

        # 验证结果正确性
        assert x_t.shape == (64, 128)
        assert velocity.shape == (64, 128)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 运行测试
    pytest.main([__file__, "-v"])

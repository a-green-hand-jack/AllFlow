"""路径插值系统测试

测试欧几里得和SO(3)空间中的路径插值实现。

Author: AllFlow Contributors
License: MIT
"""

import logging
import math
import pytest
import torch

from allflow.core.interpolation import (
    EuclideanInterpolation,
    SO3Interpolation,
    create_interpolation,
)

logger = logging.getLogger(__name__)


class TestEuclideanInterpolation:
    """欧几里得空间线性插值测试."""

    def test_initialization(self):
        """测试初始化."""
        interp = EuclideanInterpolation()
        assert interp.eps == 1e-8
        
        interp_custom = EuclideanInterpolation(eps=1e-6)
        assert interp_custom.eps == 1e-6

    def test_basic_interpolation(self):
        """测试基础线性插值."""
        interp = EuclideanInterpolation()
        
        # 简单2D向量测试
        x_0 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        x_1 = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
        t = torch.tensor([0.0, 1.0])
        
        result = interp.interpolate(x_0, x_1, t)
        
        # t=0时应该返回x_0，t=1时应该返回x_1
        expected = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
        assert torch.allclose(result, expected)

    def test_midpoint_interpolation(self):
        """测试中点插值."""
        interp = EuclideanInterpolation()
        
        x_0 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        x_1 = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        t = torch.tensor([0.5, 0.5])  # 中点
        
        result = interp.interpolate(x_0, x_1, t)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 中点值
        
        assert torch.allclose(result, expected)

    def test_batch_interpolation(self):
        """测试批量插值."""
        interp = EuclideanInterpolation()
        
        batch_size = 4
        dim = 3
        x_0 = torch.randn(batch_size, dim)
        x_1 = torch.randn(batch_size, dim)
        t = torch.rand(batch_size)
        
        result = interp.interpolate(x_0, x_1, t)
        
        # 验证形状
        assert result.shape == (batch_size, dim)
        
        # 验证边界条件
        x_0_test = torch.randn(2, dim)
        x_1_test = torch.randn(2, dim)
        
        # t=0
        t_zero = torch.zeros(2)
        result_zero = interp.interpolate(x_0_test, x_1_test, t_zero)
        assert torch.allclose(result_zero, x_0_test)
        
        # t=1
        t_one = torch.ones(2)
        result_one = interp.interpolate(x_0_test, x_1_test, t_one)
        assert torch.allclose(result_one, x_1_test)

    def test_high_dimensional_interpolation(self):
        """测试高维数据插值."""
        interp = EuclideanInterpolation()
        
        # 测试图像大小的数据
        batch_size = 2
        x_0 = torch.zeros(batch_size, 3, 32, 32)
        x_1 = torch.ones(batch_size, 3, 32, 32)
        t = torch.tensor([0.25, 0.75])
        
        result = interp.interpolate(x_0, x_1, t)
        
        # 验证形状
        assert result.shape == (batch_size, 3, 32, 32)
        
        # 验证插值值
        expected = torch.tensor([0.25, 0.75]).view(2, 1, 1, 1).expand_as(result)
        assert torch.allclose(result, expected)

    def test_shape_validation(self):
        """测试形状验证."""
        interp = EuclideanInterpolation()
        
        # 不匹配的x_0和x_1形状
        x_0 = torch.randn(2, 3)
        x_1 = torch.randn(2, 4)  # 不同的特征维度
        t = torch.rand(2)
        
        with pytest.raises(ValueError, match="x_0和x_1形状不匹配"):
            interp.interpolate(x_0, x_1, t)
        
        # 不匹配的批量大小
        x_0 = torch.randn(2, 3)
        x_1 = torch.randn(2, 3)
        t = torch.rand(3)  # 不同的批量大小
        
        with pytest.raises(ValueError, match="批量大小不匹配"):
            interp.interpolate(x_0, x_1, t)

    def test_device_compatibility(self):
        """测试设备兼容性."""
        interp = EuclideanInterpolation()
        
        # 测试CPU
        x_0 = torch.randn(2, 3)
        x_1 = torch.randn(2, 3)
        t = torch.rand(2)
        
        result = interp.interpolate(x_0, x_1, t)
        assert result.device.type == 'cpu'
        
        # 如果有CUDA或MPS，测试GPU
        if torch.cuda.is_available():
            x_0_gpu = x_0.cuda()
            x_1_gpu = x_1.cuda()
            t_gpu = t.cuda()
            
            result_gpu = interp.interpolate(x_0_gpu, x_1_gpu, t_gpu)
            assert result_gpu.device.type == 'cuda'
        
        if torch.backends.mps.is_available():
            x_0_mps = x_0.to('mps')
            x_1_mps = x_1.to('mps')
            t_mps = t.to('mps')
            
            result_mps = interp.interpolate(x_0_mps, x_1_mps, t_mps)
            assert result_mps.device.type == 'mps'


class TestSO3Interpolation:
    """SO(3)旋转空间球面插值测试."""

    def test_initialization(self):
        """测试初始化."""
        interp = SO3Interpolation()
        assert interp.eps == 1e-6
        assert interp.dot_threshold == 0.9995
        
        interp_custom = SO3Interpolation(eps=1e-7, dot_threshold=0.999)
        assert interp_custom.eps == 1e-7
        assert interp_custom.dot_threshold == 0.999

    def test_quaternion_normalization(self):
        """测试四元数归一化."""
        interp = SO3Interpolation()
        
        # 非单位四元数
        q = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0]])
        q_norm = interp.normalize_quaternion(q)
        
        # 验证归一化后的模长为1
        norms = torch.norm(q_norm, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_quaternion_dot_product(self):
        """测试四元数点积."""
        interp = SO3Interpolation()
        
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        q2 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-0.5, 0.5, 0.5, 0.5]])
        
        dot = interp.quaternion_dot(q1, q2)
        expected = torch.tensor([1.0, 0.0])  # 第一个完全相同，第二个正交
        
        assert torch.allclose(dot, expected, atol=1e-6)

    def test_identity_quaternion_interpolation(self):
        """测试恒等四元数插值."""
        interp = SO3Interpolation()
        
        # 两个相同的恒等四元数
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q_0 = identity.repeat(2, 1)
        q_1 = identity.repeat(2, 1)
        t = torch.tensor([0.3, 0.7])
        
        result = interp.interpolate(q_0, q_1, t)
        
        # 结果应该仍然是恒等四元数
        expected = identity.repeat(2, 1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_boundary_interpolation(self):
        """测试边界插值（t=0和t=1）."""
        interp = SO3Interpolation()
        
        # 90度绕Z轴旋转的四元数
        q_0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # 恒等旋转
        q_1 = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])  # 90度Z轴
        
        # t=0
        t_zero = torch.tensor([0.0])
        result_zero = interp.interpolate(q_0, q_1, t_zero)
        assert torch.allclose(result_zero, q_0, atol=1e-6)
        
        # t=1
        t_one = torch.tensor([1.0])
        result_one = interp.interpolate(q_0, q_1, t_one)
        assert torch.allclose(result_one, q_1, atol=1e-6)

    def test_90_degree_rotation_interpolation(self):
        """测试90度旋转的插值."""
        interp = SO3Interpolation()
        
        # 恒等旋转到90度绕Z轴旋转
        q_0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q_1 = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
        
        # 中点插值（应该是45度旋转）
        t_half = torch.tensor([0.5])
        result_half = interp.interpolate(q_0, q_1, t_half)
        
        # 验证结果是单位四元数
        norm = torch.norm(result_half, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)
        
        # 验证是大约45度的旋转
        expected_angle = math.pi / 4  # 45度
        actual_angle = 2 * torch.acos(torch.abs(result_half[0, 0]))
        assert torch.allclose(actual_angle, torch.tensor(expected_angle), atol=1e-2)

    def test_batch_quaternion_interpolation(self):
        """测试批量四元数插值."""
        interp = SO3Interpolation()
        
        batch_size = 5
        
        # 随机单位四元数
        q_0 = torch.randn(batch_size, 4)
        q_0 = interp.normalize_quaternion(q_0)
        
        q_1 = torch.randn(batch_size, 4)
        q_1 = interp.normalize_quaternion(q_1)
        
        t = torch.rand(batch_size)
        
        result = interp.interpolate(q_0, q_1, t)
        
        # 验证形状
        assert result.shape == (batch_size, 4)
        
        # 验证所有结果都是单位四元数
        norms = torch.norm(result, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_shape_validation(self):
        """测试形状验证."""
        interp = SO3Interpolation()
        
        # 非四元数形状（最后一维不是4）
        x_0 = torch.randn(2, 3)  # 不是四元数
        x_1 = torch.randn(2, 4)
        t = torch.rand(2)
        
        with pytest.raises(ValueError, match="SO\\(3\\)插值需要四元数输入"):
            interp.interpolate(x_0, x_1, t)
        
        # 不匹配的批量大小
        x_0 = torch.randn(2, 4)
        x_1 = torch.randn(2, 4)
        t = torch.rand(3)
        
        with pytest.raises(ValueError, match="批量大小不匹配"):
            interp.interpolate(x_0, x_1, t)

    def test_shortest_path_selection(self):
        """测试最短路径选择（处理四元数的双重覆盖）."""
        interp = SO3Interpolation()
        
        # 正四元数和负四元数表示相同旋转
        q_0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q_1 = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])  # 负四元数，但相同旋转
        
        t = torch.tensor([0.5])
        result = interp.interpolate(q_0, q_1, t)
        
        # 应该选择较短的路径，结果接近恒等旋转
        assert torch.allclose(torch.abs(result), torch.abs(q_0), atol=1e-5)


class TestInterpolationFactory:
    """插值器工厂函数测试."""

    def test_create_euclidean_interpolation(self):
        """测试创建欧几里得插值器."""
        interp = create_interpolation('euclidean')
        assert isinstance(interp, EuclideanInterpolation)
        
        interp_custom = create_interpolation('euclidean', eps=1e-6)
        assert isinstance(interp_custom, EuclideanInterpolation)
        assert interp_custom.eps == 1e-6

    def test_create_so3_interpolation(self):
        """测试创建SO(3)插值器."""
        interp = create_interpolation('so3')
        assert isinstance(interp, SO3Interpolation)
        
        interp_custom = create_interpolation('so3', eps=1e-7, dot_threshold=0.999)
        assert isinstance(interp_custom, SO3Interpolation)
        assert interp_custom.eps == 1e-7
        assert interp_custom.dot_threshold == 0.999

    def test_case_insensitive_creation(self):
        """测试大小写不敏感的创建."""
        interp1 = create_interpolation('EUCLIDEAN')
        interp2 = create_interpolation('So3')
        interp3 = create_interpolation('SO3')
        
        assert isinstance(interp1, EuclideanInterpolation)
        assert isinstance(interp2, SO3Interpolation)
        assert isinstance(interp3, SO3Interpolation)

    def test_invalid_interpolation_type(self):
        """测试无效插值类型."""
        with pytest.raises(ValueError, match="不支持的插值类型"):
            create_interpolation('invalid_type')

    def test_interpolation_performance(self):
        """测试插值性能（简单基准）."""
        euclidean = create_interpolation('euclidean')
        so3 = create_interpolation('so3')
        
        # 大批量测试
        batch_size = 1000
        
        # 欧几里得插值性能
        x_0_euc = torch.randn(batch_size, 128)
        x_1_euc = torch.randn(batch_size, 128)
        t_euc = torch.rand(batch_size)
        
        import time
        start = time.time()
        result_euc = euclidean.interpolate(x_0_euc, x_1_euc, t_euc)
        euclidean_time = time.time() - start
        
        # SO(3)插值性能
        x_0_so3 = torch.randn(batch_size, 4)
        x_0_so3 = x_0_so3 / torch.norm(x_0_so3, dim=-1, keepdim=True)
        x_1_so3 = torch.randn(batch_size, 4)
        x_1_so3 = x_1_so3 / torch.norm(x_1_so3, dim=-1, keepdim=True)
        t_so3 = torch.rand(batch_size)
        
        start = time.time()
        result_so3 = so3.interpolate(x_0_so3, x_1_so3, t_so3)
        so3_time = time.time() - start
        
        # 验证结果形状
        assert result_euc.shape == (batch_size, 128)
        assert result_so3.shape == (batch_size, 4)
        
        # 打印性能信息（用于开发参考）
        logger.info(f"欧几里得插值时间 (批量{batch_size}): {euclidean_time:.4f}s")
        logger.info(f"SO(3)插值时间 (批量{batch_size}): {so3_time:.4f}s")
        
        # 性能断言（SO(3)应该比欧几里得慢，但不应该慢太多）
        assert so3_time < euclidean_time * 50  # 允许50倍的性能差异 
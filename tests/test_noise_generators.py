"""噪声生成器系统测试

测试不同几何空间的噪声生成器实现。

Author: AllFlow Contributors  
License: MIT
"""

import logging
import math
import pytest
import torch

from allflow.core.noise_generators import (
    NoiseGeneratorBase,
    GaussianNoiseGenerator,
    SO3NoiseGenerator,
    UniformNoiseGenerator,
    create_noise_generator,
)

logger = logging.getLogger(__name__)


class TestGaussianNoiseGenerator:
    """高斯噪声生成器测试."""

    def test_initialization(self):
        """测试初始化."""
        gen = GaussianNoiseGenerator()
        assert gen.std == 1.0
        assert gen.mean == 0.0
        
        gen_custom = GaussianNoiseGenerator(std=0.5, mean=2.0)
        assert gen_custom.std == 0.5
        assert gen_custom.mean == 2.0

    def test_sample_like_basic(self):
        """测试sample_like基础功能."""
        gen = GaussianNoiseGenerator()
        
        target = torch.zeros(3, 4)
        noise = gen.sample_like(target)
        
        # 验证形状和设备
        assert noise.shape == target.shape
        assert noise.device == target.device
        assert noise.dtype == target.dtype

    def test_sample_like_statistics(self):
        """测试sample_like的统计特性."""
        gen = GaussianNoiseGenerator(std=2.0, mean=1.0)
        
        # 大样本测试统计特性
        target = torch.zeros(10000, 1)
        noise = gen.sample_like(target)
        
        # 验证均值和标准差（允许一定误差）
        actual_mean = noise.mean().item()
        actual_std = noise.std().item()
        
        assert abs(actual_mean - 1.0) < 0.1  # 允许10%误差
        assert abs(actual_std - 2.0) < 0.2   # 允许10%误差

    def test_sample_basic(self):
        """测试sample基础功能."""
        gen = GaussianNoiseGenerator()
        
        shape = torch.Size([2, 3, 4])
        noise = gen.sample(shape)
        
        assert noise.shape == shape
        assert noise.device.type == 'cpu'
        assert noise.dtype == torch.float32

    def test_sample_with_parameters(self):
        """测试sample带参数."""
        gen = GaussianNoiseGenerator()
        
        shape = torch.Size([5, 10])
        device = 'cpu'
        dtype = torch.float64
        
        noise = gen.sample(shape, device=device, dtype=dtype)
        
        assert noise.shape == shape
        assert noise.device.type == device
        assert noise.dtype == dtype

    def test_high_dimensional_sampling(self):
        """测试高维采样."""
        gen = GaussianNoiseGenerator()
        
        # 模拟图像数据
        target = torch.zeros(4, 3, 64, 64)
        noise = gen.sample_like(target)
        
        assert noise.shape == (4, 3, 64, 64)
        
        # 验证每个样本都不相同
        assert not torch.allclose(noise[0], noise[1])

    def test_device_compatibility(self):
        """测试设备兼容性."""
        gen = GaussianNoiseGenerator()
        
        # CPU测试
        target_cpu = torch.zeros(2, 3)
        noise_cpu = gen.sample_like(target_cpu)
        assert noise_cpu.device.type == 'cpu'
        
        # GPU测试（如果可用）
        if torch.cuda.is_available():
            target_cuda = torch.zeros(2, 3, device='cuda')
            noise_cuda = gen.sample_like(target_cuda)
            assert noise_cuda.device.type == 'cuda'
        
        if torch.backends.mps.is_available():
            target_mps = torch.zeros(2, 3, device='mps')
            noise_mps = gen.sample_like(target_mps)
            assert noise_mps.device.type == 'mps'


class TestSO3NoiseGenerator:
    """SO(3)噪声生成器测试."""

    def test_initialization(self):
        """测试初始化."""
        gen = SO3NoiseGenerator()
        assert gen.eps == 1e-8
        
        gen_custom = SO3NoiseGenerator(eps=1e-6)
        assert gen_custom.eps == 1e-6

    def test_sample_uniform_quaternion_basic(self):
        """测试基础四元数采样."""
        gen = SO3NoiseGenerator()
        
        batch_size = 10
        quaternions = gen.sample_uniform_quaternion(batch_size)
        
        # 验证形状
        assert quaternions.shape == (batch_size, 4)
        
        # 验证所有四元数都是单位的
        norms = torch.norm(quaternions, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_sample_uniform_quaternion_statistics(self):
        """测试四元数采样的统计特性."""
        gen = SO3NoiseGenerator()
        
        # 大样本测试
        batch_size = 10000
        quaternions = gen.sample_uniform_quaternion(batch_size)
        
        # 验证w分量的分布（应该在[-1, 1]范围内）
        w_components = quaternions[:, 0]
        assert torch.all(w_components >= -1.0)
        assert torch.all(w_components <= 1.0)
        
        # 验证所有四元数都是单位的
        norms = torch.norm(quaternions, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_sample_like_quaternion(self):
        """测试sample_like四元数生成."""
        gen = SO3NoiseGenerator()
        
        # 目标四元数张量
        target = torch.zeros(5, 4)
        noise = gen.sample_like(target)
        
        # 验证形状
        assert noise.shape == target.shape
        
        # 验证所有结果都是单位四元数
        norms = torch.norm(noise, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_sample_like_batch_quaternion(self):
        """测试批量四元数生成."""
        gen = SO3NoiseGenerator()
        
        # 批量四元数
        target = torch.zeros(3, 2, 4)  # 3个批次，每个2个四元数
        noise = gen.sample_like(target)
        
        assert noise.shape == (3, 2, 4)
        
        # 验证所有都是单位四元数
        norms = torch.norm(noise, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_sample_with_shape(self):
        """测试指定形状采样."""
        gen = SO3NoiseGenerator()
        
        shape = torch.Size([4, 4])  # 4个四元数
        noise = gen.sample(shape)
        
        assert noise.shape == shape
        
        # 验证所有都是单位四元数
        norms = torch.norm(noise, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_invalid_shape_sample_like(self):
        """测试无效形状的sample_like."""
        gen = SO3NoiseGenerator()
        
        # 最后一维不是4
        target = torch.zeros(2, 3)  # 不是四元数格式
        
        with pytest.raises(ValueError, match="SO\\(3\\)噪声生成需要四元数格式"):
            gen.sample_like(target)

    def test_invalid_shape_sample(self):
        """测试无效形状的sample."""
        gen = SO3NoiseGenerator()
        
        # 最后一维不是4
        shape = torch.Size([2, 3])
        
        with pytest.raises(ValueError, match="SO\\(3\\)噪声生成需要四元数格式"):
            gen.sample(shape)

    def test_quaternion_diversity(self):
        """测试四元数的多样性（不应该总是生成相同的）."""
        gen = SO3NoiseGenerator()
        
        # 生成多个批次，验证它们不相同
        q1 = gen.sample_uniform_quaternion(5)
        q2 = gen.sample_uniform_quaternion(5)
        
        # 应该不完全相同
        assert not torch.allclose(q1, q2, atol=1e-3)

    def test_device_compatibility(self):
        """测试设备兼容性."""
        gen = SO3NoiseGenerator()
        
        # CPU测试
        target_cpu = torch.zeros(2, 4)
        noise_cpu = gen.sample_like(target_cpu)
        assert noise_cpu.device.type == 'cpu'
        
        # GPU测试（如果可用）
        if torch.cuda.is_available():
            target_cuda = torch.zeros(2, 4, device='cuda')
            noise_cuda = gen.sample_like(target_cuda)
            assert noise_cuda.device.type == 'cuda'
        
        if torch.backends.mps.is_available():
            target_mps = torch.zeros(2, 4, device='mps')
            noise_mps = gen.sample_like(target_mps)
            assert noise_mps.device.type == 'mps'

    def test_marsaglia_method_validation(self):
        """验证Marsaglia方法实现的正确性."""
        gen = SO3NoiseGenerator()
        
        # 生成大量样本
        batch_size = 1000
        quaternions = gen.sample_uniform_quaternion(batch_size)
        
        # 计算旋转角度分布（应该符合预期的分布）
        angles = 2 * torch.acos(torch.abs(quaternions[:, 0]))
        
        # 验证角度在[0, π]范围内
        assert torch.all(angles >= 0)
        assert torch.all(angles <= math.pi + 1e-6)  # 允许小的数值误差


class TestUniformNoiseGenerator:
    """均匀噪声生成器测试."""

    def test_initialization(self):
        """测试初始化."""
        gen = UniformNoiseGenerator()
        assert gen.min_val == -1.0
        assert gen.max_val == 1.0
        
        gen_custom = UniformNoiseGenerator(min_val=-2.0, max_val=3.0)
        assert gen_custom.min_val == -2.0
        assert gen_custom.max_val == 3.0

    def test_invalid_range_initialization(self):
        """测试无效范围初始化."""
        with pytest.raises(ValueError, match="min_val .* 必须小于 max_val"):
            UniformNoiseGenerator(min_val=1.0, max_val=0.0)
        
        with pytest.raises(ValueError, match="min_val .* 必须小于 max_val"):
            UniformNoiseGenerator(min_val=5.0, max_val=5.0)

    def test_sample_like_basic(self):
        """测试sample_like基础功能."""
        gen = UniformNoiseGenerator(min_val=-2.0, max_val=3.0)
        
        target = torch.zeros(4, 5)
        noise = gen.sample_like(target)
        
        # 验证形状
        assert noise.shape == target.shape
        
        # 验证值在指定范围内
        assert torch.all(noise >= -2.0)
        assert torch.all(noise <= 3.0)

    def test_sample_like_statistics(self):
        """测试sample_like的统计特性."""
        gen = UniformNoiseGenerator(min_val=-4.0, max_val=6.0)
        
        # 大样本测试
        target = torch.zeros(50000, 1)
        noise = gen.sample_like(target)
        
        # 验证均值接近中点
        expected_mean = (-4.0 + 6.0) / 2  # 1.0
        actual_mean = noise.mean().item()
        assert abs(actual_mean - expected_mean) < 0.1

    def test_sample_basic(self):
        """测试sample基础功能."""
        gen = UniformNoiseGenerator(min_val=0.0, max_val=1.0)
        
        shape = torch.Size([3, 4])
        noise = gen.sample(shape)
        
        assert noise.shape == shape
        assert torch.all(noise >= 0.0)
        assert torch.all(noise <= 1.0)

    def test_sample_with_parameters(self):
        """测试sample带参数."""
        gen = UniformNoiseGenerator(min_val=-1.0, max_val=1.0)
        
        shape = torch.Size([2, 3])
        device = 'cpu'
        dtype = torch.float64
        
        noise = gen.sample(shape, device=device, dtype=dtype)
        
        assert noise.shape == shape
        assert noise.device.type == device
        assert noise.dtype == dtype
        assert torch.all(noise >= -1.0)
        assert torch.all(noise <= 1.0)

    def test_device_compatibility(self):
        """测试设备兼容性."""
        gen = UniformNoiseGenerator()
        
        # CPU测试
        target_cpu = torch.zeros(2, 3)
        noise_cpu = gen.sample_like(target_cpu)
        assert noise_cpu.device.type == 'cpu'
        
        # GPU测试（如果可用）
        if torch.cuda.is_available():
            target_cuda = torch.zeros(2, 3, device='cuda')
            noise_cuda = gen.sample_like(target_cuda)
            assert noise_cuda.device.type == 'cuda'
        
        if torch.backends.mps.is_available():
            target_mps = torch.zeros(2, 3, device='mps')
            noise_mps = gen.sample_like(target_mps)
            assert noise_mps.device.type == 'mps'


class TestNoiseGeneratorFactory:
    """噪声生成器工厂函数测试."""

    def test_create_gaussian_generator(self):
        """测试创建高斯噪声生成器."""
        gen = create_noise_generator('gaussian')
        assert isinstance(gen, GaussianNoiseGenerator)
        
        gen_custom = create_noise_generator('gaussian', std=0.5, mean=1.0)
        assert isinstance(gen_custom, GaussianNoiseGenerator)
        assert gen_custom.std == 0.5
        assert gen_custom.mean == 1.0

    def test_create_so3_generator(self):
        """测试创建SO(3)噪声生成器."""
        gen = create_noise_generator('so3')
        assert isinstance(gen, SO3NoiseGenerator)
        
        gen_custom = create_noise_generator('so3', eps=1e-7)
        assert isinstance(gen_custom, SO3NoiseGenerator)
        assert gen_custom.eps == 1e-7

    def test_create_uniform_generator(self):
        """测试创建均匀噪声生成器."""
        gen = create_noise_generator('uniform')
        assert isinstance(gen, UniformNoiseGenerator)
        
        gen_custom = create_noise_generator('uniform', min_val=-2.0, max_val=5.0)
        assert isinstance(gen_custom, UniformNoiseGenerator)
        assert gen_custom.min_val == -2.0
        assert gen_custom.max_val == 5.0

    def test_case_insensitive_creation(self):
        """测试大小写不敏感的创建."""
        gen1 = create_noise_generator('GAUSSIAN')
        gen2 = create_noise_generator('So3')
        gen3 = create_noise_generator('UNIFORM')
        
        assert isinstance(gen1, GaussianNoiseGenerator)
        assert isinstance(gen2, SO3NoiseGenerator)
        assert isinstance(gen3, UniformNoiseGenerator)

    def test_invalid_generator_type(self):
        """测试无效生成器类型."""
        with pytest.raises(ValueError, match="不支持的噪声生成器类型"):
            create_noise_generator('invalid_type')

    def test_noise_generator_performance(self):
        """测试噪声生成器性能."""
        gaussian = create_noise_generator('gaussian')
        so3 = create_noise_generator('so3')
        uniform = create_noise_generator('uniform')
        
        # 大批量测试
        batch_size = 5000
        
        # 高斯噪声性能
        target_gaussian = torch.zeros(batch_size, 128)
        import time
        start = time.time()
        noise_gaussian = gaussian.sample_like(target_gaussian)
        gaussian_time = time.time() - start
        
        # SO(3)噪声性能
        target_so3 = torch.zeros(batch_size, 4)
        start = time.time()
        noise_so3 = so3.sample_like(target_so3)
        so3_time = time.time() - start
        
        # 均匀噪声性能
        target_uniform = torch.zeros(batch_size, 128)
        start = time.time()
        noise_uniform = uniform.sample_like(target_uniform)
        uniform_time = time.time() - start
        
        # 验证结果
        assert noise_gaussian.shape == (batch_size, 128)
        assert noise_so3.shape == (batch_size, 4)
        assert noise_uniform.shape == (batch_size, 128)
        
        # 打印性能信息
        logger.info(f"高斯噪声生成时间 (批量{batch_size}): {gaussian_time:.4f}s")
        logger.info(f"SO(3)噪声生成时间 (批量{batch_size}): {so3_time:.4f}s")
        logger.info(f"均匀噪声生成时间 (批量{batch_size}): {uniform_time:.4f}s")
        
        # 基本性能断言
        assert gaussian_time < 1.0  # 应该在1秒内完成
        assert so3_time < 2.0       # SO(3)可以稍慢
        assert uniform_time < 1.0   # 均匀分布应该很快


class TestNoiseGeneratorComparison:
    """噪声生成器比较测试."""

    def test_gaussian_vs_uniform_distribution(self):
        """测试高斯与均匀分布的差异."""
        gaussian = GaussianNoiseGenerator(std=1.0, mean=0.0)
        uniform = UniformNoiseGenerator(min_val=-3.0, max_val=3.0)
        
        # 生成大样本
        target = torch.zeros(10000, 1)
        gaussian_noise = gaussian.sample_like(target)
        uniform_noise = uniform.sample_like(target)
        
        # 验证分布特性
        gaussian_std = gaussian_noise.std().item()
        uniform_std = uniform_noise.std().item()
        
        # 高斯分布的标准差应该接近1.0
        assert abs(gaussian_std - 1.0) < 0.1
        
        # 均匀分布在[-3,3]的标准差应该接近sqrt(12)/2 = sqrt(3) ≈ 1.732
        expected_uniform_std = math.sqrt(3)
        assert abs(uniform_std - expected_uniform_std) < 0.2

    def test_so3_quaternion_properties(self):
        """测试SO(3)四元数的特殊性质."""
        so3_gen = SO3NoiseGenerator()
        gaussian_gen = GaussianNoiseGenerator()
        
        # 生成SO(3)噪声
        target_so3 = torch.zeros(100, 4)
        so3_noise = so3_gen.sample_like(target_so3)
        
        # 生成普通高斯噪声并归一化
        gaussian_noise = gaussian_gen.sample_like(target_so3)
        gaussian_normalized = gaussian_noise / torch.norm(gaussian_noise, dim=-1, keepdim=True)
        
        # 验证SO(3)噪声确实是单位四元数
        so3_norms = torch.norm(so3_noise, dim=-1)
        assert torch.allclose(so3_norms, torch.ones_like(so3_norms), atol=1e-6)
        
        # 验证归一化高斯噪声也是单位的，但分布不同
        gaussian_norms = torch.norm(gaussian_normalized, dim=-1)
        assert torch.allclose(gaussian_norms, torch.ones_like(gaussian_norms), atol=1e-6)
        
        # SO(3)噪声应该更均匀地分布在旋转群上
        # 这里我们只做基本验证，真正的均匀性测试需要更复杂的统计方法 
"""vector_field模块的单元测试

测试AllFlow中各种几何空间的速度场计算功能。
"""

import pytest
import torch
import numpy as np

from allflow.core.vector_field import (
    VectorField,
    EuclideanVectorField, 
    SO3VectorField,
    create_vector_field
)


class TestEuclideanVectorField:
    """测试欧几里得空间速度场."""
    
    def test_initialization(self):
        """测试初始化."""
        vf = EuclideanVectorField()
        assert isinstance(vf, VectorField)
        assert isinstance(vf, EuclideanVectorField)
    
    def test_set_endpoints_basic(self):
        """测试基本的端点设置."""
        vf = EuclideanVectorField()
        
        x_0 = torch.randn(4, 3)
        x_1 = torch.randn(4, 3)
        
        vf.set_endpoints(x_0, x_1)
        
        assert torch.equal(vf._x_0, x_0)
        assert torch.equal(vf._x_1, x_1)
    
    def test_set_endpoints_shape_mismatch(self):
        """测试端点形状不匹配的错误处理."""
        vf = EuclideanVectorField()
        
        x_0 = torch.randn(4, 3)
        x_1 = torch.randn(4, 5)  # 不同形状
        
        with pytest.raises(ValueError, match="x_0和x_1形状不匹配"):
            vf.set_endpoints(x_0, x_1)
    
    def test_call_without_endpoints(self):
        """测试在未设置端点时调用的错误处理."""
        vf = EuclideanVectorField()
        
        x = torch.randn(4, 3)
        t = torch.randn(4)
        
        with pytest.raises(RuntimeError, match="必须先调用set_endpoints方法"):
            vf(x, t)
    
    def test_call_with_endpoints(self):
        """测试正常的速度场计算."""
        vf = EuclideanVectorField()
        
        x_0 = torch.randn(4, 3)
        x_1 = torch.randn(4, 3)
        x_t = torch.randn(4, 3)
        t = torch.randn(4)
        
        vf.set_endpoints(x_0, x_1)
        velocity = vf(x_t, t)
        
        # 欧几里得速度场应该是 x_1 - x_0
        expected = x_1 - x_0
        assert torch.allclose(velocity, expected)
    
    def test_different_data_shapes(self):
        """测试不同数据形状的处理."""
        vf = EuclideanVectorField()
        
        # 2D数据
        x_0_2d = torch.randn(3, 28, 28)
        x_1_2d = torch.randn(3, 28, 28)
        x_t_2d = torch.randn(3, 28, 28)
        t_2d = torch.randn(3)
        
        vf.set_endpoints(x_0_2d, x_1_2d)
        velocity_2d = vf(x_t_2d, t_2d)
        
        assert velocity_2d.shape == x_0_2d.shape
        assert torch.allclose(velocity_2d, x_1_2d - x_0_2d)
        
        # 高维数据
        x_0_hd = torch.randn(2, 10, 15, 20)
        x_1_hd = torch.randn(2, 10, 15, 20)
        x_t_hd = torch.randn(2, 10, 15, 20)
        t_hd = torch.randn(2)
        
        vf.set_endpoints(x_0_hd, x_1_hd)
        velocity_hd = vf(x_t_hd, t_hd)
        
        assert velocity_hd.shape == x_0_hd.shape
        assert torch.allclose(velocity_hd, x_1_hd - x_0_hd)
    
    def test_device_compatibility(self):
        """测试设备兼容性."""
        vf = EuclideanVectorField()
        
        # 测试当前可用设备
        device = torch.device('mps' if torch.backends.mps.is_available() 
                             else 'cuda' if torch.cuda.is_available() 
                             else 'cpu')
        
        x_0 = torch.randn(4, 3, device=device)
        x_1 = torch.randn(4, 3, device=device)
        x_t = torch.randn(4, 3, device=device)
        t = torch.randn(4, device=device)
        
        vf.set_endpoints(x_0, x_1)
        velocity = vf(x_t, t)
        
        assert velocity.device.type == device.type
        assert torch.allclose(velocity, x_1 - x_0)


class TestSO3VectorField:
    """测试SO(3)旋转群速度场."""
    
    def test_initialization(self):
        """测试初始化."""
        vf = SO3VectorField()
        assert isinstance(vf, VectorField)
        assert isinstance(vf, SO3VectorField)
    
    def test_quaternion_conjugate(self):
        """测试四元数共轭."""
        vf = SO3VectorField()
        
        q = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]])
        q_conj = vf.quaternion_conjugate(q)
        
        expected = torch.tensor([[1.0, -2.0, -3.0, -4.0], [0.5, -0.5, -0.5, -0.5]])
        assert torch.allclose(q_conj, expected)
    
    def test_quaternion_multiply(self):
        """测试四元数乘法."""
        vf = SO3VectorField()
        
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # 恒等四元数
        q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # 绕x轴90度
        
        result = vf.quaternion_multiply(q1, q2)
        expected = q2  # 恒等四元数乘法
        
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_quaternion_log_identity(self):
        """测试恒等四元数的对数映射."""
        vf = SO3VectorField()
        
        q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        log_q = vf.quaternion_log(q_identity)
        
        # 恒等四元数的对数应该是零向量
        expected = torch.zeros(2, 3)
        assert torch.allclose(log_q, expected, atol=1e-6)
    
    def test_quaternion_log_small_angle(self):
        """测试小角度四元数的对数映射."""
        vf = SO3VectorField()
        
        # 绕z轴旋转小角度（10度）
        angle = torch.tensor(10.0) * torch.pi / 180.0
        q = torch.tensor([[torch.cos(angle/2), 0.0, 0.0, torch.sin(angle/2)]])
        
        log_q = vf.quaternion_log(q)
        
        # 对数映射应该恢复轴角表示（注意：这是半角）
        expected_axis = torch.tensor([[0.0, 0.0, 1.0]]) * (angle.item() / 2.0)
        assert torch.allclose(log_q, expected_axis, atol=1e-5)
    
    def test_compute_angular_velocity(self):
        """测试角速度计算."""
        vf = SO3VectorField()
        
        # 单位四元数
        q_0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q_1 = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]])  # 绕z轴90度
        
        omega = vf.compute_angular_velocity(q_0, q_1)
        
        # 角速度应该是绕z轴
        assert omega.shape == (1, 3)
        assert abs(omega[0, 2].item()) > 0  # z分量应该非零
        assert torch.allclose(omega[0, :2], torch.zeros(2), atol=1e-6)  # x,y分量应该为0
    
    def test_set_endpoints(self):
        """测试SO(3)端点设置."""
        vf = SO3VectorField()
        
        q_0 = torch.randn(4, 4)
        q_1 = torch.randn(4, 4)
        
        vf.set_endpoints(q_0, q_1)
        
        # SO3VectorField会归一化输入的四元数
        q_0_normalized = q_0 / torch.clamp(torch.norm(q_0, dim=-1, keepdim=True), min=vf.eps)
        q_1_normalized = q_1 / torch.clamp(torch.norm(q_1, dim=-1, keepdim=True), min=vf.eps)
        
        assert torch.allclose(vf._q_0, q_0_normalized, atol=1e-6)
        assert torch.allclose(vf._q_1, q_1_normalized, atol=1e-6)
    
    def test_set_endpoints_shape_mismatch(self):
        """测试四元数形状不匹配的错误处理."""
        vf = SO3VectorField()
        
        q_0 = torch.randn(4, 4)
        q_1 = torch.randn(3, 4)  # 不同批量大小
        
        with pytest.raises(ValueError, match="q_0和q_1形状不匹配"):
            vf.set_endpoints(q_0, q_1)
    
    def test_set_endpoints_wrong_dimension(self):
        """测试四元数维度错误的处理."""
        vf = SO3VectorField()
        
        q_0 = torch.randn(4, 3)  # 错误的四元数维度
        q_1 = torch.randn(4, 3)
        
        with pytest.raises(ValueError, match="SO\\(3\\)速度场需要四元数输入"):
            vf.set_endpoints(q_0, q_1)
    
    def test_call_without_endpoints(self):
        """测试在未设置端点时调用的错误处理."""
        vf = SO3VectorField()
        
        x = torch.randn(4, 4)
        t = torch.randn(4)
        
        with pytest.raises(RuntimeError, match="必须先调用set_endpoints方法"):
            vf(x, t)
    
    def test_call_with_endpoints(self):
        """测试正常的SO(3)速度场计算."""
        vf = SO3VectorField()
        
        # 归一化的四元数
        q_0 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        q_1 = torch.tensor([[0.7071, 0.0, 0.0, 0.7071], [0.7071, 0.7071, 0.0, 0.0]])
        q_t = torch.randn(2, 4)
        t = torch.randn(2)
        
        vf.set_endpoints(q_0, q_1)
        omega = vf(q_t, t)
        
        # 角速度应该是3维向量
        assert omega.shape == (2, 3)
        assert torch.isfinite(omega).all()
    
    def test_numerical_stability(self):
        """测试数值稳定性."""
        vf = SO3VectorField()
        
        # 测试接近恒等的四元数
        eps = 1e-7
        q_0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q_1 = torch.tensor([[1.0-eps, eps, 0.0, 0.0]])
        
        vf.set_endpoints(q_0, q_1)
        q_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        t = torch.tensor([0.5])
        
        omega = vf(q_t, t)
        
        # 应该产生有限的角速度
        assert torch.isfinite(omega).all()
        assert not torch.isnan(omega).any()


class TestVectorFieldFactory:
    """测试速度场工厂函数."""
    
    def test_create_euclidean_vector_field(self):
        """测试创建欧几里得速度场."""
        vf = create_vector_field("euclidean")
        assert isinstance(vf, EuclideanVectorField)
        
        vf_upper = create_vector_field("EUCLIDEAN")
        assert isinstance(vf_upper, EuclideanVectorField)
    
    def test_create_so3_vector_field(self):
        """测试创建SO(3)速度场."""
        vf = create_vector_field("so3")
        assert isinstance(vf, SO3VectorField)
        
        vf_upper = create_vector_field("SO3")
        assert isinstance(vf_upper, SO3VectorField)
    
    def test_invalid_vector_field_type(self):
        """测试无效的速度场类型."""
        with pytest.raises(ValueError, match="不支持的速度场类型"):
            create_vector_field("invalid_type")
    
    def test_case_insensitive_creation(self):
        """测试大小写不敏感的创建."""
        vf1 = create_vector_field("euclidean")
        vf2 = create_vector_field("Euclidean")
        vf3 = create_vector_field("EUCLIDEAN")
        
        assert type(vf1) == type(vf2) == type(vf3)


class TestVectorFieldPerformance:
    """测试速度场性能."""
    
    @pytest.mark.benchmark
    def test_euclidean_performance(self, benchmark):
        """测试欧几里得速度场的性能."""
        vf = EuclideanVectorField()
        
        x_0 = torch.randn(1000, 128)
        x_1 = torch.randn(1000, 128)
        x_t = torch.randn(1000, 128)
        t = torch.randn(1000)
        
        vf.set_endpoints(x_0, x_1)
        
        result = benchmark(vf, x_t, t)
        assert result.shape == (1000, 128)
    
    @pytest.mark.benchmark
    def test_so3_performance(self, benchmark):
        """测试SO(3)速度场的性能."""
        vf = SO3VectorField()
        
        q_0 = torch.randn(1000, 4)
        q_1 = torch.randn(1000, 4)
        q_t = torch.randn(1000, 4)
        t = torch.randn(1000)
        
        # 归一化四元数
        q_0 = q_0 / torch.norm(q_0, dim=-1, keepdim=True)
        q_1 = q_1 / torch.norm(q_1, dim=-1, keepdim=True)
        q_t = q_t / torch.norm(q_t, dim=-1, keepdim=True)
        
        vf.set_endpoints(q_0, q_1)
        
        result = benchmark(vf, q_t, t)
        assert result.shape == (1000, 3)


class TestVectorFieldComparison:
    """测试不同速度场的对比."""
    
    def test_euclidean_vs_so3_interface(self):
        """测试欧几里得和SO(3)速度场的接口一致性."""
        euclidean_vf = EuclideanVectorField()
        so3_vf = SO3VectorField()
        
        # 两者都应该有相同的抽象方法
        assert hasattr(euclidean_vf, '__call__')
        assert hasattr(euclidean_vf, 'set_endpoints')
        assert hasattr(so3_vf, '__call__')
        assert hasattr(so3_vf, 'set_endpoints')
        
        # 两者都应该继承自VectorField
        assert isinstance(euclidean_vf, VectorField)
        assert isinstance(so3_vf, VectorField) 
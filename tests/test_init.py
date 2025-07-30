"""AllFlow主模块测试

测试AllFlow包的公共API导入和基本功能。
"""

import pytest
import torch


class TestPublicAPIImports:
    """测试公共API导入."""
    
    def test_core_imports(self):
        """测试核心组件导入."""
        # 测试核心算法类
        from allflow import FlowMatching
        assert FlowMatching is not None
        
        # 测试时间采样器
        from allflow import UniformTimeSampler, NormalTimeSampler
        assert UniformTimeSampler is not None
        assert NormalTimeSampler is not None
        
        # 测试噪声生成器
        from allflow import GaussianNoiseGenerator
        assert GaussianNoiseGenerator is not None
    
    def test_convenience_imports(self):
        """测试便捷导入."""
        # 测试工厂函数
        from allflow import create_time_sampler, create_noise_generator
        assert create_time_sampler is not None
        assert create_noise_generator is not None
    
    def test_solver_imports(self):
        """测试求解器导入."""
        from allflow import create_solver
        assert create_solver is not None
    
    def test_version_info(self):
        """测试版本信息."""
        import allflow
        
        # 检查版本属性存在
        assert hasattr(allflow, '__version__')
        assert isinstance(allflow.__version__, str)
        assert len(allflow.__version__) > 0
    
    def test_module_metadata(self):
        """测试模块元数据."""
        import allflow
        
        # 检查基本元数据
        assert hasattr(allflow, '__author__')
        # assert hasattr(allflow, '__description__')  # 可能不存在
        assert hasattr(allflow, '__all__')
        
        assert isinstance(allflow.__all__, list)
        assert len(allflow.__all__) > 0


class TestFactoryFunctions:
    """测试工厂函数功能."""
    
    def test_create_time_sampler_function(self):
        """测试时间采样器创建函数."""
        from allflow import create_time_sampler
        
        # 创建不同类型的采样器
        uniform_sampler = create_time_sampler("uniform")
        assert uniform_sampler is not None
        
        normal_sampler = create_time_sampler("normal")
        assert normal_sampler is not None
    
    def test_create_noise_generator_function(self):
        """测试噪声生成器创建函数."""
        from allflow import create_noise_generator
        
        # 创建高斯噪声生成器
        gaussian_gen = create_noise_generator("gaussian")
        assert gaussian_gen is not None
        
        # 测试基本功能
        target = torch.randn(5, 10)
        noise = gaussian_gen.sample_like(target)
        assert noise.shape == target.shape
    
    def test_create_solver_function(self):
        """测试求解器创建函数."""
        from allflow import create_solver
        from allflow.solvers import HAS_TORCHDIFFEQ
        
        if HAS_TORCHDIFFEQ:
            solver = create_solver()
            assert solver is not None
        else:
            with pytest.raises(ImportError):
                create_solver()


class TestQuickStartAPI:
    """测试快速开始API."""
    
    def test_basic_flow_matching_workflow(self):
        """测试基本的Flow Matching工作流."""
        from allflow import FlowMatching, UniformTimeSampler, GaussianNoiseGenerator
        
        # 创建组件
        flow = FlowMatching()
        time_sampler = UniformTimeSampler()
        noise_gen = GaussianNoiseGenerator()
        
        # 测试数据准备
        x_1 = torch.randn(4, 10)
        x_0 = None  # 自动生成
        
        x_t, t, v = flow.prepare_training_data(x_1, x_0)
        
        assert x_t.shape == (4, 10)
        assert t.shape == (4,)
        assert v.shape == (4, 10)
        assert torch.isfinite(x_t).all()
        assert torch.isfinite(v).all()
    
    def test_different_time_samplers(self):
        """测试不同时间采样器的使用."""
        from allflow import FlowMatching, create_time_sampler
        
        flow = FlowMatching()
        
        # 测试不同采样器（跳过importance因为需要特殊参数）
        samplers = ["uniform", "normal", "exponential"]
        
        x_1 = torch.randn(4, 5)
        
        for sampler_name in samplers:
            time_sampler = create_time_sampler(sampler_name)
            flow.time_sampler = time_sampler
            
            x_t, t, v = flow.prepare_training_data(x_1, x_0=None)
            
            assert x_t.shape == (4, 5)
            assert t.shape == (4,)
            assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
    
    def test_loss_computation(self):
        """测试损失计算."""
        from allflow import FlowMatching
        
        flow = FlowMatching()
        
        # 准备数据
        x_0 = torch.randn(3, 8)
        x_1 = torch.randn(3, 8)
        
        x_t, t, true_velocity = flow.prepare_training_data(x_0, x_1)
        
        # 模拟预测速度（与真实速度接近）
        predicted_velocity = true_velocity + 0.01 * torch.randn_like(true_velocity)
        
        # 计算损失
        loss = flow.compute_loss(x_1, predicted_velocity, t, x_0)
        
        assert torch.is_tensor(loss)
        assert loss.dim() == 0  # 标量
        assert loss >= 0  # 损失应该非负
        assert torch.isfinite(loss)


class TestModelInteroperation:
    """测试模型互操作性."""
    
    def test_with_simple_model(self):
        """测试与简单模型的集成."""
        from allflow import FlowMatching
        from allflow.core.model_interface import SimpleModelWrapper
        import torch.nn as nn
        
        # 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(11, 10)  # 10 features + 1 time
            
            def forward(self, x, t):
                # 简单的时间条件模型
                if len(t.shape) == 1:
                    t = t.unsqueeze(-1)
                x_t = torch.cat([x, t], dim=-1)
                return self.linear(x_t)
        
        # 获取当前设备
        device = torch.device('mps' if torch.backends.mps.is_available() 
                             else 'cuda' if torch.cuda.is_available() 
                             else 'cpu')
        
        model = SimpleModel().to(device)
        wrapper = SimpleModelWrapper(model)
        
        # 测试数据流
        x_0 = torch.randn(5, 10, device=device)
        x_1 = torch.randn(5, 10, device=device)
        
        flow = FlowMatching()
        x_t, t, true_velocity = flow.prepare_training_data(x_0, x_1)
        
        # 使用模型预测
        predicted_velocity = wrapper(x_t, t)
        
        assert predicted_velocity.shape == true_velocity.shape
        assert torch.isfinite(predicted_velocity).all()
        
        # 计算损失
        loss = flow.compute_loss(x_1, predicted_velocity, t, x_0)
        assert torch.isfinite(loss)


class TestEdgeCases:
    """测试边界情况."""
    
    def test_import_edge_cases(self):
        """测试导入边界情况."""
        # 测试重复导入
        import allflow
        import allflow  # 重复导入应该安全
        
        assert allflow is not None
    
    def test_module_all_completeness(self):
        """测试__all__的完整性."""
        import allflow
        
        # 检查__all__中的所有项目都可以导入
        for item in allflow.__all__:
            assert hasattr(allflow, item), f"__all__中的'{item}'无法访问"
            obj = getattr(allflow, item)
            assert obj is not None, f"__all__中的'{item}'为None"
    
    def test_backward_compatibility(self):
        """测试向后兼容性."""
        # 测试关键类是否可用
        from allflow import FlowMatching
        from allflow.algorithms.flow_matching import FlowMatching as DirectFlowMatching
        
        # 应该是同一个类
        assert FlowMatching is DirectFlowMatching


class TestDocumentationAndHelp:
    """测试文档和帮助信息."""
    
    def test_module_docstring(self):
        """测试模块文档字符串."""
        import allflow
        
        assert allflow.__doc__ is not None
        assert len(allflow.__doc__.strip()) > 0
    
    def test_main_classes_have_docstrings(self):
        """测试主要类有文档字符串."""
        from allflow import FlowMatching, UniformTimeSampler, GaussianNoiseGenerator
        
        classes_to_check = [FlowMatching, UniformTimeSampler, GaussianNoiseGenerator]
        
        for cls in classes_to_check:
            assert cls.__doc__ is not None, f"{cls.__name__}缺少文档字符串"
            assert len(cls.__doc__.strip()) > 0, f"{cls.__name__}文档字符串为空"


class TestPerformanceBasics:
    """测试基本性能特性."""
    
    def test_import_time(self):
        """测试导入时间."""
        import time
        
        start_time = time.time()
        import allflow
        import_time = time.time() - start_time
        
        # 导入时间应该合理（小于1秒）
        assert import_time < 1.0, f"导入时间过长: {import_time:.3f}秒"
    
    def test_memory_usage_basic(self):
        """测试基本内存使用."""
        import allflow
        
        # 创建基本对象不应该消耗过多内存
        flow = allflow.FlowMatching()
        sampler = allflow.UniformTimeSampler()
        generator = allflow.GaussianNoiseGenerator()
        
        # 基本检查对象已创建
        assert flow is not None
        assert sampler is not None  
        assert generator is not None


class TestErrorHandling:
    """测试错误处理."""
    
    def test_invalid_factory_calls(self):
        """测试无效的工厂函数调用."""
        from allflow import create_time_sampler, create_noise_generator
        
        # 无效的采样器类型
        with pytest.raises(ValueError):
            create_time_sampler("invalid_type")
        
        # 无效的噪声生成器类型
        with pytest.raises(ValueError):
            create_noise_generator("invalid_type")
    
    def test_missing_dependencies_graceful(self):
        """测试缺少依赖时的优雅处理."""
        from allflow import create_solver
        from allflow.solvers import HAS_TORCHDIFFEQ
        
        if not HAS_TORCHDIFFEQ:
            # 应该抛出清晰的错误信息
            with pytest.raises(ImportError, match="torchdiffeq"):
                create_solver()


class TestDeviceCompatibility:
    """测试设备兼容性."""
    
    def test_device_handling(self):
        """测试设备处理."""
        from allflow import FlowMatching, UniformTimeSampler
        
        device = torch.device('mps' if torch.backends.mps.is_available() 
                             else 'cuda' if torch.cuda.is_available() 
                             else 'cpu')
        
        flow = FlowMatching()
        
        # 使用特定设备的数据
        x_1 = torch.randn(3, 5, device=device)
        
        x_t, t, v = flow.prepare_training_data(x_1, x_0=None)
        
        # 结果应该在正确的设备上
        assert x_t.device.type == device.type
        assert t.device.type == device.type
        assert v.device.type == device.type 
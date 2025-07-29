"""Flow Matching端到端集成测试

本模块测试Flow Matching算法的完整工作流程，包括与UNet网络和MNIST数据集
的集成，验证训练和推理的端到端功能。

测试内容:
- 完整的训练循环
- 模型推理和采样
- 数据加载集成
- 收敛性验证
- 生成质量评估

Author: AllFlow Test Suite Contributors
"""

import logging
import sys
from pathlib import Path

import pytest  # type: ignore
import torch
import torch.optim as optim

# 将项目根目录下的 src 目录添加到 Python 解释器的搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from allflow.algorithms.flow_matching import FlowMatching
from tests.data.mnist_loader import MNISTTestSampler
from tests.models.unet import create_test_unet

logger = logging.getLogger(__name__)


class TestFlowMatchingIntegration:
    """Flow Matching端到端集成测试类."""

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
    def flow_setup(self, device):
        """设置Flow Matching系统."""
        # 创建Flow Matching算法
        flow = FlowMatching(device=device)

        # 创建UNet模型
        model = create_test_unet(image_size=28, channels=1)
        model = model.to(device)

        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 创建数据采样器
        sampler = MNISTTestSampler(device=device)

        return flow, model, optimizer, sampler

    def test_training_step(self, flow_setup, device):
        """测试单个训练步骤."""
        flow, model, optimizer, sampler = flow_setup

        # 获取训练数据
        batch_size = 8
        x_0, x_1 = sampler.sample_pair(batch_size)

        # 检查数据形状
        assert x_0.shape == (batch_size, 1, 28, 28)
        assert x_1.shape == (batch_size, 1, 28, 28)
        assert x_0.device == device
        assert x_1.device == device

        # 前向传播
        model.train()
        loss = flow.compute_loss(x_0, x_1, model=model)

        # 检查损失
        assert loss.dim() == 0  # 标量损失
        assert torch.isfinite(loss)
        assert loss >= 0

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 检查梯度
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert torch.isfinite(param.grad).all()

        assert has_grad, "模型应该有梯度"

        # 优化器步骤
        optimizer.step()

        logger.info(f"训练步骤完成，损失: {loss.item():.6f}")

    def test_mini_training_loop(self, flow_setup, device):
        """测试小规模训练循环."""
        flow, model, optimizer, sampler = flow_setup

        batch_size = 4
        num_steps = 10
        losses = []

        model.train()

        for step in range(num_steps):
            # 获取数据
            x_0, x_1 = sampler.sample_pair(batch_size)

            # 前向传播
            loss = flow.compute_loss(x_0, x_1, model=model)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            losses.append(loss.item())

            if step % 5 == 0:
                logger.info(f"步骤 {step}, 损失: {loss.item():.6f}")

        # 验证训练过程
        assert len(losses) == num_steps
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

        # 检查损失是否在合理范围内（不要求收敛，只要求稳定）
        final_loss = losses[-1]
        assert 0 <= final_loss <= 100, f"最终损失超出合理范围: {final_loss}"

        logger.info(f"小规模训练完成，最终损失: {final_loss:.6f}")

    def test_inference_sampling(self, flow_setup, device):
        """测试推理和采样功能."""
        flow, model, optimizer, sampler = flow_setup

        # 简单训练几步以获得非随机模型
        model.train()
        for _ in range(5):
            x_0, x_1 = sampler.sample_pair(4)
            loss = flow.compute_loss(x_0, x_1, model=model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 切换到评估模式
        model.eval()

        batch_size = 4

        with torch.no_grad():
            # 生成初始噪声
            x_init = torch.randn(batch_size, 1, 28, 28, device=device)

            # 使用Flow Matching生成样本
            generated = flow.generate_sample(
                x_init, model, num_steps=20, method="euler"
            )

            # 检查生成结果
            assert generated.shape == (batch_size, 1, 28, 28)
            assert generated.device == device
            assert torch.isfinite(generated).all()

            # 检查生成的图像值在合理范围内（MNIST归一化到[-1,1]）
            assert generated.min() >= -2.0  # 允许一些超出范围
            assert generated.max() <= 2.0

        logger.info("推理采样测试完成")

    def test_model_architecture_compatibility(self, device):
        """测试模型架构与Flow Matching的兼容性."""
        # 测试不同尺寸的模型
        sizes = [(16, 16), (28, 28), (32, 32)]

        for size in sizes:
            # 创建对应尺寸的UNet
            model = create_test_unet(image_size=size[0], channels=1)
            model = model.to(device)

            # 创建Flow Matching
            flow = FlowMatching(device=device)

            # 测试前向传播
            batch_size = 2
            x = torch.randn(batch_size, 1, size[0], size[1], device=device)
            t = torch.rand(batch_size, device=device)

            with torch.no_grad():
                output = model(x, t)

            # 检查输出形状
            assert output.shape == x.shape
            assert torch.isfinite(output).all()

        logger.info("模型架构兼容性测试完成")

    def test_gradient_flow(self, flow_setup, device):
        """测试梯度流动正确性."""
        flow, model, optimizer, sampler = flow_setup

        batch_size = 4
        x_0, x_1 = sampler.sample_pair(batch_size)

        # 计算损失
        loss = flow.compute_loss(x_0, x_1, model=model)

        # 反向传播
        loss.backward()

        # 检查所有参数都有梯度
        param_count = 0
        grad_count = 0

        for name, param in model.named_parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1

                # 检查梯度不是零（至少有一些参数应该有非零梯度）
                if torch.norm(param.grad) > 1e-8:
                    assert torch.isfinite(param.grad).all(), (
                        f"参数 {name} 的梯度包含非有限值"
                    )

        # 确保大部分参数都有梯度
        grad_ratio = grad_count / param_count
        assert grad_ratio > 0.5, f"只有 {grad_ratio:.2%} 的参数有梯度"

        logger.info(f"梯度流动测试完成，{grad_count}/{param_count} 参数有梯度")

    def test_numerical_consistency(self, flow_setup, device):
        """测试数值一致性."""
        flow, model, optimizer, sampler = flow_setup

        # 固定随机种子确保可重现性
        torch.manual_seed(42)

        batch_size = 4
        x_0, x_1 = sampler.sample_pair(batch_size)

        # 第一次计算
        model.train()
        loss1 = flow.compute_loss(x_0, x_1, model=model)

        # 重置模型状态
        torch.manual_seed(42)

        # 第二次计算（应该得到相同结果）
        loss2 = flow.compute_loss(x_0, x_1, model=model)

        # 验证一致性
        assert torch.allclose(loss1, loss2, atol=1e-6), (
            f"损失不一致: {loss1} vs {loss2}"
        )

        logger.info("数值一致性测试完成")

    def test_memory_efficiency(self, flow_setup, device):
        """测试内存效率."""
        flow, model, optimizer, sampler = flow_setup

        if device.type == "cuda":
            # 记录初始GPU内存
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)

            batch_size = 16
            num_steps = 10

            # 运行多个训练步骤
            for _ in range(num_steps):
                x_0, x_1 = sampler.sample_pair(batch_size)

                optimizer.zero_grad()
                loss = flow.compute_loss(x_0, x_1, model=model)
                loss.backward()
                optimizer.step()

                # 清理中间变量
                del x_0, x_1, loss

            # 强制垃圾收集
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated(device)

            # 内存增长应该在合理范围内
            memory_growth = final_memory - initial_memory
            assert memory_growth < 100 * 1024 * 1024, (
                f"内存增长过大: {memory_growth / 1024 / 1024:.2f} MB"
            )

            logger.info(
                f"内存效率测试完成，内存增长: {memory_growth / 1024 / 1024:.2f} MB"
            )
        else:
            logger.info("跳过内存效率测试（非CUDA设备）")

    def test_batch_size_scalability(self, flow_setup, device):
        """测试批量大小可扩展性."""
        flow, model, optimizer, sampler = flow_setup

        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            try:
                x_0, x_1 = sampler.sample_pair(batch_size)

                # 前向传播
                loss = flow.compute_loss(x_0, x_1, model=model)

                # 检查结果
                assert loss.dim() == 0
                assert torch.isfinite(loss)

                logger.info(f"批量大小 {batch_size} 测试通过，损失: {loss.item():.6f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"批量大小 {batch_size} 超出内存限制")
                    break
                else:
                    raise

        logger.info("批量大小可扩展性测试完成")


@pytest.mark.slow
class TestFlowMatchingConvergence:
    """Flow Matching收敛性测试（较慢的测试）."""

    def test_convergence_basic(self, device="cpu"):
        """测试基础收敛性（使用CPU以确保稳定性）."""
        device = torch.device(device)

        # 创建系统组件
        flow = FlowMatching(device=device)
        model = create_test_unet(image_size=16, channels=1)  # 使用较小尺寸加速
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        sampler = MNISTTestSampler(target_size=(16, 16), device=device)

        # 训练参数
        batch_size = 8
        num_steps = 50
        losses = []

        model.train()

        for step in range(num_steps):
            x_0, x_1 = sampler.sample_pair(batch_size)

            optimizer.zero_grad()
            loss = flow.compute_loss(x_0, x_1, model=model)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            losses.append(loss.item())

            if step % 10 == 0:
                logger.info(f"收敛测试步骤 {step}, 损失: {loss.item():.6f}")

        # 分析收敛性
        initial_loss = sum(losses[:10]) / 10  # 前10步平均
        final_loss = sum(losses[-10:]) / 10  # 后10步平均

        # 验证损失下降或至少保持稳定
        assert final_loss <= initial_loss * 1.1, (
            f"损失未收敛: {initial_loss:.6f} -> {final_loss:.6f}"
        )

        # 验证最终损失在合理范围内
        assert 0 <= final_loss <= 10, f"最终损失异常: {final_loss}"

        logger.info(f"收敛测试完成: {initial_loss:.6f} -> {final_loss:.6f}")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 运行测试
    pytest.main([__file__, "-v", "-s"])

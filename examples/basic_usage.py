#!/usr/bin/env python3
"""AllFlow基础使用示例

这个示例展示了如何使用AllFlow库进行Flow Matching算法的基本操作，
包括算法初始化、轨迹采样、速度场计算和基础训练流程。

运行方式:
    python examples/basic_usage.py

Author: AllFlow Contributors
"""

import sys
from pathlib import Path

# 将项目根目录下的 src 目录添加到 Python 解释器的搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 导入AllFlow
from allflow.algorithms.flow_matching import FlowMatching


def simple_model_example():
    """演示基础Flow Matching算法的使用."""
    print("🎯 基础Flow Matching示例")
    print("=" * 50)

    # 1. 创建Flow Matching实例
    flow = FlowMatching(device="cpu")
    print(f"✅ Flow Matching创建成功，设备: {flow.device}")

    # 2. 创建测试数据
    batch_size = 32
    dim = 64

    # 源分布：标准高斯分布
    x_0 = torch.randn(batch_size, dim)

    # 目标分布：偏移的高斯分布
    x_1 = torch.randn(batch_size, dim) * 0.5 + 2.0

    # 随机时间点
    t = torch.rand(batch_size)

    print("✅ 测试数据创建完成")
    print(f"   源分布形状: {x_0.shape}, 均值: {x_0.mean():.3f}")
    print(f"   目标分布形状: {x_1.shape}, 均值: {x_1.mean():.3f}")

    # 3. 轨迹采样
    x_t = flow.sample_trajectory(x_0, x_1, t)
    print(f"✅ 轨迹采样完成，输出形状: {x_t.shape}")

    # 4. 速度场计算
    velocity = flow.compute_vector_field(x_t, t, x_0=x_0, x_1=x_1)
    print(f"✅ 速度场计算完成，输出形状: {velocity.shape}")

    # 5. 验证数学正确性
    expected_velocity = x_1 - x_0
    error = torch.norm(velocity - expected_velocity)
    print(f"✅ 数学正确性验证: 误差 = {error:.2e}")

    return flow, x_0, x_1, t


def training_example():
    """演示Flow Matching的训练过程."""
    print("\n🚀 Flow Matching训练示例")
    print("=" * 50)

    # 1. 创建Flow Matching和简单神经网络
    flow = FlowMatching(device="cpu")

    # 简单的多层感知机作为速度场预测器
    class SimpleVelocityModel(nn.Module):
        def __init__(self, dim: int, time_embed_dim: int = 64):
            super().__init__()
            self.time_mlp = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            self.velocity_net = nn.Sequential(
                nn.Linear(dim + time_embed_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, dim),
            )

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # 时间嵌入
            t_embed = self.time_mlp(t.unsqueeze(-1))

            # 拼接位置和时间信息
            x_t_cat = torch.cat([x, t_embed], dim=-1)

            # 预测速度场
            return self.velocity_net(x_t_cat)

    dim = 32
    model = SimpleVelocityModel(dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"✅ 模型创建完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 2. 训练循环
    num_epochs = 50
    batch_size = 128
    losses = []

    print(f"🔄 开始训练 ({num_epochs} epochs)...")

    for epoch in range(num_epochs):
        # 生成训练数据
        x_0 = torch.randn(batch_size, dim)
        x_1 = torch.randn(batch_size, dim) * 0.8 + torch.tensor(
            [2.0, -1.0] + [0.0] * (dim - 2)
        )

        # 前向传播
        optimizer.zero_grad()
        x_t, t, true_velocity = flow.prepare_training_data(x_0, x_1)
        predicted_velocity = model(x_t, t)
        loss = flow.compute_loss(x_0, x_1, t, predicted_velocity)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"   Epoch {epoch:3d}: 损失 = {loss.item():.6f}")

    print(f"✅ 训练完成！最终损失: {losses[-1]:.6f}")

    # 3. 生成采样
    print("🎨 生成采样...")

    model.eval()
    with torch.no_grad():
        # 从标准高斯分布开始
        num_samples = 100
        x_init = torch.randn(num_samples, dim)

        # 使用训练好的模型生成样本
        generated = flow.generate_sample(x_init, model, num_steps=50, method="euler")

        print("✅ 生成完成")
        print(f"   初始分布均值: {x_init.mean(dim=0)[:2].numpy()}")
        print(f"   生成分布均值: {generated.mean(dim=0)[:2].numpy()}")
        print("   目标分布均值: [2.0, -1.0]")

    return model, losses


def visualization_example():
    """2D可视化示例."""
    print("\n📊 2D可视化示例")
    print("=" * 50)

    try:
        # 创建2D Flow Matching
        flow = FlowMatching(device="cpu")

        # 生成2D数据
        num_points = 500

        # 源分布：圆形分布
        theta = torch.rand(num_points) * 2 * np.pi
        r = torch.randn(num_points) * 0.3 + 1.0
        x_0 = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

        # 目标分布：心形分布
        t_heart = torch.rand(num_points) * 2 * np.pi
        x_heart = 16 * torch.sin(t_heart) ** 3
        y_heart = (
            13 * torch.cos(t_heart)
            - 5 * torch.cos(2 * t_heart)
            - 2 * torch.cos(3 * t_heart)
            - torch.cos(4 * t_heart)
        )
        x_1 = torch.stack([x_heart, y_heart], dim=1) * 0.1

        print(f"✅ 2D数据生成完成: {num_points} 个点")

        # 可视化轨迹
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        time_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for i, t_val in enumerate(time_points):
            ax = axes[i // 3, i % 3]

            t_tensor = torch.full((num_points,), t_val)
            x_t = flow.sample_trajectory(x_0, x_1, t_tensor)

            ax.scatter(x_t[:, 0].numpy(), x_t[:, 1].numpy(), alpha=0.6, s=20)
            ax.set_title(f"t = {t_val:.1f}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("flow_matching_trajectory.png", dpi=150, bbox_inches="tight")
        print("✅ 轨迹可视化保存到: flow_matching_trajectory.png")

    except ImportError:
        print("⚠️  matplotlib未安装，跳过可视化示例")


def main():
    """主函数."""
    print("🌟 AllFlow - Flow Matching算法演示")
    print("https://github.com/your-username/allflow")
    print()

    # 设置随机种子确保可重现性
    torch.manual_seed(42)

    try:
        # 基础示例
        simple_model_example()

        # 训练示例
        training_example()

        # 可视化示例
        visualization_example()

        print("\n🎉 所有示例运行完成！")
        print("\n📚 更多信息:")
        print("   - 文档: docs/")
        print("   - 测试: pytest tests/")
        print("   - 教程: notebooks/")

    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

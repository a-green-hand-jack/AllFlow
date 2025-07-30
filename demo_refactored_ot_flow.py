"""演示重构后的Optimal Transport Flow (OT-Flow)

展示独立最优传输类的优势和在不同几何空间上的应用。
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

# 导入重构后的AllFlow组件
from allflow.algorithms.flow_matching import FlowMatching
from allflow.algorithms.ot_flow import OptimalTransportFlow, create_ot_flow
from allflow.core.optimal_transport import (
    EuclideanOptimalTransport,
    SO3OptimalTransport,
    create_optimal_transport,
)


def create_euclidean_test_data(
    batch_size: int = 16, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建欧几里得空间测试数据"""
    # 源分布：标准高斯噪声
    x_0 = torch.randn(batch_size, 3, device=device)

    # 目标分布：偏移的高斯分布
    x_1 = torch.randn(batch_size, 3, device=device) * 0.5 + torch.tensor(
        [3.0, 1.0, -2.0]
    )

    return x_0, x_1


def create_so3_test_data(
    batch_size: int = 12, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建SO3空间测试数据（四元数）"""
    # 创建随机归一化四元数
    q_0 = F.normalize(torch.randn(batch_size, 4, device=device), dim=1)
    q_1 = F.normalize(torch.randn(batch_size, 4, device=device), dim=1)

    return q_0, q_1


def demo_independent_optimal_transport():
    """演示独立最优传输类的功能"""
    print("🔧 演示独立最优传输类")
    print("=" * 50)

    device = torch.device("cpu")
    batch_size = 8

    # 欧几里得空间数据
    x_0_euclidean, x_1_euclidean = create_euclidean_test_data(batch_size, device)

    # SO3空间数据
    q_0_so3, q_1_so3 = create_so3_test_data(batch_size, device)

    print(f"📊 测试数据: batch_size={batch_size}")
    print(f"   欧几里得数据形状: {x_0_euclidean.shape}")
    print(f"   SO3数据形状: {q_0_so3.shape}")

    # 1. 欧几里得最优传输
    print("\n📐 欧几里得最优传输:")
    ot_euclidean = EuclideanOptimalTransport(method="approx", device=device)

    plan_euclidean, cost_euclidean = ot_euclidean.compute_transport_plan(
        x_0_euclidean, x_1_euclidean, return_cost=True
    )

    print(f"   ✓ 传输计划形状: {plan_euclidean.shape}")
    print(f"   ✓ 传输成本: {cost_euclidean.item():.6f}")

    # 重排序质量
    _, x_1_reordered = ot_euclidean.reorder_by_transport_plan(
        x_0_euclidean, x_1_euclidean, plan_euclidean
    )
    original_dist = torch.mean(torch.norm(x_0_euclidean - x_1_euclidean, dim=1))
    reordered_dist = torch.mean(torch.norm(x_0_euclidean - x_1_reordered, dim=1))
    improvement = (original_dist - reordered_dist) / original_dist * 100

    print(f"   ✓ 原始平均距离: {original_dist:.4f}")
    print(f"   ✓ 重排序平均距离: {reordered_dist:.4f}")
    print(f"   ✓ 改进幅度: {improvement:.2f}%")

    # 2. SO3最优传输
    print("\n🔄 SO3最优传输:")

    metrics = ["geodesic", "chordal", "frobenius"]
    for metric in metrics:
        ot_so3 = SO3OptimalTransport(
            method="approx", distance_metric=metric, device=device
        )

        plan_so3, cost_so3 = ot_so3.compute_transport_plan(
            q_0_so3, q_1_so3, return_cost=True
        )

        print(f"   ✓ {metric:>10} 度量 - 成本: {cost_so3.item():.6f}")

    # 3. 工厂函数演示
    print("\n🏭 工厂函数演示:")

    ot_factory_euclidean = create_optimal_transport("euclidean", method="sinkhorn")
    print(f"   ✓ 欧几里得传输器: {type(ot_factory_euclidean).__name__}")

    ot_factory_so3 = create_optimal_transport("so3", distance_metric="chordal")
    print(
        f"   ✓ SO3传输器: {type(ot_factory_so3).__name__} (度量: {ot_factory_so3.distance_metric})"
    )


def demo_euclidean_ot_flow():
    """演示欧几里得空间OT-Flow"""
    print("\n📐 演示欧几里得空间OT-Flow")
    print("=" * 50)

    device = torch.device("cpu")
    batch_size = 12

    # 创建测试数据
    x_0, x_1 = create_euclidean_test_data(batch_size, device)

    # 创建标准Flow Matching和OT-Flow
    standard_flow = FlowMatching(device=device)
    ot_flow = OptimalTransportFlow(
        space_type="euclidean", ot_method="approx", device=device
    )

    print(f"🔬 比较标准FM与OT-Flow (batch_size={batch_size}):")

    # 设置相同随机种子
    torch.manual_seed(42)
    x_t_std, t_std, v_std = standard_flow.prepare_training_data(x_1, x_0)

    torch.manual_seed(42)
    x_t_ot, t_ot, v_ot = ot_flow.prepare_training_data(x_1, x_0, use_ot_reordering=True)

    # 比较速度场特性
    v_std_magnitude = torch.mean(torch.norm(v_std, dim=1)).item()
    v_ot_magnitude = torch.mean(torch.norm(v_ot, dim=1)).item()

    print(f"   标准FM速度场平均幅度: {v_std_magnitude:.4f}")
    print(f"   OT-Flow速度场平均幅度: {v_ot_magnitude:.4f}")
    print(
        f"   速度场改进: {(v_std_magnitude - v_ot_magnitude) / v_std_magnitude * 100:.2f}%"
    )

    # 测试OT损失
    predicted_velocity = torch.randn_like(x_0)
    t = torch.rand(batch_size, device=device)

    total_loss, fm_loss, ot_loss = ot_flow.compute_ot_loss(
        x_1, predicted_velocity, t, x_0, ot_weight=0.1
    )

    print("\n💰 损失分解:")
    print(f"   Flow Matching损失: {fm_loss.item():.6f}")
    print(f"   最优传输损失: {ot_loss.item():.6f}")
    print(f"   总损失: {total_loss.item():.6f}")
    print(f"   OT损失贡献: {(ot_loss.item() * 0.1 / total_loss.item() * 100):.2f}%")

    # 算法信息
    info = ot_flow.get_algorithm_info()
    print("\n📋 算法信息:")
    print(f"   空间类型: {info['space_type']}")
    print(f"   传输器类型: {info['optimal_transport']['transport_type']}")
    print(f"   传输方法: {info['optimal_transport']['method']}")


def demo_so3_ot_flow():
    """演示SO3空间OT-Flow"""
    print("\n🔄 演示SO3空间OT-Flow")
    print("=" * 50)

    device = torch.device("cpu")
    batch_size = 10

    # 创建SO3测试数据
    q_0, q_1 = create_so3_test_data(batch_size, device)

    print("📊 SO3数据特性:")
    print(f"   四元数形状: {q_0.shape}")
    print(
        f"   q_0归一化检查: {torch.allclose(torch.norm(q_0, dim=1), torch.ones(batch_size))}"
    )
    print(
        f"   q_1归一化检查: {torch.allclose(torch.norm(q_1, dim=1), torch.ones(batch_size))}"
    )

    # 测试不同距离度量的OT-Flow
    metrics = ["geodesic", "chordal", "frobenius"]

    for metric in metrics:
        print(f"\n🎯 测试 {metric} 度量:")

        ot_flow_so3 = OptimalTransportFlow(
            space_type="so3", ot_method="approx", distance_metric=metric, device=device
        )

        # 计算传输计划
        transport_plan, cost = ot_flow_so3.compute_optimal_transport_plan(
            q_0, q_1, return_cost=True
        )

        print(f"   ✓ 传输成本: {cost.item():.6f}")

        # 准备训练数据
        x_t, t, v = ot_flow_so3.prepare_training_data(q_1, q_0, use_ot_reordering=True)

        print(f"   ✓ 训练数据形状: x_t={x_t.shape}, t={t.shape}, v={v.shape}")

        # 速度场特性
        v_magnitude = torch.mean(torch.norm(v, dim=1)).item()
        print(f"   ✓ 速度场平均幅度: {v_magnitude:.4f}")

        # 算法信息
        info = ot_flow_so3.get_algorithm_info()
        print(f"   ✓ 距离度量: {info['optimal_transport']['distance_metric']}")


def demo_factory_functions():
    """演示工厂函数的便利性"""
    print("\n🏭 演示工厂函数")
    print("=" * 50)

    # 欧几里得OT-Flow
    flow_euclidean = create_ot_flow(
        space_type="euclidean", ot_method="sinkhorn", reg_param=0.05
    )
    print(f"✓ 欧几里得OT-Flow: {type(flow_euclidean.optimal_transport).__name__}")
    print(f"   方法: {flow_euclidean.optimal_transport.method}")
    print(f"   正则化参数: {flow_euclidean.optimal_transport.reg_param}")

    # SO3 OT-Flow
    flow_so3 = create_ot_flow(
        space_type="so3", ot_method="approx", distance_metric="chordal", reg_param=0.1
    )
    print(f"✓ SO3 OT-Flow: {type(flow_so3.optimal_transport).__name__}")
    print(f"   距离度量: {flow_so3.optimal_transport.distance_metric}")
    print(f"   方法: {flow_so3.optimal_transport.method}")

    # 自定义传输器注入
    custom_ot = EuclideanOptimalTransport(method="exact", reg_param=0.001)
    flow_custom = OptimalTransportFlow(optimal_transport=custom_ot)
    print(f"✓ 自定义传输器注入: 正则化={flow_custom.optimal_transport.reg_param}")


def demo_pairing_quality_comparison():
    """演示不同空间的配对质量比较"""
    print("\n⚖️ 配对质量比较")
    print("=" * 50)

    device = torch.device("cpu")
    batch_size = 16

    # 欧几里得空间对比
    print("📐 欧几里得空间配对质量:")
    x_0_euclidean, x_1_euclidean = create_euclidean_test_data(batch_size, device)

    # 随机配对 vs 最优传输配对
    random_perm = torch.randperm(batch_size)
    x_1_random = x_1_euclidean[random_perm]

    ot_euclidean = EuclideanOptimalTransport(method="approx", device=device)
    plan_euclidean = ot_euclidean.compute_transport_plan(x_0_euclidean, x_1_euclidean)
    _, x_1_ot = ot_euclidean.reorder_by_transport_plan(
        x_0_euclidean, x_1_euclidean, plan_euclidean
    )

    # 计算配对质量
    random_dist = torch.mean(torch.norm(x_0_euclidean - x_1_random, dim=1)).item()
    ot_dist = torch.mean(torch.norm(x_0_euclidean - x_1_ot, dim=1)).item()
    euclidean_improvement = (random_dist - ot_dist) / random_dist * 100

    print(f"   随机配对平均距离: {random_dist:.4f}")
    print(f"   最优传输配对距离: {ot_dist:.4f}")
    print(f"   改进幅度: {euclidean_improvement:.2f}%")

    # SO3空间对比
    print("\n🔄 SO3空间配对质量:")
    q_0_so3, q_1_so3 = create_so3_test_data(batch_size, device)

    # 随机配对 vs 最优传输配对
    q_1_random = q_1_so3[torch.randperm(batch_size)]

    ot_so3 = SO3OptimalTransport(
        method="approx", distance_metric="geodesic", device=device
    )
    plan_so3 = ot_so3.compute_transport_plan(q_0_so3, q_1_so3)
    _, q_1_ot = ot_so3.reorder_by_transport_plan(q_0_so3, q_1_so3, plan_so3)

    # 使用四元数内积衡量相似性（值越大越相似）
    random_sim = torch.mean(torch.abs(torch.sum(q_0_so3 * q_1_random, dim=1))).item()
    ot_sim = torch.mean(torch.abs(torch.sum(q_0_so3 * q_1_ot, dim=1))).item()
    so3_improvement = (ot_sim - random_sim) / random_sim * 100

    print(f"   随机配对平均相似性: {random_sim:.4f}")
    print(f"   最优传输配对相似性: {ot_sim:.4f}")
    print(f"   改进幅度: {so3_improvement:.2f}%")


def main():
    """主演示函数"""
    print("🎯 重构后的Optimal Transport Flow (OT-Flow) 演示")
    print("=" * 60)
    print("展示独立最优传输类的优势和多几何空间支持")

    # 设置随机种子确保可重现
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 演示1: 独立最优传输类
        demo_independent_optimal_transport()

        # 演示2: 欧几里得空间OT-Flow
        demo_euclidean_ot_flow()

        # 演示3: SO3空间OT-Flow
        demo_so3_ot_flow()

        # 演示4: 工厂函数
        demo_factory_functions()

        # 演示5: 配对质量比较
        demo_pairing_quality_comparison()

        print("\n" + "=" * 60)
        print("🎉 演示完成！")
        print("\n🚀 重构后的OT-Flow主要优势:")
        print("  • 独立的最优传输类，模块化设计")
        print("  • 支持欧几里得空间和SO3旋转群")
        print("  • 多种距离度量和求解方法")
        print("  • 灵活的工厂函数和自定义注入")
        print("  • 显著改善的配对质量")
        print("  • 面向对象的可扩展架构")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

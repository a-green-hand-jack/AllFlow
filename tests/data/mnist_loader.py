"""MNIST数据加载器用于Flow Matching测试

本模块提供简化的MNIST数据集加载和预处理功能，专门为Flow Matching
算法的测试验证而设计。

主要功能:
- 自动下载和缓存MNIST数据集
- 标准化预处理和数据增强
- 批量数据加载器配置
- 测试友好的小批量采样

设计目标:
- 快速数据加载，减少测试等待时间
- 简化的预处理管道
- 灵活的批量大小配置
- 跨设备兼容性支持

Author: AllFlow Test Suite Contributors
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision  # type: ignore
import torchvision.transforms as transforms  # type: ignore
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MNISTFlowDataset(Dataset):
    """MNIST数据集包装器，用于Flow Matching训练.
    
    将标准MNIST数据集适配为Flow Matching所需的格式，包括
    数据归一化、形状调整和设备管理。
    """
    
    def __init__(
        self,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (28, 28),
        normalize: bool = True,
        data_dir: Optional[Union[str, Path]] = None
    ):
        """初始化MNIST数据集.
        
        Args:
            train: 是否使用训练集，False为测试集
            transform: 可选的数据变换
            target_size: 目标图像尺寸
            normalize: 是否进行归一化到[-1, 1]
            data_dir: 数据存储目录，None时使用默认位置
        """
        self.target_size = target_size
        self.normalize = normalize
        
        # 设置数据目录
        if data_dir is None:
            data_dir = Path.home() / '.cache' / 'allflow' / 'mnist'
        else:
            data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建基础变换
        base_transform = []
        
        # 调整图像尺寸（如果需要）
        if target_size != (28, 28):
            base_transform.append(transforms.Resize(target_size))
        
        # 转换为张量
        base_transform.append(transforms.ToTensor())
        
        # 归一化到[-1, 1]区间（Flow Matching的标准范围）
        if normalize:
            base_transform.append(transforms.Normalize((0.5,), (0.5,)))
        
        # 组合变换
        if transform is not None:
            self.transform = transforms.Compose(base_transform + [transform])
        else:
            self.transform = transforms.Compose(base_transform)
        
        # 加载MNIST数据集
        try:
            self.dataset = torchvision.datasets.MNIST(
                root=str(data_dir),
                train=train,
                download=True,
                transform=self.transform
            )
            logger.info(f"MNIST数据集加载成功: {'训练集' if train else '测试集'}, 样本数: {len(self.dataset)}")
            
        except Exception as e:
            logger.error(f"MNIST数据集加载失败: {e}")
            raise RuntimeError(f"无法加载MNIST数据集: {e}")
    
    def __len__(self) -> int:
        """返回数据集大小."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """获取数据样本.
        
        Args:
            idx: 样本索引
            
        Returns:
            图像张量, shape: (1, H, W), 范围 [-1, 1] 或 [0, 1]
        """
        image, _ = self.dataset[idx]  # 忽略标签，Flow Matching是无监督的
        return image
    
    def get_sample_batch(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """获取一个样本批次，用于快速测试.
        
        Args:
            batch_size: 批量大小
            device: 目标设备
            
        Returns:
            样本批次, shape: (batch_size, 1, H, W)
        """
        indices = torch.randperm(len(self))[:batch_size].tolist()
        batch = torch.stack([self[i] for i in indices])
        
        if device is not None:
            batch = batch.to(device)
        
        return batch


def create_mnist_loaders(
    batch_size: int = 64,
    train_split: float = 0.8,
    num_workers: int = 0,
    pin_memory: bool = True,
    target_size: Tuple[int, int] = (28, 28),
    data_dir: Optional[Union[str, Path]] = None
) -> Tuple[DataLoader, DataLoader]:
    """创建MNIST训练和验证数据加载器.
    
    Args:
        batch_size: 批量大小
        train_split: 训练集比例（从训练数据中划分）
        num_workers: 数据加载进程数
        pin_memory: 是否启用内存固定
        target_size: 目标图像尺寸
        data_dir: 数据存储目录
        
    Returns:
        (train_loader, val_loader): 训练和验证数据加载器
    """
    # 创建完整的训练数据集
    full_train_dataset = MNISTFlowDataset(
        train=True,
        target_size=target_size,
        data_dir=data_dir
    )
    
    # 划分训练和验证集
    total_size = len(full_train_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_indices = torch.randperm(total_size)[:train_size].tolist()
    val_indices = torch.randperm(total_size)[train_size:train_size + val_size].tolist()
    
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 确保批量大小一致
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"数据加载器创建完成: 训练={len(train_subset)}, 验证={len(val_subset)}")
    
    return train_loader, val_loader


def create_test_loader(
    batch_size: int = 32,
    num_workers: int = 0,
    target_size: Tuple[int, int] = (28, 28),
    data_dir: Optional[Union[str, Path]] = None
) -> DataLoader:
    """创建MNIST测试数据加载器.
    
    Args:
        batch_size: 批量大小
        num_workers: 数据加载进程数
        target_size: 目标图像尺寸
        data_dir: 数据存储目录
        
    Returns:
        测试数据加载器
    """
    test_dataset = MNISTFlowDataset(
        train=False,
        target_size=target_size,
        data_dir=data_dir
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"测试数据加载器创建完成: 样本数={len(test_dataset)}")
    
    return test_loader


def generate_noise_batch(
    batch_size: int,
    image_shape: Tuple[int, int, int] = (1, 28, 28),
    device: Optional[torch.device] = None,
    noise_type: str = "gaussian"
) -> torch.Tensor:
    """生成噪声批次，用作Flow Matching的源分布.
    
    Args:
        batch_size: 批量大小
        image_shape: 图像形状 (channels, height, width)
        device: 目标设备
        noise_type: 噪声类型，'gaussian' 或 'uniform'
        
    Returns:
        噪声批次, shape: (batch_size, *image_shape)
    """
    shape = (batch_size,) + image_shape
    
    if noise_type == "gaussian":
        # 标准高斯噪声
        noise = torch.randn(shape)
    elif noise_type == "uniform":
        # 均匀分布噪声，范围[-1, 1]
        noise = torch.rand(shape) * 2 - 1
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")
    
    if device is not None:
        noise = noise.to(device)
    
    return noise


class MNISTTestSampler:
    """MNIST测试采样器，用于快速获取测试样本."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (28, 28),
        device: Optional[torch.device] = None,
        data_dir: Optional[Union[str, Path]] = None
    ):
        """初始化测试采样器.
        
        Args:
            target_size: 目标图像尺寸
            device: 计算设备
            data_dir: 数据存储目录
        """
        self.device = device
        self.dataset = MNISTFlowDataset(
            train=False,
            target_size=target_size,
            data_dir=data_dir
        )
    
    def sample_pair(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样源-目标数据对，用于Flow Matching训练.
        
        Args:
            batch_size: 批量大小
            
        Returns:
            (x_0, x_1): 源分布和目标分布的样本对
        """
        # 目标分布：真实MNIST图像
        x_1 = self.dataset.get_sample_batch(batch_size, self.device)
        
        # 源分布：高斯噪声
        shape = tuple(x_1.shape[1:])  # 转换为tuple
        x_0 = generate_noise_batch(
            batch_size, 
            shape, 
            self.device, 
            noise_type="gaussian"
        )
        
        return x_0, x_1
    
    def sample_single(self, batch_size: int, distribution: str = "target") -> torch.Tensor:
        """采样单一分布的批次.
        
        Args:
            batch_size: 批量大小
            distribution: 分布类型，'source' 或 'target'
            
        Returns:
            样本批次
        """
        if distribution == "target":
            return self.dataset.get_sample_batch(batch_size, self.device)
        elif distribution == "source":
            sample_shape = (1,) + self.dataset.target_size
            return generate_noise_batch(batch_size, sample_shape, self.device)
        else:
            raise ValueError(f"不支持的分布类型: {distribution}")


# 便捷函数
def quick_mnist_setup(
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> Tuple[MNISTTestSampler, DataLoader]:
    """快速设置MNIST测试环境.
    
    Args:
        batch_size: 批量大小
        device: 计算设备
        
    Returns:
        (sampler, test_loader): 测试采样器和数据加载器
    """
    sampler = MNISTTestSampler(device=device)
    test_loader = create_test_loader(batch_size=batch_size)
    
    return sampler, test_loader 
"""Data loading for CIFAR-10: transforms, datasets, and DataLoaders.

The train recipe is shared by all models (random flip + random crop + normalization).
EfficientNet-V2-S additionally trains with RandAugment to counter overfitting, so it
gets its own augmented training loader; the CNN and WRN keep the plain recipe.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

__all__ = ["build_train_transform", "build_test_transform", "build_cifar10_loaders"]


def build_train_transform(mean, std, randaugment=False):
    """Standard train transform; prepend RandAugment for the transfer-learned model."""
    ops = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
    ]
    if randaugment:
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(ops)


def build_test_transform(mean, std):
    """Test transform: normalization only, so metrics reflect real-world performance."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_cifar10_loaders(data_dir, mean, std, batch_size=128, num_workers=2):
    """Return ``(trainloader, testloader, trainloader_effnet)`` for CIFAR-10.

    All loaders read from ``data_dir`` (already downloaded by Part 1). Pinned memory is
    enabled on CUDA so GPU transfers never stall the training loop.
    """
    pin_memory = torch.cuda.is_available()

    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=False,
        transform=build_train_transform(mean, std),
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=False,
        transform=build_test_transform(mean, std),
    )
    trainset_effnet = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=False,
        transform=build_train_transform(mean, std, randaugment=True),
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=pin_memory)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    trainloader_effnet = DataLoader(trainset_effnet, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_memory)
    return trainloader, testloader, trainloader_effnet

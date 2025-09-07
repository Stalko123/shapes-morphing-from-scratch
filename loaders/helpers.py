from typing import Tuple, Callable, Dict, Union
import inspect
import torch
from torch.utils.data import random_split, Subset
from torchvision import datasets as dsets

from torchvision import transforms as T


def _first_batch_shape(dataset) -> Tuple[int, int, int]:
    x, _ = dataset[0]
    return tuple(x.shape)  # (C,H,W)


_MNIST_LIKE = {"MNIST", "FashionMNIST", "EMNIST", "QMNIST"}
_RGB_DEFAULT = lambda name: name not in _MNIST_LIKE

def _norm(rgb: bool):
    return T.Normalize((0.5,)* (3 if rgb else 1), (0.5,)* (3 if rgb else 1))

def transform_factory(
    dataset_name: str,
    img_size: int | None = None,            # if None: keep native size
    normalize_to_neg1: bool = True,
    rgb: bool | None = None,
    apply_augmentation: bool = False,
):
    """
    Returns a torchvision transform.Compose for the given dataset.
    Normalizes to [-1,1] when normalize_to_neg1=True.
    """
    name = dataset_name
    rgb = _RGB_DEFAULT(name) if rgb is None else rgb

    aug: list[T.Compose | T.Transform] = []
    pre: list[T.Compose | T.Transform] = []
    post: list[T.Compose | T.Transform] = []

    # size control
    if img_size is not None:
        # Use center crop for eval; a slightly jittery crop for train
        if apply_augmentation:
            # dataset-specific resizing/cropping choices
            if name in {"CIFAR10", "CIFAR100"} and img_size == 32:
                # classic CIFAR recipe
                aug += [T.RandomCrop(32, padding=4)]
            elif name == "STL10":
                aug += [T.RandomResizedCrop(img_size, scale=(0.9, 1.0))]
            elif name == "SVHN":
                # conservative: tiny crop, no flips
                aug += [T.RandomCrop(32 if img_size is None else img_size, padding=2)]
            elif name in {"CelebA"}:
                # faces: maybe just small jitter; keep geometry mostly intact
                aug += [T.CenterCrop(img_size)]
            else:
                # generic natural images
                aug += [T.RandomResizedCrop(img_size, scale=(0.8, 1.0))]
        else:
            pre += [T.Resize(img_size), T.CenterCrop(img_size)]

    # geometric / photometric augs
    if apply_augmentation:
        if name in {"CIFAR10", "CIFAR100", "STL10", "Flowers102", "Food101", "ImageFolder"}:
            aug += [T.RandomHorizontalFlip(p=0.5)]
            # very mild color jitter for natural images
            aug += [T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)]
        elif name in _MNIST_LIKE:
            # digits: no flips; small affine
            aug += [T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))]
        elif name == "SVHN":
            # digits in the wild: avoid flips; tiny affine at most
            aug += [T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02))]
        elif name == "CelebA":
            # faces: flip is generally okay, avoid rotations
            aug += [T.RandomHorizontalFlip(p=0.5)]

    # tensor & normalization
    post += [T.ToTensor()]
    if normalize_to_neg1:
        post += [_norm(rgb)]

    return T.Compose(pre + aug + post)

# convenience: build train/val pair
def make_transforms_pair(
    dataset_name: str,
    img_size: int | None = None,
    normalize_to_neg1: bool = True,
    rgb: bool | None = None,
    apply_aug: bool = True
):
    """
    Returns (transform_train, transform_val) with augs on train only.
    """
    tfm_train = transform_factory(
        dataset_name,
        img_size=img_size,
        normalize_to_neg1=normalize_to_neg1,
        rgb=rgb,
        apply_augmentation=apply_aug,
    )
    tfm_val = transform_factory(
        dataset_name,
        img_size=img_size,
        normalize_to_neg1=normalize_to_neg1,
        rgb=rgb,
        apply_augmentation=False,
    )
    return tfm_train, tfm_val



# return type: a single Dataset or (train_ds, val_ds)
AdapterReturn = Union[torch.utils.data.Dataset, Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]]

def _rebuild_subset_with_transform(dataset_cls, indices, *, root, transform, download, **passthrough):
    """
    Rebuild the same dataset with a different transform and re-index it.
    Useful to give val a transform without augmentation.
    """
    ds = dataset_cls(root=root, transform=transform, download=download, **passthrough)
    return Subset(ds, indices)

# -------------------------
# CIFAR10 / CIFAR100
# -------------------------
def _adapter_cifar10(split, root, transform, download, val_ratio=0.1, seed=42, transform_val=None, **_)->AdapterReturn:
    if split == "test":
        return dsets.CIFAR10(root=root, train=False, transform=transform_val or transform, download=download)

    full_train = dsets.CIFAR10(root=root, train=True, transform=transform, download=download)
    if split == "train":
        return full_train

    # "train+val"
    n_total = len(full_train)
    n_val = int(n_total * val_ratio); n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=g)

    if transform_val is not None:
        val_subset  = _rebuild_subset_with_transform(dsets.CIFAR10,  val_subset.indices,
                                                     root=root, transform=transform_val, download=False, train=True)
        train_subset = _rebuild_subset_with_transform(dsets.CIFAR10, train_subset.indices,
                                                     root=root, transform=transform, download=False, train=True)
    return train_subset, val_subset


def _adapter_cifar100(split, root, transform, download, val_ratio=0.1, seed=42, transform_val=None, **_)->AdapterReturn:
    if split == "test":
        return dsets.CIFAR100(root=root, train=False, transform=transform_val or transform, download=download)

    full_train = dsets.CIFAR100(root=root, train=True, transform=transform, download=download)
    if split == "train":
        return full_train

    n_total = len(full_train)
    n_val = int(n_total * val_ratio); n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=g)

    if transform_val is not None:
        val_subset  = _rebuild_subset_with_transform(dsets.CIFAR100,  val_subset.indices,
                                                     root=root, transform=transform_val, download=False, train=True)
        train_subset = _rebuild_subset_with_transform(dsets.CIFAR100, train_subset.indices,
                                                     root=root, transform=transform,     download=False, train=True)
    return train_subset, val_subset

# -------------------------
# MNIST-like (MNIST/FashionMNIST/EMNIST/QMNIST)
# -------------------------
def _adapter_mnist_like(Cls):
    def fn(split, root, transform, download, val_ratio=0.1, seed=42, transform_val=None, **extra)->AdapterReturn:
        if split == "test":
            return Cls(root=root, train=False, transform=transform_val or transform, download=download, **extra)

        full_train = Cls(root=root, train=True, transform=transform, download=download, **extra)
        if split == "train":
            return full_train

        n_total = len(full_train)
        n_val = int(n_total * val_ratio); n_train = n_total - n_val
        g = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=g)

        if transform_val is not None:
            val_subset  = _rebuild_subset_with_transform(Cls,  val_subset.indices,
                                                         root=root, transform=transform_val, download=False, train=True, **extra)
            train_subset = _rebuild_subset_with_transform(Cls, train_subset.indices,
                                                         root=root, transform=transform,     download=False, train=True, **extra)
        return train_subset, val_subset
    return fn

# -------------------------
# SVHN (train/test/extra) – we split "train" for val
# -------------------------
def _adapter_svhn(split, root, transform, download, val_ratio=0.1, seed=42, transform_val=None, **_)->AdapterReturn:
    if split == "test":
        return dsets.SVHN(root=root, split="test", transform=transform_val or transform, download=download)

    full_train = dsets.SVHN(root=root, split="train", transform=transform, download=download)
    if split == "train":
        return full_train

    n_total = len(full_train)
    n_val = int(n_total * val_ratio); n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=g)

    if transform_val is not None:
        val_subset  = _rebuild_subset_with_transform(dsets.SVHN,  val_subset.indices,
                                                     root=root, transform=transform_val, download=False, split="train")
        train_subset = _rebuild_subset_with_transform(dsets.SVHN, train_subset.indices,
                                                     root=root, transform=transform,     download=False, split="train")
    return train_subset, val_subset

# -------------------------
# STL10 (train/test/unlabeled) – we split "train" for val
# -------------------------
def _adapter_stl10(split, root, transform, download, val_ratio=0.1, seed=42, transform_val=None, **_)->AdapterReturn:
    if split == "test":
        return dsets.STL10(root=root, split="test", transform=transform_val or transform, download=download)

    full_train = dsets.STL10(root=root, split="train", transform=transform, download=download)
    if split == "train":
        return full_train

    n_total = len(full_train)
    n_val = int(n_total * val_ratio); n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=g)

    if transform_val is not None:
        val_subset  = _rebuild_subset_with_transform(dsets.STL10,  val_subset.indices,
                                                     root=root, transform=transform_val, download=False, split="train")
        train_subset = _rebuild_subset_with_transform(dsets.STL10, train_subset.indices,
                                                     root=root, transform=transform,     download=False, split="train")
    return train_subset, val_subset

# -------------------------
# CelebA (train/valid/test) – native valid
# -------------------------
def _adapter_celeba(split, root, transform, download, transform_val=None, **_)->AdapterReturn:
    if split == "test":
        return dsets.CelebA(root=root, split="test",  transform=transform_val or transform, download=download)
    if split == "train":
        return dsets.CelebA(root=root, split="train", transform=transform,              download=download)
    # "train+val" -> native "valid"
    train_ds = dsets.CelebA(root=root, split="train", transform=transform,              download=download)
    val_ds   = dsets.CelebA(root=root, split="valid", transform=transform_val or transform, download=download)
    return train_ds, val_ds

# -------------------------
# Flowers102 (train/val/test) – native val
# -------------------------
def _adapter_flowers102(split, root, transform, download, transform_val=None, **_)->AdapterReturn:
    if split == "test":
        return dsets.Flowers102(root=root, split="test",  transform=transform_val or transform, download=download)
    if split == "train":
        return dsets.Flowers102(root=root, split="train", transform=transform,              download=download)
    # "train+val"
    train_ds = dsets.Flowers102(root=root, split="train", transform=transform,              download=download)
    val_ds   = dsets.Flowers102(root=root, split="val",   transform=transform_val or transform, download=download)
    return train_ds, val_ds

# -------------------------
# Food101 (train/test) – split train for val
# -------------------------
def _adapter_food101(split, root, transform, download, val_ratio=0.1, seed=42, transform_val=None, **_)->AdapterReturn:
    if split == "test":
        return dsets.Food101(root=root, split="test", transform=transform_val or transform, download=download)

    full_train = dsets.Food101(root=root, split="train", transform=transform, download=download)
    if split == "train":
        return full_train

    n_total = len(full_train)
    n_val = int(n_total * val_ratio); n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=g)

    if transform_val is not None:
        val_subset  = _rebuild_subset_with_transform(dsets.Food101,  val_subset.indices,
                                                     root=root, transform=transform_val, download=False, split="train")
        train_subset = _rebuild_subset_with_transform(dsets.Food101, train_subset.indices,
                                                     root=root, transform=transform,     download=False, split="train")
    return train_subset, val_subset

# -------------------------
# Generic fallback – supports "train", "test", and "train+val" via train split
# -------------------------
def _adapter_generic(name: str) -> Callable[..., AdapterReturn]:
    Cls = getattr(dsets, name)

    def fn(split, root, transform, download, val_ratio=0.1, seed=42, transform_val=None, **extra)->AdapterReturn:
        sig = inspect.signature(Cls.__init__).parameters
        has_train = "train" in sig
        has_split = "split" in sig

        def build(tr, *, train=None, split_kw=None, need_download=True):
            kwargs = {"root": root, "transform": tr}
            if "download" in sig: kwargs["download"] = need_download and download
            if has_train and train is not None: kwargs["train"] = train
            if has_split and split_kw is not None: kwargs["split"] = split_kw
            # forward supported extras
            for k, v in (extra or {}).items():
                if k in sig: kwargs[k] = v
            return Cls(**kwargs)

        # test
        if split == "test":
            if has_train:
                return build(transform_val or transform, train=False)
            if has_split:
                # try "test", fallback to "val" or error
                try:    return build(transform_val or transform, split_kw="test")
                except: return build(transform_val or transform, split_kw="val")
            return build(transform_val or transform)

        # train
        if split == "train":
            if has_train: return build(transform, train=True)
            if has_split: return build(transform, split_kw="train")
            return build(transform)

        # train+val
        if has_train:
            full_train = build(transform, train=True)
            n_total = len(full_train)
            n_val = int(n_total * val_ratio); n_train = n_total - n_val
            g = torch.Generator().manual_seed(seed)
            train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=g)

            if transform_val is not None:
                full_for_val = build(transform_val, train=True, need_download=False)
                val_subset   = Subset(full_for_val, val_subset.indices)
                train_subset = Subset(full_train,  train_subset.indices)
            return train_subset, val_subset

        if has_split:
            # If dataset actually has native val, return (train, val)
            try:
                train_ds = build(transform, split_kw="train")
                val_ds   = build(transform_val or transform, split_kw="val")
                return train_ds, val_ds
            except:
                # fallback: split "train" into (train,val)
                full_train = build(transform, split_kw="train")
                n_total = len(full_train)
                n_val = int(n_total * val_ratio); n_train = n_total - n_val
                g = torch.Generator().manual_seed(seed)
                train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=g)

                if transform_val is not None:
                    full_for_val = build(transform_val, split_kw="train", need_download=False)
                    val_subset   = Subset(full_for_val, val_subset.indices)
                    train_subset = Subset(full_train,  train_subset.indices)
                return train_subset, val_subset

        # No concept of split: just return the dataset (caller can manage)
        return build(transform)

    return fn

# -------------------------
# Registry
# -------------------------
ADAPTERS: Dict[str, Callable[..., AdapterReturn]] = {
    "CIFAR10":       _adapter_cifar10,
    "CIFAR100":      _adapter_cifar100,
    "MNIST":         _adapter_mnist_like(dsets.MNIST),
    "FashionMNIST":  _adapter_mnist_like(dsets.FashionMNIST),
    "EMNIST":        _adapter_mnist_like(dsets.EMNIST),   # pass EMNIST-specific extras via dataset_kwargs (e.g., split="byclass")
    "QMNIST":        _adapter_mnist_like(dsets.QMNIST),
    "SVHN":          _adapter_svhn,
    "STL10":         _adapter_stl10,
    "CelebA":        _adapter_celeba,
    "Flowers102":    _adapter_flowers102,
    "Food101":       _adapter_food101,
}
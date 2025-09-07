from typing import Optional
from torch.utils.data import DataLoader
from loaders.helpers import make_transforms_pair, ADAPTERS, _adapter_generic, _first_batch_shape

class Loader:
    """
    A modular dataloader that normalizes split names and adapts to many torchvision datasets.
    """
    def __init__(
        self,
        name: str,
        batch_size: int,
        num_workers: int = 4,
        split: str = "train",                 # "train" | "train+val" | "test"
        val_ratio: float = 0.05,
        seed: int = 42,
        normalize_to_neg1: bool = True,
        apply_aug: bool = True,
        img_size: Optional[int] = None,       # optional resize
        rgb: Optional[bool] = None,           # if None, we guess from name
        shuffle: Optional[bool] = None,       # if None, True for train, False otherwise
        drop_last: Optional[bool] = True,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        download: bool = True,
        **dataset_kwargs,                      # extra args forwarded when supported
    ):
        self.name = name
        self.root = f"../data/{name}"
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize_to_neg1 = normalize_to_neg1
        self.apply_aug = apply_aug

        if persistent_workers is None:
            persistent_workers = (num_workers > 0)

        # heuristic channel guess if not provided
        if rgb is None:
            rgb = name not in {"MNIST", "FashionMNIST", "EMNIST", "QMNIST"}

        self.transform_train, self.transform_val = make_transforms_pair(
            dataset_name=name,
            img_size=img_size,
            normalize_to_neg1=normalize_to_neg1,
            apply_aug=apply_aug
        )

        # pick adapter
        adapter = ADAPTERS.get(name, _adapter_generic(name))
        if split == "train" :
            dataset = adapter(
                split=split,
                root=self.root,
                transform=self.transform_train,
                download=download,
                **dataset_kwargs
            )
        elif split == "test" :
            dataset = adapter(
                split=split,
                root=self.root,
                transform=self.transform_val,
                download=download,
                **dataset_kwargs
            )
        else :
            dataset, dataset_val = adapter(
                split=split,
                root=self.root,
                transform=self.transform_train,
                download=download,
                val_ratio=val_ratio,
                seed=seed,
                transform_val=self.transform_val,
                **dataset_kwargs
            )
            self.dataloader_val = DataLoader(
                dataset_val,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

        self.image_shape = _first_batch_shape(dataset)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class Dataloader:

    def __init__(self, name, batch_size, n_workers, n_epochs):

        self.name_dataset = name.upper()
        self.batch_size = batch_size 
        self.n_workers = n_workers
        self.train = n_epochs > 0  # if n_epochs is 0 we perform inference
        self.dataloader, self.image_shape = self.get_dataloader_and_shape()

    def get_dataloader_and_shape(self):
        Dataclass = getattr(torchvision.datasets, self.name_dataset)
        dataset = Dataclass(
            root=f'./data/{self.name_dataset}',
            train=self.train,
            download=True,
            transform=ToTensor()
        )

        # Peek at the first sample to get image shape
        sample_img, _ = dataset[0]          # sample_img: torch.Tensor, shape [C, H, W]
        image_shape = tuple(sample_img.shape)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers
        )

        return dataloader, image_shape

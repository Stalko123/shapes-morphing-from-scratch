import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class Dataloader:

    def __init__(self, args):

        self.name_dataset= args.name_dataset #In capital letters like MNIST, CIFAR10, etc.
        self.batch_size = args.batch_size 
        self.n_workers = args.n_workers
        self.train = args.n_epochs > 0 #If n_epochs is 0, we don't train the model (We will have separate test/train dataloaders)
        self.dataloader = self.get_dataloader()


    def get_dataloader(self):
        Dataclass = getattr(torchvision.datasets, self.name_dataset)
        dataset = Dataclass(root=f'./data/{self.name_dataset}', train=self.train, download=True,
                            transform=ToTensor())

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)
        


            
    


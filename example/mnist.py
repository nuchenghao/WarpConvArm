import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential()


def main(
    batch_size: int = 128,
    test_batch_size: int = 1000,
    epochs: int = 1,
    lr: float = 1e-3,
    scheduler_step_size: int = 10,
    gamma: float = 0.7,
    device: str = "cuda",
):
    torch.manual_seed(1)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=test_batch_size,
        shuffle=True,
    )


if __name__ == "__main__":
    fire.Fire(main)

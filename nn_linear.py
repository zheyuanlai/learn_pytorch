import torch
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch import nn
dataset = torchvision.datasets.CIFAR10('./dataset', train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = Linear(196608, 10)
    def forward(self, input):
        output = self.linear(input)
        return output

model = Model()

for data in dataloader:
    imgs, targets = data
    output = torch.flatten(imgs)
    output = model(output)
    print(output)
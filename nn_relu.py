import torch
from torch import nn
from torch.nn import ReLU, Sigmoid
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10('./dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
    def forward(self, input):
        output = self.relu(input)
        return output

model = Model()
# output = model(input)
# print(output)

writer = SummaryWriter('logs_relu')
step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images('input', imgs, step)
    output = model(imgs)
    writer.add_images('output', output, step)
    step += 1

writer.close()

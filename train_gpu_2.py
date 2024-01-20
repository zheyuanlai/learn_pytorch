import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.nn import Sequential

device = torch.device('mps')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

train_data = torchvision.datasets.CIFAR10('./dataset', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('./dataset', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

train_data_size, test_data_size = len(train_data), len(test_data)
# print(train_data_size, test_data_size)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

model = Model(); model = model.to(device)
loss_fn = nn.CrossEntropyLoss(); loss_fn = loss_fn.to(device)
learning_rate = 1e-2  # 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 30
writer = SummaryWriter('logs_train')

for i in range(epoch):
    print(f'Starting Test Round {i + 1}')

    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f'Testing {total_train_step} step(s), Loss: {loss.item()}')
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f'Whole Loss: {total_test_loss}')
    print(f'Accuracy of Whole Test Cases: {total_accuracy / test_data_size}')
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / test_data_size, total_test_step)
    total_test_step += 1
    torch.save(model, f'model {i + 1}.pth')
    # torch.save(model.state_dict(), f'model {i + 1}.pth')
    print(f'Model {i + 1} has been saved.')
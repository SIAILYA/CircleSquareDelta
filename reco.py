import numpy as np
import pendulum
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from statistics import mean


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def load_dataset(data_path='datasets/images/', shuffle=True):
    trans = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = torchvision.datasets.ImageFolder(root=data_path,
                                               transform=trans)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=64,
                                         num_workers=0,
                                         shuffle=shuffle)
    return loader


if __name__ == '__main__':
    model = Net()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    loss_list = []
    main_timer = pendulum.now()
    for epoch in range(200):
        timer = pendulum.now()
        for batch_idx, (data, target) in enumerate(load_dataset()):
            data = data.cuda()
            target = target.cuda()
            # Run the forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}; Loss {mean(loss_list)}; Time: {timer.diff(pendulum.now()).as_timedelta()}')
    print(f'All training time: {main_timer.diff(pendulum.now()).as_timedelta()}')

    torch.save(model.state_dict(), 'models/my_model')
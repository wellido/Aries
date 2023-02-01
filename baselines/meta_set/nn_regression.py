import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import torch.nn.init as init
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()
        # Batch x Channel x Height x Width; 64- feature dimension
        dimension = 200
        self.conv1 = nn.Conv2d(1, int(dimension/2), (dimension, 1), 1).apply(kaiming_init)
        self.conv2 = nn.Conv2d(int(dimension/2), 1, 1, 1).apply(kaiming_init)
        self.fc1 = nn.Linear(dimension, int(dimension/2)).apply(kaiming_init)
        self.fc2 = nn.Linear(dimension, int(dimension/2)).apply(kaiming_init)
        self.fc3 = nn.Linear(dimension + 1, int(dimension/2)).apply(kaiming_init)
        self.fc4 = nn.Linear(int(dimension/2), 1).apply(kaiming_init)
        self.dropout1 = nn.Dropout2d(0.15)
        self.dropout2 = nn.Dropout2d(0.15)
        self.dropout3 = nn.Dropout2d(0.5)

        # dimension = 50
        # self.conv1 = nn.Conv2d(1, 32, (64, 1), 1).apply(kaiming_init)
        # self.conv2 = nn.Conv2d(32, 1, 1, 1).apply(kaiming_init)
        # self.fc1 = nn.Linear(64, 32).apply(kaiming_init)
        # self.fc2 = nn.Linear(64, 32).apply(kaiming_init)
        # self.fc3 = nn.Linear(64 + 1, 32).apply(kaiming_init)
        # self.fc4 = nn.Linear(32, 1).apply(kaiming_init)
        # self.dropout1 = nn.Dropout2d(0.15)
        # self.dropout2 = nn.Dropout2d(0.15)
        # self.dropout3 = nn.Dropout2d(0.5)

    def forward(self, x, y, f):
        # x: cov; y: mean
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)

        y = self.fc1(y)
        y = self.dropout2(y)
        z = torch.cat([x, y, f], dim=1)  # mean, variance, and fid
        z = self.fc3(z)
        z = self.dropout3(z)
        z = self.fc4(z)

        output = z.view(-1)
        return output


class REG(data.Dataset):
    """
    Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, path, data, label, fid, transform=None, target_transform=None):
        super(REG, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.path = path
        self.label_file = label
        self.fid = fid

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (mean, var, target) where target is index of the target class.
        """
        mean = np.load(self.path + self.data[index].split(".")[-2] + '_mean.npy')
        var = np.load(self.path + self.data[index].split(".")[-2] + '_variance.npy')
        # mean = np.load(self.path + "new_data_" + self.data[index].zfill(3) + '_mean.npy')
        # var = np.load(self.path + "new_data_" + self.data[index].zfill(3) + '_variance.npy')
        # print(mean.shape)
        # print(var.shape)
        target = self.label_file[index]
        fid = self.fid[index]
        fid = torch.as_tensor(fid, dtype=torch.float).view(1)

        mean = torch.as_tensor(mean, dtype=torch.float)
        var = torch.as_tensor(var, dtype=torch.float).view(1, 256, 256)

        target = torch.as_tensor(target, dtype=torch.float)
        return var, mean, target, fid

    def __len__(self):
        return len(self.data)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (var, mean, target, fid) in enumerate(train_loader):
        var, mean, target, fid = var.to(device), mean.to(device), target.to(device), fid.to(device)
        optimizer.zero_grad()
        output = model(var, mean, fid)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(var), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    pred_acc = []
    target_acc = []
    with torch.no_grad():
        for var, mean, target, fid in test_loader:
            print(var.shape)
            print(mean.shape)
            print(fid.shape)
            var, mean, target, fid = var.to(device), mean.to(device), target.to(device), fid.to(device)
            output = model(var, mean, fid)
            pred_acc.append(output.cpu())
            target_acc.append(target.cpu())
            test_loss += F.smooth_l1_loss(output, target, reduction='sum').item()  # sum up batch loss

    R2 = r2_score(torch.cat(target_acc).numpy(), torch.cat(pred_acc).numpy())
    RMSE = mean_squared_error(torch.cat(target_acc).numpy(), torch.cat(pred_acc).numpy(), squared=False)
    MAE = mean_absolute_error(torch.cat(target_acc).numpy(), torch.cat(pred_acc).numpy())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f} R2 :{:.4f} RMSE: {:.4f} MAE: {:.4f}\n'.format(test_loss, R2, RMSE, MAE))


def main_cifar10():
    # CIFAR10
    data = sorted(os.listdir('/home/qhu/qhu-data/TS4code/meta_set/cifar_data_3/'))
    acc = np.load('/home/qhu/qhu-data/TS4code/meta_set/inter_data/cifar10_vgg16_acc_3.npy')
    fid = np.load('/home/qhu/qhu-data/TS4code/meta_set/inter_data/fd_cifar10_vgg16_3.npy')
    feature_path = '/home/qhu/qhu-data/TS4code/meta_set/inter_data/cifar10_feature_vgg16_3/'

    # SVHN
    # data = sorted(os.listdir('/home/qhu/qhu-data/TS4code/meta_set/svhn_data_2/'))
    # acc = np.load('/home/qhu/qhu-data/TS4code/meta_set/inter_data/svhn_lenet5_acc_2.npy')
    # fid = np.load('/home/qhu/qhu-data/TS4code/meta_set/inter_data/fd_svhn_lenet5_2.npy')
    # feature_path = '/home/qhu/qhu-data/TS4code/meta_set/inter_data/svhn_feature_lenet5_2/'

    # ImageNet
    # data = sorted(os.listdir('/home/qhu/qhu-data/TS4code/meta_set/imagenet_data_3/'))
    # acc = np.load('/home/qhu/qhu-data/TS4code/meta_set/inter_data/imagenet_acc_resnet_3.npy')
    # fid = np.load('/home/qhu/qhu-data/TS4code/meta_set/inter_data/fd_imagenet_resnet_3.npy')
    # feature_path = '/home/qhu/qhu-data/TS4code/meta_set/inter_data/imagenet_feature_resnet_3/'


    index = 30


    train_data = data[index:]
    train_acc = acc[index:]
    train_fid = fid[index:]

    test_data = data[:index]
    test_acc = acc[:index]
    test_fid = fid[:index]
    use_cuda = 0
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    batch_size = 16
    test_batch_size = 1000
    lr = 1.0
    epochs = 210
    # epochs = 1
    gamma = 0.8
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = torch.utils.data.DataLoader(
        REG(feature_path, train_data, train_acc, train_fid),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        REG(feature_path, test_data, test_acc, test_fid),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    model = RegNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=gamma)

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    torch.save(model.state_dict(), "cifar10_vgg16_model/cifar10_regnet_vgg16_3.pt")


if __name__ == "__main__":
    main_cifar10()

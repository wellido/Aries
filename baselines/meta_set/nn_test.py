import torch
import numpy as np
from nn_regression import RegNet

model = RegNet()
model.load_state_dict(torch.load("imagenet_densenet_model/imagenet_regnet_densenet_3.pt", map_location=torch.device('cpu')))
# model.load_state_dict(torch.load("cifar10_model/cifar10_regnet_1.pt", map_location=torch.device('cpu')))
# print(model)

data_types = ['brightness',	'contrast',	'defocus_blur',	'elastic_transform', 'fog',	'frost',
              'gaussian_blur', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise',
              'snow', 'zoom_blur']

# data_type = "brightness"

accs = []
# for data_type in data_types:
for i in range(len(data_types)):
    model.eval()
    var = np.load("inter_data/imagenet_densenet/new_data_" + str(i).zfill(3) + "_variance.npy")
    mean = np.load("inter_data/imagenet_densenet/new_data_" + str(i).zfill(3) + "_mean.npy")
    fid = np.load("inter_data/4test/fd_imagenet_densenet_4test.npy")[0]

    # var = np.load("inter_data/cifar10_vgg16/new_data_" + data_type + "_variance.npy")
    # mean = np.load("inter_data/cifar10_vgg16/new_data_" + data_type + "_mean.npy")
    # fid = np.load("inter_data/4test/fd_cifar10_vgg16_4test.npy")[0]

    # var = np.load("inter_data/dataset_feature/new_data_" + data_type + "_variance.npy")
    # mean = np.load("inter_data/dataset_feature/new_data_" + data_type + "_mean.npy")
    # fid = np.load("inter_data/dataset_feature/fd_cifar10_4test.npy")[0]

    fid = torch.as_tensor(fid, dtype=torch.float).view(1, 1)
    mean = torch.as_tensor(mean, dtype=torch.float).view(1, -1)
    var = torch.as_tensor(var, dtype=torch.float).view(1, 1, 200, 200)
    use_cuda = 0
    device = torch.device("cuda" if use_cuda else "cpu")
    with torch.no_grad():
        var, mean, fid = var.to(device), mean.to(device), fid.to(device)
        output = model(var, mean, fid)
        print(output)
        accs.append(output.numpy()[0])

print(accs)


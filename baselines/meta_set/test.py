import numpy as np

# a = np.load("cifar_data/new_data_000.npy")
# print(a.shape)
# print(str(1).zfill(3))

a = "new_data_999.npy"
data = np.load(a)
print(data.shape)

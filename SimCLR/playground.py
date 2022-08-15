import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np


training_data = datasets.CIFAR10(
    root="./datasets",
    train=True,
    download=True,
    transform=ToTensor()
)
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
samples_weight = torch.zeros(9)
sample_idx = torch.randint(len(training_data), size=(1,)).item()
img, label = training_data[sample_idx]
target_idx = torch.randint(len(training_data.targets == label), size=(1,)).item()


# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
    
#     img = img.permute(1,2,0)
#     figure.add_subplot(rows, cols, i)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
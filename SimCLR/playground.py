import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN_Classifier(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = 1
training_data = datasets.CIFAR10(
    root="./datasets",
    train=True,
    download=False,
    transform=ToTensor()
)
# x
sample_idx = torch.randint(len(training_data), size=(1,)).item()
img, label = training_data[sample_idx]
# R = model(img)
# print(training_data.targets)
# print(label)

# x'
target_idx_list = training_data.targets
print(target_idx_list)
# target_idx = torch.randint(torch.tensor(training_data.targets) == label, size=(1,)).item()
# target_img, target_label = training_data[target_idx]
# img = img.permute(1,2,0)
# target_img = target_img.permute(1,2,0)

# f, axarr = plt.subplots(1,2)
# axarr[0].imshow(img.squeeze(), cmap="gray")
# axarr[1].imshow(target_img.squeeze(), cmap="gray")

# plt.show()
# training_sample = torch.cat(target_img, R)

# classifier = CNN_Classifier()


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
    
#     img = img.permute(1,2,0)
#     figure.add_subplot(rows, cols, i)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
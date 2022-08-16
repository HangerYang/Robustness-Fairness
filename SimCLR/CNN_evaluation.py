import os
import argparse
from random import triangular
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from model import save_model

from simclr import SimCLR
from simclr.modules import cnn, get_resnet
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook


def inference(data, simclr_model, device):
    x = data
    x = x[None, :]
    x = x.to(device)
    with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)
    # print("Features shape {}".format(h.shape))
    return h


def train(args, train_dataset, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for x, y in train_dataset:
        optimizer.zero_grad()
        R = inference(x, simclr_model, args.device)
        target_idx_list = train_dataset.targets
        target_idx = np.random.choice(np.where(np.array(target_idx_list) == y)[0])
        target_img, _ = train_dataset[target_idx]
        target_img = target_img.to(args.device) #inefficient
        output = model(target_img, R)
        predicted = torch.argmax(output)
        acc = (predicted == y).sum()
        accuracy_epoch += acc
        loss = criterion(output, torch.tensor(y).to(args.device))
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    return loss_epoch, accuracy_epoch

def test(args, test_dataset, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for x, y in test_dataset:
        model.zero_grad()
        x = x.to(args.device)
        y = torch.tensor(y).to(args.device)
        R = inference(x, simclr_model, args.device)
        output = model(x, R)
        loss = criterion(output, y)

        predicted = torch.argmax(output)
        acc = (predicted == y).sum()
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
        args.dataset_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    elif args.dataset == "CelebA":
        train_dataset = torchvision.datasets.CelebA(
            args.dataset_dir,
            download=True,
            transform=transforms.ToTensor(),
        )
        test_dataset = torchvision.datasets.CelebA(
            args.dataset_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    else:
        raise NotImplementedError
    
    

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=args.workers,
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=args.logistic_batch_size,
    #     shuffle=False,
    #     drop_last=True,
    #     num_workers=args.workers,
    # )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    # model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    ## Logistic Regression
    n_classes = 10  # CIFAR-10 / STL-10
    # model = LogisticRegression(simclr_model.n_features, n_classes)
    model = cnn.CNN_Classifier()
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(
                args, train_dataset, simclr_model, model, criterion, optimizer)
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_dataset)} \t Accuracy: {accuracy_epoch / len(train_dataset)}"
        )

        if epoch % 50 == 0:
            save_model(args, model, optimizer)
    # print("### Creating features from pre-trained context model ###")
    # (train_X, train_y, test_X, test_y) = get_features(
    #     simclr_model, train_loader, test_loader, args.device
    # )

    # arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
    #     train_X, train_y, test_X, test_y, args.logistic_batch_size
    # )

    # for epoch in range(args.logistic_epochs):
    #     loss_epoch, accuracy_epoch = train(
    #         args, arr_train_loader, simclr_model, model, criterion, optimizer
    #     )
    #     print(
    #         f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
    #     )

    # final testing
    loss_epoch, accuracy_epoch = test(
        args, test_dataset, simclr_model, model, criterion, optimizer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_dataset)}\t Accuracy: {accuracy_epoch / len(test_dataset)}"
    )

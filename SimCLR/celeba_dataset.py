import random
import torch
from functools import partial
from torchvision import transforms, datasets
import os
import numpy as np
import pandas

from torch.utils.data import Dataset
from natsort import natsorted
from PIL import Image



class CelebADataset(Dataset):
  def __init__(self, root_dir, target_feat='Smiling', sensitive_feat='Male', transform=None, target_type=["attr"]):
    """
    Args:
      root_dir (string): Directory with all the dataset information
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    self.base_folder = 'celeba'
    image_names = os.listdir(os.path.join(root_dir, self.base_folder, 'img_align_celeba'))

    self.root = root_dir
    self.transform = transform 
    self.filename = natsorted(image_names)


    fn = partial(os.path.join, self.root, self.base_folder)
    identity = pandas.read_csv(fn("identity_CelebA.txt"),
                                delim_whitespace=True, header=None,
                                index_col=0)
    bbox = pandas.read_csv(fn("list_bbox_celeba.txt"),
                            delim_whitespace=True, header=1, index_col=0)
    landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"),
                                        delim_whitespace=True, header=1)
    attr = pandas.read_csv(fn("list_attr_celeba.txt"),
                            delim_whitespace=True, header=1)

    self.identity = torch.as_tensor(identity.values)
    self.bbox = torch.as_tensor(bbox.values)
    self.landmarks_align = torch.as_tensor(landmarks_align.values)
    self.attr = torch.as_tensor(attr.values)
    self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
    self.attr_names = list(attr.columns)

    self.target_feat = target_feat
    self.sensitive_feat = sensitive_feat
    self.target_type = target_type
    self.target_feat_col_id = (np.array(self.attr_names) == self.target_feat).argmax()
    self.sensitive_feat_col_id = (np.array(self.attr_names) == self.sensitive_feat).argmax()


  def __len__(self): 
    return len(self.filename)

  def __getitem__(self, index):
    X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba",
                         self.filename[index])).convert('RGB')

    target = []
    for t in self.target_type:
        if t == "attr":
            target.append(self.attr[index, :])
        elif t == "identity":
            target.append(self.identity[index, 0])
        elif t == "bbox":
            target.append(self.bbox[index, :])
        elif t == "landmarks":
            target.append(self.landmarks_align[index, :])
        else:
            # TODO: refactor with utils.verify_str_arg
            raise ValueError(
                "Target type \"{}\" is not recognized.".format(t))

    # Apply transformations to the image
    if self.transform:
      X = self.transform(X)

    target = target[0]
    
    return X, target[self.target_feat_col_id], target[self.sensitive_feat_col_id]



def main():
    '''
    # Number of gpus available
    ngpu = 1
    device = torch.device('cuda:0' if (
        torch.cuda.is_available() and ngpu > 0) else 'cpu')
    '''
    # this code creates a dataloader for the celeba dataset and prints the first data

    # Root directory for the dataset
    data_root = 'data'

    # Spatial size of training images, images are resized to this size.
    image_size = 64
    # Transformations to be applied to each individual image sample
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    ])
    # Load the dataset from file and apply transformations
    celeba_dataset = CelebADataset(root_dir=f'{data_root}', transform=transform)
    # Batch size during training
    batch_size = 128
    # Number of workers for the dataloader
    num_workers = 2
    # Whether to put fetched data tensors to pinned memory
    pin_memory =  False

    celeba_dataloader = torch.utils.data.DataLoader(celeba_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory,
                                                    shuffle=True)
    
    print(celeba_dataset[0])

if __name__ == "__main__":
    main()
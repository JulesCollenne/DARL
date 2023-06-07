import os
import platform

import numpy as np
import pandas as pd

import torch
from torch.utils import  data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def get_loader(dataset="2020", img_size=224, batch_size=16, workers=4, distributed=False):
    if dataset == "2020":
        normalize = transforms.Normalize(mean=[0.805, 0.615, 0.587],
                                         std=[0.148, 0.175, 0.201])
        if platform.system() == 'Windows':
            dataset_path = "E:\\BDD\\ISIC_2020\\"
            images_path = "E:\\BDD\\ISIC_2020\\train\\"
        else:
            dataset_path = "/home/jules/Travail/isic/ISIC_2020/"
            images_path = "/home/jules/Travail/isic/ISIC_2020/train/"
    elif dataset == "2020":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if platform.system() == 'Windows':
            dataset_path = "E:\\BDD\\ISIC_2020\\train\\"
        else:
            dataset_path = "/home/jules/Travail/isic/ISIC_2020/train/"
    else:
        print("Error in dataset!")
        exit()

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    transforms_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transforms_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
        transforms.ToTensor(),
        normalize
    ])

    # Data loading code
    if dataset == "2019":
        traindir = os.path.join(dataset_path, "train")
        valdir = os.path.join(dataset_path, "val")
        testdir = os.path.join(dataset_path, "test")
        train_dataset = CustomImageLoaderFolder(
            traindir,
            transforms_train
        )
        val_dataset = CustomImageLoaderFolder(valdir, transforms_val)
        test_dataset = CustomImageLoaderFolder(testdir, transforms_val)
    else:
        train_df = pd.read_csv(dataset_path+"y_train.csv")
        val_df = pd.read_csv(dataset_path+"y_val.csv")
        test_df = pd.read_csv(dataset_path+"y_test.csv")
        train_dataset = ISIC2020_Dataset(
            train_df,
            transforms_train,
            dataset_path=images_path
        )
        val_dataset = ISIC2020_Dataset(
            val_df,
            transforms_val,
            dataset_path=images_path
        )
        test_dataset = ISIC2020_Dataset(
            test_df,
            transforms_train,
            dataset_path=images_path
        )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    # random.seed(0)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader, test_loader


class CustomImageLoaderFolder(data.Dataset):
    def __init__(self, root, transform=None, get_names=False):
        self.root = root
        self.transform = transform
        self.images = []
        self.targets = []
        self.classes = []
        self.class_to_idx = {}
        self.num_classes = 0
        self.samples = []
        self.get_names = get_names

        for target_class in os.listdir(root):
            class_path = os.path.join(root, target_class)
            if not os.path.isdir(class_path):
                continue

            self.classes.append(target_class)
            self.class_to_idx[target_class] = self.num_classes
            self.num_classes += 1

            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                if not os.path.isfile(img_path):
                    continue

                self.images.append(img_path)
                self.targets.append(self.class_to_idx[target_class])

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.targets[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.get_names:
            return img, label, img_path
        return img, label

    def __len__(self):
        return len(self.images)


class ISIC2020_Dataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None, dataset_path="./dataset/", feature_extraction=False,
                 get_names=False):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_path = dataset_path
        self.feature_extraction = feature_extraction
        self.get_names = get_names

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.dataset_path + self.df['image_name'].iloc[idx] + ".jpg")
        labels = self.df['target'].iloc[idx]
        names = self.df['image_name'].iloc[idx]
        label = torch.tensor(np.asarray(labels))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.feature_extraction:
            return image, names
        if self.get_names:
            return image, label, names
        return image, label

    def get_weights(self):
        return len(self.df["target"]) / len([i for i in self.df["target"] if i == 0.]), len(self.df["target"]) / len(
            [i for i in self.df["target"] if i == 1.])

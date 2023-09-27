"""Repeatable code parts concerning data loading."""
import numpy as np

import torch
from torch.utils.data import TensorDataset, Subset
import torchvision
from torchvision import transforms

import os

from ..consts import *

from .data import _build_bsds_sr, _build_bsds_dn
from .loss import Classification, PSNR


def construct_dataloaders(dataset, defs, val=False, data_path='/localscratch2/hbzhang', shuffle=True, normalize=True, resize=(32, 32)):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    path = os.path.expanduser(data_path)

    if dataset == 'CIFAR10':
        trainset, validset, testset = _build_cifar10(path, val, defs.augmentations, normalize, resize=resize)
        loss_fn = Classification()
    elif dataset == 'CIFAR100':
        trainset, validset = _build_cifar100(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'STL':
        trainset, validset = _build_STL(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST':
        trainset, validset = _build_mnist(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST_GRAY':
        trainset, validset = _build_mnist_gray(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'ImageNet':
        trainset, testset = _build_imagenet(path, defs.augmentations, normalize, resize=(224,224))
        validset = None
        loss_fn = Classification()
    elif dataset == 'BSDS-SR':
        trainset, validset = _build_bsds_sr(path, defs.augmentations, normalize, upscale_factor=3, RGB=True)
        loss_fn = PSNR()
    elif dataset == 'BSDS-DN':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=False)
        loss_fn = PSNR()
    elif dataset == 'BSDS-RGB':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=True)
        loss_fn = PSNR()

    if MULTITHREAD_DATAPROCESSING:
        num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    else:
        num_workers = 0

    if trainset:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(defs.batch_size, len(trainset)),
                                                shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    else: trainloader = None
    testloader = torch.utils.data.DataLoader(testset, batch_size=min(defs.batch_size, len(testset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    # if validset:
    return loss_fn, trainset, validset, testset, trainloader, testloader
    # else:
    #     return loss_fn, trainset, testset, trainloader, testloader


def _build_cifar10(data_path, val=False, augmentations=False, normalize=True, resize=(32, 32)):
    """Define CIFAR-10 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar10_mean, cifar10_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    testset.transform = transform
    if val:
        trainset, valset = split_trainset(trainset, 0.2)
        return trainset, valset, testset
    return trainset, None, testset

def _build_STL(data_path, augmentations=True, normalize=True):
    """Define CIFAR-10 with everything considered."""
    # Load data
    cifar10_mean = None
    TRAIN_DATA_PATH = '/localscratch2/hbzhang/stl10_binary/train_X.bin'
    TRAIN_LABEL_PATH = '/localscratch2/hbzhang/stl10_binary/train_y.bin'
    TEST_DATA_PATH = '/localscratch2/hbzhang/stl10_binary/test_X.bin'
    TEST_LABEL_PATH = '/localscratch2/hbzhang/stl10_binary/test_y.bin'

    def read_labels(path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def read_all_images(path_to_data):
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            # images = np.transpose(images, (0, 3, 2, 1))
            return images

    train_X = read_all_images(TRAIN_DATA_PATH)
    train_y = read_labels(TRAIN_LABEL_PATH)
    test_X = read_all_images(TEST_DATA_PATH)
    test_y = read_labels(TEST_LABEL_PATH)

    train_X = torch.from_numpy(train_X).float()
    train_y = torch.from_numpy(train_y).long()
    test_X = torch.from_numpy(test_X).float()
    test_y = torch.from_numpy(test_y).long()

    trainset = TensorDataset(train_X, train_y)
    validset = TensorDataset(test_X, test_y)

    if cifar10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar10_mean, cifar10_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_cifar100(data_path, augmentations=True, normalize=True):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_mnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_mnist_gray(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_imagenet(data_path, augmentations=False, normalize=True, resize=(224, 224)):
    """Define ImageNet with everything considered."""
    # Load data
    print('build imagenet')
    try:
        trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    except:
        trainset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())
    '''    
        if imagenet_mean is None:
            data_mean, data_std = _get_meanstd(trainset)
        else:
            data_mean, data_std = imagenet_mean, imagenet_std
    '''
    # if trainset:
    #     data_mean, data_std = _get_meanstd(trainset)
    # else:
    #     data_mean, data_std = _get_meanstd(validset)
    # print(data_mean, data_std)
    data_mean=[0.485, 0.456, 0.406]
    data_std=[0.229, 0.224, 0.225]

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(resize),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _get_meanstd(trainset):
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std

def split_trainset(train_dataset, val_ratio=0.3, shuffle=True):
    '''
    split the training set into training set and validation set
    '''
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_ratio * num_train))

    if shuffle:
        # np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_set = Subset(train_dataset, indices=train_idx)
    val_set = Subset(train_dataset, indices=val_idx)

    return train_set, val_set



'''
    Codes for inversion with aug data.
'''
import os, sys, math, random, argparse, time
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from collections import OrderedDict
from numbers import Number

import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models.resnet import Bottleneck, BasicBlock

from defense import *

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

cifar10_mean = [0.4913996756076813, 0.48215848207473755, 0.44653090834617615]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
cifar100_mean = [0.5070751905441284, 0.48654890060424805, 0.44091784954071045]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)
fmnist_mean = (0.28604060411453247,)
fmnist_std = (0.3530242443084717,)
mmnist_mean = None
mmnist_std = None
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

'''
    load models:
'''
def load_model(model, num_classes=10, num_channels=3, data_shape=(3, 32, 32), seed=1):
    set_random_seed(seed)
    if model == 'SMLP':
        return SMLP(num_classes=num_classes, data_shape=data_shape)
    elif model == 'DMLP':
        return DMLP(num_classes=num_classes, data_shape=data_shape)
    elif model == 'sDMLP':
        return SDMLP(num_classes=num_classes, data_shape=data_shape)
    elif model == 'TinyMLP':
        return TinyMLP(num_classes=num_classes, data_shape=data_shape)
    elif model == 'LeNet':
        return ConvNet5(num_classes=num_classes, num_channels=num_channels, width=data_shape[-1])
    elif model == 'ResNet20':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_channels=num_channels, num_classes=num_classes, base_width=16)
    elif model == 'ResNet18':
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_channels=num_channels, num_classes=num_classes, base_width=64)
    elif model == 'ResNet32-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_channels=num_channels, num_classes=num_classes, base_width=16 * 10)
    elif model == 'ResNet56':
        return ResNet(torchvision.models.resnet.BasicBlock, [9, 9, 9], num_channels=num_channels, num_classes=num_classes, base_width=16)
    elif model == 'ResNet101':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], num_channels=num_channels, num_classes=num_classes, base_width=64)
    elif model == 'ResNet152':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], num_channels=num_channels, num_classes=num_classes, base_width=64)
    elif model == 'conv8':
        return ConvNet(width=32, num_channels=num_channels, num_classes=num_classes)
    elif model == 'dlg':
        return DLG_model()
    elif model == 'gs':
        return GS_model(num_classes=num_classes, num_channels=num_channels)
    elif model == 'linear':
        return LinearModel(num_classes=num_classes, data_shape=data_shape)
    elif model == 'vit':
        return ViT(image_size=data_shape[1], patch_size=4, num_classes=num_classes, channels=num_channels,
                   dim=512, depth=4, heads=6, mlp_dim=256)

class LinearModel(nn.Module):
    def __init__(self, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.input_shape = np.prod(data_shape)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(self.input_shape, num_classes)
    def forward(self, x):
        return self.fc(self.flat(x))

class TinyMLP(nn.Module):
    def __init__(self, width=512, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
    
class SMLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

class DMLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, width)
        self.l4 = nn.Linear(width, width)
        self.l5 = nn.Linear(width, num_classes)
        self.num_blocks = 10

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.l5(x)
        return x
    
class SDMLP(DMLP):
    def __init__(self, width=1024, num_classes=10, data_shape=(3, 32, 32)):
        super().__init__(width, num_classes, data_shape)
        self.feature = None

    def forward(self, x: torch.Tensor):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        feature = x.view(x.size(0), -1)
        out = self.l5(feature)
        self.feature = feature
        return out

    def extract_feature(self,):
        return self.feature
    
class LeNet(nn.Module):
    '''LeNet in PyTorch from https://github.com/Princeton-SysML/GradAttack/blob/master/gradattack/models/LeNet.py'''
    def __init__(self, num_classes=10, num_channels=3, width=32):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        if width == 32:
            self.fc1 = nn.Linear(400, 120)
        elif width == 28:
            self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, p=False):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        y = self.relu(y)
        return y
    
class ConvNet5(nn.Module):
    def __init__(self, num_classes=10, num_channels=3):
        super(ConvNet5, self).__init__()
        self.num_blocks = 10
        self.conv1 = nn.Conv2d(num_channels, 12, 5, padding=5//2, stride=2)
        self.conv2 = nn.Conv2d(12, 12, 5, padding=5//2, stride=2)
        self.conv3 = nn.Conv2d(12, 12, 5, padding=5//2, stride=1)
        self.conv4 = nn.Conv2d(12, 12, 5, padding=5//2, stride=1)
        self.sigmoid = nn.Sigmoid()
        if num_channels == 3:
            self.fc = nn.Linear(768, num_classes)
        elif num_channels == 1:
            self.fc = nn.Linear(588, num_classes)

    def forward(self, x:torch.Tensor):
        x = self.sigmoid(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        feature = x.view(x.shape[0], -1)
        out = self.fc(feature)
        return out

class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR thingies."""
    def __init__(self, block, layers, num_channels=3, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=[1, 2, 2, 2], pool='avg'):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        self.feature = None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # feature = torch.flatten(x, 1)
        # x = self.fc(feature)
        # self.feature = feature
        return x

    # def extract_feature(self,):
    #     return self.feature

class ConvNet(torch.nn.Module):
    """ConvNetBN."""
    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', torch.nn.BatchNorm2d(4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d(3)),
            ('flatten', torch.nn.Flatten()),
            # ('linear', torch.nn.Linear(36 * width, num_classes))
        ]))
        self.linear = nn.Linear(36*width, num_classes)
        self.num_blocks = 34

    def forward(self, x):
        feature = self.model(x)
        out = self.linear(feature)
        return out

class DLG_model(nn.Module):
    def __init__(self):
        super(DLG_model, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 10),
            #act(),
            #nn.Linear(256, 10)
        )
        
    def forward(self, x):
        out = self.body(x)
        feature = out.view(out.size(0), -1)
        #print(feature.size())
        out = self.fc(feature)
        return out, feature

class GS_model(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.kernel_size = 5
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=self.kernel_size, padding=1)),
            ('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=self.kernel_size, padding=1)),
            ('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=self.kernel_size, padding=1)),
            ('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=self.kernel_size, padding=1)),
            ('bn3', torch.nn.BatchNorm2d(4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=self.kernel_size, padding=1)),
            ('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=self.kernel_size, padding=1)),
            ('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=self.kernel_size, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=self.kernel_size, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=self.kernel_size, padding=1)),
            ('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d(3)),
            ('flatten', torch.nn.Flatten())
            #('linear', torch.nn.Linear(36 * width, num_classes))
        ]))
        self.linear = torch.nn.Linear(36 * width, num_classes)
        self.feature = None

    def forward(self, input):
        self.feature = self.model(input)
        return self.linear(self.feature)

    def extract_feature(self):
        return self.feature

# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) # 224
        patch_height, patch_width = pair(patch_size) # 4

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) # 3136
        patch_dim = channels * patch_height * patch_width # 48
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

'''
    load datasets:
'''
def load_datasets(args, dataset, path='./data', val=True, augmentations=False, normalize=True, resize=(32,32), shuffle=False):
    set_random_seed(args.seed)
    if dataset == 'cifar10':
        trainset, testset, data_mean, data_std = _build_cifar10(path, augmentations, normalize, resize=resize)
    elif dataset == 'cifar100':
        trainset, testset, data_mean, data_std = _build_cifar100(path, augmentations, normalize)
    elif dataset == 'mnist':
        trainset, testset, data_mean, data_std = _build_mnist(path, augmentations, normalize)
    elif dataset == 'imagenet':
        path = './data/ILSVRC2012/'
        trainset, testset, data_mean, data_std = _build_imagenet(path, augmentations, normalize, resize=resize)

    testloader = torch.utils.data.DataLoader(testset, batch_size=min(args.train_bsz, len(testset)),
                                              shuffle=False, drop_last=False, num_workers=64)
    if val:
        trainset, valset = split_trainset(trainset, 0.1)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(args.train_bsz, len(trainset)),
                                                shuffle=shuffle, drop_last=True, num_workers=64)
        valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                                shuffle=True, drop_last=True, num_workers=64)
        return trainset, valset, testset, trainloader, valloader, testloader, data_mean, data_std
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(args.train_bsz, len(trainset)),
                                                shuffle=shuffle, drop_last=True, num_workers=64)
        return trainset, testset, trainloader, testloader, data_mean, data_std

def _build_cifar10(data_path, augmentations=False, normalize=True, resize=(32, 32)):
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
    return trainset, testset, data_mean, data_std

def _build_cifar100(data_path, augmentations=False, normalize=True):
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

    return trainset, validset, data_mean, data_std

def _build_mnist(data_path, augmentations=False, normalize=True):
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

    return trainset, validset, data_mean, data_std

def _build_imagenet(data_path='/localscratch2/hbzhang/ILSVRC2012/', augmentations=False, normalize=True, resize=(3, 224, 224)):
    validset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=transforms.ToTensor())
    trainset = validset

    if imagenet_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = imagenet_mean, imagenet_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset, data_mean, data_std

def _get_meanstd(trainset):
    num_channels = trainset[0][0].shape[0]
    cc = torch.cat([trainset[i][0].reshape(num_channels, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std

def split_trainset(train_dataset, val_ratio=0.1, shuffle=True):
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
    
'''
    some funcs:
'''
def test_model(model, validloader, setup):
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    losses = []
    model.eval()
    with torch.no_grad():
        for data, target in validloader:
            data, target = Variable(data).to(**setup), Variable(target).to(setup['device'])
            output = model(data)
            if len(output) != data.shape[0]:
                output = output[0]
            loss = criterion(output, target)
            losses.append(loss.item())
            # get the index of the max log-probability
            _, predictions = output.max(1)
            total += predictions.size(0)
            correct += torch.sum(predictions == target.data).float()
    acc = correct / total
    loss = sum(losses) / len(losses)
    return loss, acc.item()

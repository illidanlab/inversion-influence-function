'''
    A demo for calculating I2F and its lower bound.
'''

import numpy as np
import torch
from torch import nn
import inversefed
from defense import *
import baseline_utils
import utils

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

defs = inversefed.training_strategy('conservative')
setup = inversefed.utils.system_startup(gpu=0)
args = utils.dotdict()
args.seed = 0

'''set dataset params:'''
# number of classes, img size, data shape:
args.num_classes = 10
args.dataset == 'mnist'
resize = (28, 28)
data_shape = (1, 28, 28)
# num of img channels:
args.num_channels = 1
# data loaders
args.train_bsz = 1
args.train_lr = 0.1
args.normalize = True

'''load dataset'''
train_set, test_set, trainloader, testloader, dm, ds = baseline_utils.load_datasets(args, 'mnist', val=False, resize=resize, normalize=args.normalize)
dm = torch.as_tensor(dm, **setup)[:, None, None]
ds = torch.as_tensor(ds, **setup)[:, None, None]

'''load modules:'''
model = baseline_utils.load_model('conv5', num_classes=args.num_classes, data_shape=data_shape, num_channels=args.num_channels)
model.to(**setup)
model.apply(weights_init)
model.eval()
criterion = nn.CrossEntropyLoss()
model.zero_grad()
loss_fn = nn.CrossEntropyLoss()

'''sample an image from the test set:'''
img, label = test_set[0]
labels = torch.as_tensor((label,), device=setup['device'])
ground_truth = img.to(**setup).unsqueeze(0)

'''sample random noise from Gaussian distribution:'''
loss = criterion(model(ground_truth), labels)
input_gradient = torch.autograd.grad(loss, model.parameters())
input_gradient = [grad.detach().clone() for grad in input_gradient]
# flatten the gradient to a vector:
flat_input_grad, _shape, _cum = flat_recover_vector(input_gradient, func='flat')
# set std:
std = np.sqrt(1e-4) * torch.ones_like(flat_input_grad.data).detach().clone()
# sample noise:
flat_noise = torch.normal(mean=0., std=std).to(**setup)
# recover the flat noise to the same shape as the input_gradient
noise = flat_recover_vector(flat_noise, func='recover', shapes=_shape, cumulated_num=_cum)

'''calculate I2F and its lower bound:'''
# compute I2F:
I2F = utils.compute_exact_bound(args, model, ground_truth, labels, noise, setup, regu=1)
I2F = torch.norm(I2F)
# compute I2F lower bound:
I2F_lb = utils.I2F_lb(model, ground_truth, labels, noise, setup['device'])

print(f"\nI2F: {I2F}\nI2F-lb: {I2F_lb.item()}")
import os, argparse, sys, time
import numpy as np
import random
from tqdm import tqdm
from typing import List, Tuple
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision
from torchvision import transforms

import utils

'''
    some utils funcs:
'''
def get_gradients(model:nn.Module, data:torch.Tensor, label:torch.Tensor, setup:dict) -> List[torch.Tensor]:
    model.eval().to(**setup)
    pred = model(data.to(setup['device']))
    if len(pred) != data.shape[0]:
        pred = pred[0]
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, label)
    loss.backward()
    gradients = [param.grad.detach() for param in model.parameters()]
    return gradients

def grad_cosine_similarity(grad1, grad2) -> torch.Tensor:
    if (isinstance(grad1, (list, tuple))) and (isinstance(grad2, (list, tuple))):
        # if input a list of gradient, first expand it to flat
        flat_grad1 = torch.tensor([]).to(grad1[0].device)
        flat_grad2 = torch.tensor([]).to(grad2[0].device)
        for subgrad1, subgrad2 in zip(grad1, grad2):
            flat_grad1 = torch.cat((flat_grad1, subgrad1.detach().flatten()))
            flat_grad2 = torch.cat((flat_grad2, subgrad2.detach().flatten()))
        # print(flat_grad1.shape)
        return F.cosine_similarity(flat_grad1, flat_grad2, dim=0).detach()#.item()
    elif (isinstance(grad1, torch.Tensor)) and (isinstance(grad2, torch.Tensor)):
        return F.cosine_similarity(grad1.detach(), grad2.detach(), dim=0).detach()#.item()
    else:
        raise TypeError('The type of input is {}'.format(type(grad1)))

def flat_recover_vector(vector, func:str='flat', shapes=None, cumulated_num=None):
    '''
        flat: Tuple[torch.Tensor] -> torch.Tensor
        recover: torch.Tensor -> List[torch.Tensor]
    '''
    if func == 'flat':
        # flat the vector:
        shapes = []
        cumulated_num = []
        flat_vector = torch.tensor([]).to(vector[0].device)
        for tensor_idx, tensor in enumerate(vector):
            flat_vector = torch.cat((flat_vector, tensor.flatten()))
            shapes.append(tensor.shape)
            # cummulated_num.append(torch.prod(torch.tensor(tensor.shape)))
            cumulated_num.append(tensor.numel())
        return flat_vector, shapes, cumulated_num
    elif func == 'recover':
        # reshape the vector:
        recovered_tensor = []
        cum = 0
        for size_idx, size in enumerate(cumulated_num):
            try:
                recovered_tensor.append(vector[cum:cum+size].reshape(shapes[size_idx]))
            except:
                recovered_tensor.append(vector[cum:cum+size].squeeze(0))
            cum += size
        return recovered_tensor
    else: raise NotImplementedError('Not implement %s' % func)

def defend(args, model:nn.Module, ground_truth:torch.Tensor, labels:torch.Tensor, data_shape:tuple, setup:dict,):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.zero_grad()
    loss = criterion(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(loss, model.parameters())
    input_gradient = [grad.detach().clone() for grad in input_gradient]
    flat_input_grad, _shape, _cum = flat_recover_vector(input_gradient, func='flat')
    input_grad_norm = torch.norm(flat_input_grad).item()
    energy = float(args.energy)
    def_stats = {
        'CosSim': 0.,
        'InputNorm': input_grad_norm,
        'NoiseNorm': 0.,
        'NormRatio': 0.,
        'NoisyInputNorm': 0.,
        'loss': loss.item(),
    }
    if args.modify == 'pruning':
        # global pruning
        thres = torch.quantile(torch.abs(flat_input_grad), args.ratio)
        _flat_input_grad = torch.where(torch.abs(flat_input_grad) > thres, flat_input_grad, 0.)
        flat_noise = _flat_input_grad - flat_input_grad
        input_gradient = flat_recover_vector(_flat_input_grad, func='recover', shapes=_shape, cumulated_num=_cum)
        noise = flat_recover_vector(flat_noise, func='recover', shapes=_shape, cumulated_num=_cum)
    elif args.modify == 'noise':
        mean = float(args.mean)
        std = np.sqrt(float(args.var)) * torch.ones_like(flat_input_grad.data).detach().clone()

        flat_noise = energy * torch.normal(mean=mean, std=std).to(**setup)
        
        noise = flat_recover_vector(flat_noise, func='recover', shapes=_shape, cumulated_num=_cum)
        for i, input_grad in enumerate(input_gradient):
            input_gradient[i] = input_gradient[i] + noise[i]
    else:
        raise NotImplementedError('Not implemented for args.modify = %s' % args.modify)
    def_stats['CosSim'] = grad_cosine_similarity(flat_input_grad, flat_noise).item()
    noise_norm = torch.norm(flat_noise)
    if noise_norm == 0:
        noise_norm += torch.finfo(torch.float32).eps
    def_stats['NoiseNorm'] = noise_norm.item()
    def_stats['NormRatio'] = input_grad_norm / noise_norm.item()
    noisy_flat_input_grad, _shape, _cum = flat_recover_vector(input_gradient, func='flat')
    def_stats['NoisyInputNorm'] = torch.norm(noisy_flat_input_grad).item()

    return input_gradient, noise, def_stats

import os, sys, time, math
import numpy as np
import torch
from torch import optim
from torch import nn
from matplotlib import pyplot as plt
from copy import deepcopy
from typing import Tuple, List
from scipy.sparse.linalg import eigsh, LinearOperator

def visualize_img_tensor(images, name, index = 0):
    img = images[index].data.detach().cpu().numpy()
    img = (img*255).astype(np.int)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.transpose(img, (1,2,0)))
    plt.show()
    plt.savefig('%s.png'%(name))

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
            cumulated_num.append(torch.prod(torch.tensor(tensor.shape)))
        return flat_vector, shapes, cumulated_num
    elif func == 'recover':
        # reshape the vector:
        recovered_tensor = []
        cum = 0
        for size_idx, size in enumerate(cumulated_num):
            recovered_tensor.append(vector[cum:cum+size].reshape(shapes[size_idx]))
            cum += size
        return recovered_tensor
    else: raise NotImplementedError('Not implement %s' % func)

class dotdict(dict):
    '''dot.notation access to dictionary attributes'''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def insert(self, key, value):
        self[key] = value

class myLogger():
    def __init__(self, logger):
        self.logger = logger
    def info(self, message):
        if self.logger:
            self.logger.info(message)
    def debug(self, message):
        if self.logger:
            self.logger.debug(message)
    def warning(self, message):
        if self.logger:
            self.logger.warning(message)

class AverageMeter:
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg

def Acc(targets, preds):
    '''
    PyTorch operation: Accuracy. 

    Args:
        targets: Tensor. Ground truth targets of data.
        preds: Tensor. Predictions on data.
    
    Returns:
        acc: float
    '''
    correct = preds.eq(targets.view_as(preds)).sum().item()
    total = torch.numel(preds)
    acc = correct / total
    return acc

def FPR(targets, preds):
    '''
    PyTorch operation: False positive rate. 

    Args:
        targets: Tensor. Ground truth targets of data.
        preds: Tensor. Predictions on data.
    
    Returns:
        FPR: float
    '''
    N = (targets == 0).sum().item() # negative sample number 
    FP = torch.logical_and(targets == 0, preds.squeeze() == 1).sum().item() # FP sample number
    FPR = FP/N
    return FPR

def FNR(targets, preds):
    '''
    PyTorch operation: False negative rate. 

    Args:
        targets: Tensor. Ground truth targets of data.
        preds: Tensor. Predictions on data.

    Returns:
        FNR: float
    '''
    P = (targets == 1).sum().item() # positive sample number 
    FN = torch.logical_and(targets == 1, preds.squeeze() == 0).sum().item() # FP sample number
    FNR = FN/P
    return FNR

def F1score(targets, preds):
    TP = torch.logical_and(targets == 1, preds.squeeze() == 1).sum().item() # TP sample number
    TN = torch.logical_and(targets == 0, preds.squeeze() == 0).sum().item() # TN sample number
    FP = torch.logical_and(targets == 0, preds.squeeze() == 1).sum().item() # FP sample number
    FN = torch.logical_and(targets == 1, preds.squeeze() == 0).sum().item() # FP sample number
    F1score = TP/(TP+0.5*FP+0.5*FN)
    return F1score

def count_model_parameters(model:nn.Module):
    count = 0
    for param in model.parameters():
        count += param.numel()
    return count

class JVPFunc_v1(nn.Module):
    def __init__(self, outputs1, outputs2, inputs1, inputs2):
        super().__init__()
        self.inputs = [inputs1, inputs2]
        self.gradients = [
            torch.autograd.grad(outputs1, inputs1, create_graph=True, retain_graph=True),
            torch.autograd.grad(outputs2, inputs2, create_graph=True, retain_graph=True)
        ]

    def forward(self, v, order=0):
        gradient = self.gradients[order]
        inputs = self.inputs[(order+1) % 2]

        elemwise_products = 0
        for grad_elem, v_elem in zip(gradient, v):
            elemwise_products += torch.sum(grad_elem * v_elem)
        
        return_grads = torch.autograd.grad(elemwise_products, inputs, create_graph=True, retain_graph=True)
        return return_grads

    def two_jvp_v1(self, v):
        jvp1 = self.forward(v, order=1)  # shape: d_x
        jvp2 = self.forward([jvp1_t.detach() for jvp1_t in jvp1], order=0)
        return jvp2

def max_eigen_new(x, y, model, loss_fh, dev):
    loss = loss_fh(model(x), y)
    loss1 = loss_fh(model(x), y)
    jvp = JVPFunc_v1(loss, loss1, [x], list(model.parameters()))
    shape = [p.shape for p in model.parameters()]
    eigen_vec = power_iteration(jvp.two_jvp_v1, shape, dev, num_iterations=100)
    loss = np.sum([torch.sum(v1*v2).item() for v1, v2 in zip(eigen_vec, jvp.two_jvp_v1(eigen_vec))])
    print(f"max_eigen loss: {loss}")
    return eigen_vec

def max_eigen(x, y, model, loss_fh, dev):
    # print('Compute max eigen')
    def two_jvp(v):
        loss = loss_fh(model(x), y)
        jvp1 = jvp(loss, [x], list(model.parameters()), v)  # shape: d_x
        loss = loss_fh(model(x), y)
        jvp2 = jvp(loss, model.parameters(), [x], [jvp1_t.detach() for jvp1_t in jvp1])  # shpe: d_param
        return jvp2
    # size = np.sum([np.prod(p.shape) for p in model.parameters()])
    shape = [p.shape for p in model.parameters()]
    eigen_vec = power_iteration(two_jvp, shape, dev, num_iterations=100)
    loss = np.sum([torch.sum(v1*v2).item() for v1, v2 in zip(eigen_vec, two_jvp(eigen_vec))])
    print(f"max_eigen loss: {loss}")
    return eigen_vec

def normalize(b_k1):
    b_k1_norm = np.sqrt(np.sum([(torch.norm(b_k_t)**2).item() for b_k_t in b_k1]))
    normed_b_k1 = [b_k1_t / b_k1_norm for b_k1_t in b_k1]
    return normed_b_k1, b_k1_norm

def power_iteration(compute_jvp, shapes, dev, num_iterations=100):
    '''
        A: a tensor (matrix) with shape (N * M)
    '''
    # b_k = torch.randn((size, 1))
    b_k = [torch.randn(s).to(dev) for s in shapes]
    b_k, _ = normalize(b_k)
    # with torch.no_grad():
    for t in range(num_iterations):
        # b_k1 = torch.mm(A, b_k)
        b_k1 = compute_jvp(b_k)
        normed_b_k1, b_k1_norm = normalize(b_k1)
        # b_k1_norm = np.sqrt(np.sum([(torch.norm(b_k_t)**2).item() for b_k_t in b_k1]))
        # normed_b_k1 = [b_k1_t / b_k1_norm for b_k1_t in b_k1]
        diff = np.sum([(torch.norm(b1-b2)**2).item() for b1, b2 in zip(normed_b_k1, b_k)])
        b_k = normed_b_k1
        # print(f"[{t}] diff: {diff}, b_k1_norm: {b_k1_norm}")
        if diff < 1e-12:
            continue
        # b_k = nn.functional.normalize(b_k1)
        # b_k1_norm = torch.norm(b_k1)
        # b_k = b_k1 / b_k1_norm
    return b_k

def jvp(outputs, inputs1, inputs2, v, disp=False):
    """
    Return: Shape the same as inputs1.
    """
    for inp, v_elm in zip(inputs2, v):
        assert inp.shape == v_elm.shape
    gradient = torch.autograd.grad(outputs, inputs2, create_graph=True, retain_graph=True)
    elemwise_products = 0
    for grad_elem, v_elem in zip(gradient, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    return_grads = torch.autograd.grad(elemwise_products, inputs1)
    disp_grads = [grad.detach().clone() for grad in return_grads]
    if disp:
        for grad in disp_grads:
            print(grad.shape)
        print('==============')
    return return_grads

def compute_exact_bound(args, model:nn.Module, img:torch.Tensor, label:torch.Tensor, noise:List[torch.Tensor], setup:dict, regu:float):
    # bound_log = open('./log', 'w+')
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    img.requires_grad = True
    b = torch.randn_like(img).detach().clone().to(**setup)
    deltas = []
    losses = []
    # lr = 0.1
    lr = 0.01
    iterations = 2000
    # iterations = 500
    if args.model == 'linear':
        # lr = 0.001
        iterations = 2000
    elif args.model in ['ResNet32-10', 'ResNet56']:
        lr = 1e-8
    lr_decay = 1
    for t in range(iterations):
        model.zero_grad()
        loss = criterion(model(img), label)
        JT_times_b = jvp(loss, list(model.parameters()), [img], [b])
        delta = []
        for noise_ele, Jb_ele in zip(noise, JT_times_b):
            delta.append(noise_ele - Jb_ele)
        model.zero_grad()
        loss = criterion(model(img), label)
        delta = -1 * jvp(loss, [img], list(model.parameters()), delta)[0]
        delta = delta + regu * b

        deltas.append(torch.norm(delta).cpu().item())

        b = b - lr*(lr_decay**t) * delta

        grad_gt = deepcopy(img)
        grad_gt.requires_grad = True
        model.zero_grad()
        loss = criterion(model(grad_gt), label)
        Jb = jvp(loss, list(model.parameters()), [grad_gt], [b])
        norm = 0.
        for Jb_ele, noise_ele in zip(Jb, noise):
            norm += (Jb_ele-noise_ele).pow(2).sum().item()
        losses.append(np.sqrt(norm))
        if t % 100 == 0: print(f"T: {t}  delta: {deltas[-1]}  Loss: {losses[-1]}")
        if (deltas[-1] < 9e-5): break
        if math.isnan(losses[-1]): break
    return b

# utilities for parameters
def foreach(p1, p2, op, alpha:float=1) -> List[torch.Tensor]:
    # this may be replaced with nested_tensor in the future
    if len(p1) != len(p2):
        # need more sophisticated check
        raise ValueError('p1 and p2 must have the same structure')
    return tuple(op(p1, alpha * p2) for p1, p2 in zip(p1, p2))

def get_Jacob(model:nn.Module, img:torch.Tensor, label:torch.Tensor, setup:dict) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    x_ = img.reshape(-1)
    x_.requires_grad = True
    x = x_.reshape(img.shape)
    loss = criterion(model(x), label)
    jacobi_time = time.time()
    gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    J = torch.tensor([]).to(**setup)
    for grad in gradients:
        if len(grad.shape) == 1:
            for i in range(grad.shape[0]):
                J_i = torch.autograd.grad(grad[i], x_, retain_graph=True, create_graph=True)[0]
                J = torch.cat((J, J_i.unsqueeze(0)))
        elif len(grad.shape) == 2:
            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    J_ij = torch.autograd.grad(grad[i, j], x_, retain_graph=True, create_graph=True)[0]
                    J = torch.cat((J, J_ij.unsqueeze(0)))
        elif len(grad.shape) == 4:
            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    for k in range(grad.shape[2]):
                        for m in range(grad.shape[3]):
                            J_ijkm = torch.autograd.grad(grad[i, j, k, m], x_, retain_graph=True, create_graph=True)[0]
                            J = torch.cat((J, J_ijkm.unsqueeze(0)))
    jacobi_time = time.time() - jacobi_time
    # print(f"Computing Jacob for {jacobi_time}s")
    return J.detach()

def max_eigen_val(model, img, label, device) -> np.ndarray:
    # dim = np.sum([np.prod(p.shape) for p in model.parameters()])
    dim = np.sum([p.numel() for p in model.parameters()])

    def two_jvp(x:List[torch.Tensor], model, img, label,):
        criterion = torch.nn.CrossEntropyLoss()
        model.zero_grad()
        loss = criterion(model(img), label)
        jvp1 = jvp(loss, [img], list(model.parameters()), x)  # shape: d_param
        loss = criterion(model(img), label)
        jvp2 = jvp(loss, list(model.parameters()), [img], [jvp1_t.detach() for jvp1_t in jvp1])  # shpe: d_img
        return jvp2

    def mv(v):
        """assume v is of shape [d, 1] or [d,] (has to handle both cases.)"""
        # JVP
        v_t = torch.from_numpy(v).to(device)
        shapes = [p.shape for p in model.parameters()]
        # cumulated_num = [np.prod(p.shape) for p in model.parameters()]
        cumulated_num = [p.numel() for p in model.parameters()]
        v_t_list = flat_recover_vector(
            v_t, func='recover', shapes=shapes, cumulated_num=cumulated_num)
        for v_t_list_el in v_t_list:
            v_t_list_el.requires_grad_(True)
        ret = two_jvp(v_t_list, model, img, label)
        ret, _, _ = flat_recover_vector(ret, func='flat')
        return ret.data.cpu().numpy()

    A = LinearOperator((dim, dim), matvec=mv)
    eigenval, eigenvec = eigsh(A, k=1)
    return eigenval

def I2F_lb(model:nn.Module, img:torch.Tensor, label:torch.Tensor, noise:List[torch.Tensor], device:str='cuda:0') -> torch.Tensor:
    '''
        noise: the list of tensors; should be the same shape as the model parameters
        img: input image
        label: corresponding label of input image
    '''
    criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    grad_gt = deepcopy(img)
    grad_gt.requires_grad = True
    loss = criterion(model(grad_gt), label)
    
    JtDelta = jvp(loss, [grad_gt], list(model.parameters()), noise)  # shape: d_x
    JtDelta = flat_recover_vector(JtDelta, func='flat')[0].detach()
    JtDelta = JtDelta.pow(2).sum().sqrt()

    _inputs = deepcopy(img)
    _inputs.requires_grad = True
    max_eigen_value = max_eigen_val(model, _inputs, label, device)

    return JtDelta.cpu() / torch.tensor(max_eigen_value)

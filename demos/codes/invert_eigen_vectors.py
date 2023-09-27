'''
    invert with different eigen vectors
'''
import os, sys, math, random, argparse, time
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import wandb

import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from torch.distributions.laplace import Laplace

import clip
from skimage.metrics import structural_similarity
from sklearn.decomposition import PCA
import inversefed
from inversefed.data.loss import Classification
from metrics import ssim_batch
import myreconstruction
import defense
from defense import *
import baseline_utils
import utils
from scipy.sparse.linalg import LinearOperator, eigsh

import warnings
warnings.filterwarnings('ignore')

data_root = './data'

def jvp(outputs, inputs1, inputs2, v, disp=False):
    """
    Return: Shape the same as inputs1.
    """
    for inp, v_elm in zip(inputs2, v):
        assert inp.shape == v_elm.shape, f"Mismatched dimensions: {inp.shape}, {v_elm.shape}"
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

def two_jvp(x:List[torch.Tensor], model, img, label,):
    criterion = torch.nn.CrossEntropyLoss()
    model.zero_grad()
    loss = criterion(model(img), label)
    jvp1 = jvp(loss, [img], list(model.parameters()), x)  # shape: d_param
    loss = criterion(model(img), label)
    jvp2 = jvp(loss, list(model.parameters()), [img], [jvp1_t.detach() for jvp1_t in jvp1])  # shpe: d_img
    return jvp2

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

def set_random_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def load_model() -> nn.Module:
    model_path = {
        'ResNet18-cifar10': f'{data_root}/results_exactbound/conv5_cifar10_sgd.pt',
        'ResNet18-mnist': f'{data_root}/results_exactbound/conv5_mnist.pt',
        'conv5-cifar10': f'{data_root}/results_exactbound/conv5_cifar10.pt',
        'conv5-mnist': f'{data_root}/results_exactbound/conv5_mnist_sgd.pt',
    }
    print(f"Load model from {args.model}-{args.dataset}")
    model = torch.load(model_path[f'{args.model}-{args.dataset}'], map_location=setup['device'])['model']
    return model

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

def batch_recover(model:nn.Module, ground_truth:torch.Tensor, labels:torch.Tensor, results_dir, log_dir, images_dir, figure_dir, input_gradient, eigen_vec):
    model.zero_grad()
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    bound_coefs = None
    if args.stage == 'pure':
        target_loss = loss_fn(model(ground_truth), labels)
        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        flat_input_grad, _shape, _cum = flat_recover_vector(input_gradient, func='flat')
        input_grad_norm = torch.norm(flat_input_grad).item()
        norm_ratio, cossim, noise_norm = 0., 0., 0.
    elif args.stage == 'defend':
        # compute max eigen value:
        '''
            ||g_0||: input_grad_norm; \epsilon: MSE;
            \alpha||J*\delta||: JtDelta; ||\alpha\delta||: noise_norm;
        '''
        flat_input_grad, _shape, _cum = flat_recover_vector(input_gradient, func='flat')
        input_grad_norm = torch.norm(flat_input_grad)
        noise_norm = torch.norm(eigen_vec)
        noise_norm = torch.norm(eigen_vec)
        noisy_input_grad = flat_input_grad + eigen_vec
        noisy_input_grad_norm = torch.norm(noisy_input_grad)
        noisy_input_grad = flat_recover_vector(noisy_input_grad, func='recover', shapes=_shape, cumulated_num=_cum)
        norm_ratio = input_grad_norm / noise_norm
        cossim = grad_cosine_similarity(flat_input_grad, eigen_vec)

    config = dict(signed=False if args.inv_goal == 'l2' else True,
                boxed=False if args.inv_goal == 'l2' else True,
                cost_fn=args.inv_goal,
                indices='def',
                weights='equal',
                lr=args.inv_lr,
                optim=args.inv_optimizer,
                restarts=1,
                max_iterations=args.inv_iterations,
                total_variation=args.tv_loss,
                init='randn',
                filter='none',
                lr_decay=False if args.inv_goal == 'l2' else True,
                scoring_choice='loss',
                results_dir=results_dir,
                log_dir=log_dir,
                images_dir=images_dir,
                figure_dir=figure_dir,
                random_idx=-1,
                state='running')
    start_time = time.time()
    rec_machine = myreconstruction.GradientReconstructor(model, deepcopy(ground_truth).detach(), (dm, ds), config, num_images=len(labels))
    output, inv_stats = rec_machine.reconstruct(noisy_input_grad, labels, img_shape=ground_truth.shape[1:])
    end_time = (time.time() - start_time) / 60

    if args.normalize:
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        ground_truth_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
    else:
        output_denormalized = torch.clamp(output, 0, 1)
        ground_truth_denormalized = torch.clamp(ground_truth, 0, 1)
    
    test_mse = (output.detach() - ground_truth).pow(2).mean() # original of Geiping
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds) # original of Geiping
    mean_ssim, batch_ssims = ssim_batch(output, ground_truth) # original of Geiping
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()

    inv_metrics = {
        'MSE': test_mse.item(),
        'PSNR': test_psnr,
        'RecLoss': inv_stats['opt'],
        'FMSE': feat_mse.item(),
        'SSIM': mean_ssim,
        'CosSim': cossim.item(),
        'InputGradNorm': input_grad_norm.item(),
        'NoiseNorm': noise_norm.item(),
        'NormRatio': norm_ratio.item(),
    }

    print(f"Rec. loss: {inv_stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} "
            f"| Time: {end_time:4.2f}")
    step_info = ''
    if args.stage == 'defend':
        step_info = f'NoiseNorm: {noise_norm} | InputNorm: {input_grad_norm} | NoisyInputNorm: {noisy_input_grad_norm} | MSE: {test_mse.item()}\n'
    return output, inv_metrics, step_info, norm_ratio

def invert():
    invert_result_dir = os.path.join(result_dir, '%s-%s-seed%d'%(args.model, args.dataset, args.seed))
    os.makedirs(invert_result_dir, exist_ok=True)
    # invert_result_dir = result_dir
    inversion_log_file = os.path.join(invert_result_dir, 'log.log')
    bound_log_file = os.path.join(invert_result_dir, 'bound.log')
    log = open(inversion_log_file, 'w+')
    bound_log = open(bound_log_file, 'w+')
    log.write(f'{args}\n')
    log.flush()
    bound_log.write(f'{args}\n')
    bound_log.flush()

    model.to(**setup)
    model.eval()
    # select samples from the index list
    print('Start inverting')
    for i in range(args.num_inversion_batches):
        ground_truth, labels = [], []
        img, label = test_set[i+args.seed]
        labels.append(torch.as_tensor((label,), device=setup['device']))
        ground_truth.append(img.to(**setup))
        mixed_img = torch.stack(ground_truth)
        mixed_labels = torch.cat(labels)

        img = deepcopy(mixed_img)
        label = deepcopy(mixed_labels)
        model.eval()
        img.requires_grad = True

        dim = np.sum([np.prod(p.shape) for p in model.parameters()])

        def mv(v):
            """assume v is of shape [d, 1] or [d,] (has to handle both cases.)"""
            # JVP
            v_t = torch.from_numpy(v).to(**setup)
            shapes = [p.shape for p in model.parameters()]
            cumulated_num = [np.prod(p.shape) for p in model.parameters()]
            v_t_list = flat_recover_vector(
                v_t, func='recover', shapes=shapes, cumulated_num=cumulated_num)
            for v_t_list_el in v_t_list:
                v_t_list_el.requires_grad_(True)
            ret = two_jvp(v_t_list, model, img, label)
            ret, _, _ = flat_recover_vector(ret, func='flat')
            return ret.data.cpu().numpy()

        A = LinearOperator((dim, dim), matvec=mv)
        eigenvalues, eigenvectors = eigsh(A, k=100, which='LM') # you may choose your own k here based on your device capacity

        model.zero_grad()
        loss = criterion(model(mixed_img), mixed_labels)
        input_gradient = torch.autograd.grad(loss, model.parameters())
        input_gradient = [grad.detach().clone() for grad in input_gradient]

        if args.normalize:
            ground_truth_denormalized = torch.clamp(mixed_img * ds + dm, 0, 1)
        else:
            ground_truth_denormalized = mixed_img

        start_time = time.time()
        '''
            start batch recover and defense:
        '''
        lins = np.linspace(0, eigenvalues.shape[0]-1, num=4, dtype=np.int32, endpoint=True)
        for inv_idx in lins:
            images_dir = os.path.join(invert_result_dir, 'images/', 'batch%d_inv%d'%(i, inv_idx))
            img_pure_dir = os.path.join(invert_result_dir, 'figures/', 'batch%d_inv%d'%(i, inv_idx))
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(img_pure_dir, exist_ok=True)

            torchvision.utils.save_image(ground_truth_denormalized, os.path.join(images_dir, 'img%d_gt.png'%(i)), nrow=5)
            output, inv_metrics, step_info, norm_ratio = \
                batch_recover(model, mixed_img, mixed_labels, invert_result_dir, 'log_dir', images_dir, img_pure_dir, input_gradient, 5*torch.from_numpy(eigenvectors[:, inv_idx]).to(**setup))

            end_time = (time.time() - start_time) / 60 # minute
            if args.normalize:
                output_denormalized = torch.clamp(output * ds + dm, 0, 1)
            else:
                output_denormalized = torch.clamp(output, 0, 1)
            torchvision.utils.save_image(output_denormalized, os.path.join(images_dir, 'img%d_output.png'%(i)), nrow=5)

            recover_info = f"Batch: {i} | Idx: {inv_idx} | EigenValue: {eigenvalues[inv_idx].item()} | Rec. loss: {inv_metrics['RecLoss']:2.4f} | MSE: {inv_metrics['MSE']:2.4f} | SSIM: {inv_metrics['SSIM']:2.4f}\n"
            print(recover_info)
            log.write(recover_info)
            bound_log.write(f"Batch: {i} | Idx: {inv_idx} | " + step_info)
            log.flush()
            bound_log.flush()

    log.flush()
    log.close()

def main():
    invert()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic settings:
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--baseline', type=str, default='debug', choices=['debug', 'run'], help='the baseline whose setting is used')
    parser.add_argument('--model', type=str, default='conv5', )
    parser.add_argument('--dataset', type=str, default='mnist', choices=['cifar10', 'cifar100', 'mnist', 'mmnist', 'fmnist'])
    parser.add_argument('--epoch', type=int, default=0, choices=[0, 50, 100, 150], help='epoch choice to load checkpoint')
    # params for inversion:
    parser.add_argument('--num_inversion_batches', type=int, default=100)
    parser.add_argument('--num_img_per_invbatch', type=int, default=1)
    parser.add_argument('--inv_goal', type=str, default='l2', choices=['l2', 'l1', 'sim'], help='the loss function used in GI attacks')
    parser.add_argument('--inv_iterations', type=int, default=24000)
    parser.add_argument('--inv_optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--inv_lr', type=float, default=0.1, help='from Geiping: 0.1 for GS and 1e-4 for DLG')
    parser.add_argument('--tv_loss', type=float, default=0., help='tv loss coefficient')
    # some actions:
    parser.add_argument('--uniform_init', action='store_true', help='whether to use uniform initialization of model')
    parser.add_argument('--trained_model', action='store_true', help='whether to use uniform initialization of model')
    # params for modifying gradients:
    parser.add_argument('--stage', type=str, default='defend', choices=['pure', 'defend'], help='whether to modify the gradients')
    parser.add_argument('--modify', type=str, default='noise', choices=['noise'], help='use which method to modify the gradients; "zero" means set grad as zeros; "random" means set grad as random noise')
    # params for noise control:
    parser.add_argument('--energy', type=float, default=1, help='noise energy control or the scaling factor when modify==scale; should be deactive if args.project is True; also the noise multiplier of DP')
    args = parser.parse_args()

    '''env settings:'''
    set_random_seed(args.seed)
    defs = inversefed.training_strategy('conservative')
    setup = inversefed.utils.system_startup(gpu=args.gpu)

    '''set dataset:'''
    # number of classes, img size, data shape:
    args.num_classes = 10
    if args.dataset == 'cifar100':
        args.num_classes = 100
    if args.dataset in ['cifar10', 'cifar100']:
        resize = (32, 32)
        data_shape = (3, 32, 32)
    else:
        resize = (28, 28)
        data_shape = (1, 28, 28)
    # num of img channels:
    if args.dataset in ['mnist', 'fmnist', 'mmnist']: args.num_channels = 1
    else: args.num_channels = 3
    # data loaders
    args.train_bsz = 1024
    args.train_lr = 0.1
    args.normalize = False
    if args.model == 'ResNet18': args.normalize = True
    if args.inv_goal == 'sim': args.normalize = True
    train_set, test_set, trainloader, testloader, dm, ds = baseline_utils.load_datasets(args, args.dataset, val=False, resize=resize, normalize=args.normalize)
    # compute the mean and std of training dataset:
    dm = torch.as_tensor(dm, **setup)[:, None, None]
    ds = torch.as_tensor(ds, **setup)[:, None, None]
    
    '''set modules:'''
    if args.trained_model:
        model = load_model()
    else:
        model = baseline_utils.load_model(args.model, num_classes=args.num_classes, num_channels=args.num_channels, data_shape=data_shape)
    if args.uniform_init:
        model.apply(weights_init)
    model.to(**setup)
    criterion = nn.CrossEntropyLoss()

    result_dir = 'results_invert_eigen_vec'
    result_dir = os.path.join(data_root, result_dir)

    print(args)
    main()


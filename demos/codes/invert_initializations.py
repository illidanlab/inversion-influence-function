'''
    Invert with different init methods;
    Run the figure of I2F vs. MSE
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

import lpips
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

import warnings
warnings.filterwarnings('ignore')

data_root = './data'

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

def model_init(init):
    def uniform_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    def normal_init(m):
        if hasattr(m, "weight"):
            m.weight.data.normal_(0, 0.5)
        if hasattr(m, "bias"):
            m.weight.data.normal_(0, 0.5)
    def kaiming_uniform(m):
        if hasattr(m, "weight"):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='sigmoid')
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    def xavier_uniform(m):
        if hasattr(m, "weight"):
            nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    if init == 'uniform': return uniform_init
    elif init == 'normal': return normal_init
    elif init == 'xavier': return xavier_uniform
    elif init == 'kaiming': return kaiming_uniform

def batch_recover(model:nn.Module, ground_truth:torch.Tensor, labels:torch.Tensor, results_dir, log_dir, images_dir, figure_dir, input_gradient, max_eigen_value, mean_lambda_inv):
    model.zero_grad()
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    bound_coefs = None
    if args.stage == 'pure':
        # target_loss = loss_fn(model(ground_truth), labels)
        # input_gradient = torch.autograd.grad(target_loss, model.parameters())
        # input_gradient = [grad.detach() for grad in input_gradient]
        flat_input_grad, _shape, _cum = flat_recover_vector(input_gradient, func='flat')
        input_grad_norm = torch.norm(flat_input_grad).item()
        norm_ratio, cossim, noise_norm = 0., 0., 0.
    elif args.stage == 'defend':
        flat_input_grad, _shape, _cum = flat_recover_vector(input_gradient, func='flat')
        flat_noise = args.energy * torch.normal(mean=0, std=float(args.var), size=flat_input_grad.shape).to(**setup)
        noise = flat_recover_vector(flat_noise, func='recover', shapes=_shape, cumulated_num=_cum)
        # compute max eigen value:
        '''
            ||g_0||: input_grad_norm; \epsilon: MSE;
            \alpha||J*\delta||: JtDelta; ||\alpha\delta||: noise_norm;
        '''
        model.zero_grad()
        grad_gt = deepcopy(ground_truth)
        grad_gt.requires_grad = True
        loss = criterion(model(grad_gt), labels)
        JtDelta = utils.jvp(loss, [grad_gt], list(model.parameters()), noise)  # shape: d_x
        JtDelta = flat_recover_vector(JtDelta, func='flat')[0].detach()
        JtDelta = JtDelta.pow(2).sum().sqrt()

        input_grad_norm = torch.norm(flat_input_grad)
        noise_norm = torch.norm(flat_noise)
        noisy_input_grad = flat_input_grad + flat_noise
        noisy_input_grad_norm = torch.norm(noisy_input_grad)
        noisy_input_grad = flat_recover_vector(noisy_input_grad, func='recover', shapes=_shape, cumulated_num=_cum)
        norm_ratio = input_grad_norm / noise_norm
        bound_coefs = {
            'JtDelta': JtDelta.item(),
            'NoiseNorm': noise_norm,
            'InputNorm': input_grad_norm,
            'NoisyInputNorm': noisy_input_grad_norm,
            'MaxEigenValue': max_eigen_value.item(),
        }

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
    
    test_mse = (output.detach() - ground_truth).pow(2).mean() # original of Geiping
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds) # original of Geiping
    mean_ssim, batch_ssims = ssim_batch(output, ground_truth) # original of Geiping
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()

    # lpips:
    with torch.no_grad():
        if args.dataset != 'mnist':
            lpips_alexnet = alexnet.forward(ground_truth.detach().squeeze(0).to(setup['device']), output.detach().squeeze(0).to(setup['device']))
        else:
            lpips_alexnet = torch.tensor(-1)
        lpips_vgg = vgg.forward(ground_truth.detach().squeeze(0).to(setup['device']), output.detach().squeeze(0).to(setup['device']))
        lpips_squeeze = squeeze.forward(ground_truth.detach().squeeze(0).to(setup['device']), output.detach().squeeze(0).to(setup['device']))

    inv_metrics = {
        'MSE': test_mse.item(),
        'PSNR': test_psnr,
        'RecLoss': inv_stats['opt'],
        'FMSE': feat_mse.item(),
        'SSIM': mean_ssim,
        'InputGradNorm': input_grad_norm.item(),
        'NoiseNorm': noise_norm.item(),
        'NormRatio': norm_ratio.item(),
    }
    if args.stage == 'defend':
        bound_coefs['MSE'] = test_mse.item()

    print(f"Rec. loss: {inv_stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} "
            f"| Time: {end_time:4.2f}")
    step_info = ''
    if args.stage == 'defend':
        step_info = f'JtDelta: {JtDelta.item()} | MaxEigenValue: {max_eigen_value.item()} | MeanInvLambda: {mean_lambda_inv.item()} | MSE: {test_mse.item()}\n'
    return output, inv_metrics, step_info, norm_ratio

def invert():
    invert_result_dir = os.path.join(result_dir, '%s-%s-var%s'%(args.model, args.dataset, args.var))
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
    seeds = [1, 123, 12345, 234, 345]
    for i in range(args.num_inversion_batches):
        # log_dir = os.path.join(invert_result_dir, 'logs/')
        eigenvalue_dir = os.path.join(invert_result_dir, 'eigenvalues/')
        os.makedirs(eigenvalue_dir, exist_ok=True)

        ground_truth, labels = [], []
        img, label = test_set[i]
        labels.append(torch.as_tensor((label,), device=setup['device']))
        ground_truth.append(img.to(**setup))
        mixed_img = torch.stack(ground_truth)
        mixed_labels = torch.cat(labels)

        J = utils.get_Jacob(model, mixed_img, mixed_labels, setup)
        svd_time = time.time()
        U, S, Vh = torch.linalg.svd(J.transpose(0,1).detach(), full_matrices=False)
        svd_time = time.time() - svd_time
        print(f"Computing SVD for {svd_time}s")
        torch.save({
            'img': mixed_img.detach().cpu(),
            'label': mixed_labels.detach().cpu(),
            'init': args.init,
            'U': U.detach().cpu(),
            'S': S.detach().cpu(),
            'Vh': Vh.detach().cpu(),
        }, os.path.join(eigenvalue_dir, f'batch{i}.pt'))
        mean_lambda_inv = (1/S).mean()

        model.zero_grad()
        loss = criterion(model(mixed_img), mixed_labels)
        input_gradient = torch.autograd.grad(loss, model.parameters())
        input_gradient = [grad.detach().clone() for grad in input_gradient]

        start_time = time.time()
        '''
            start batch recover and defense:
        '''
        for inv_idx in range(5):
            images_dir = os.path.join(invert_result_dir, 'images/', f"batch{i}", f"inv{inv_idx}")
            img_pure_dir = os.path.join(invert_result_dir, 'figures/', f"batch{i}", f"inv{inv_idx}")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(img_pure_dir, exist_ok=True)
            if args.normalize:
                ground_truth_denormalized = torch.clamp(mixed_img * ds + dm, 0, 1)
            else:
                ground_truth_denormalized = mixed_img
            torchvision.utils.save_image(ground_truth_denormalized, os.path.join(images_dir, 'img%d_gt.png'%(i)), nrow=5)

            set_random_seed(seeds[inv_idx])
            output, inv_metrics, step_info, norm_ratio = \
                batch_recover(model, mixed_img, mixed_labels, invert_result_dir, 'log_dir', images_dir, img_pure_dir, input_gradient, max_eigen_value=S[0], mean_lambda_inv=mean_lambda_inv)

            end_time = (time.time() - start_time) / 60 # minute
            if args.normalize:
                output_denormalized = torch.clamp(output * ds + dm, 0, 1)
            else:
                output_denormalized = torch.clamp(output, 0, 1)
            torchvision.utils.save_image(output_denormalized, os.path.join(images_dir, 'img%d_output.png'%(i)), nrow=5)
            
            recover_info = f"Batch: {i} | Idx: {inv_idx} | MeanInvLambda: {mean_lambda_inv.item()} | Rec. loss: {inv_metrics['RecLoss']:2.4f} | MSE: {inv_metrics['MSE']:2.4f} | SSIM: {inv_metrics['SSIM']:2.4f}\n"
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
    parser.add_argument('--dataset', type=str, default='mnist', choices=['cifar10', 'cifar100', 'mnist'])
    parser.add_argument('--epoch', type=int, default=0, choices=[0, 50, 100, 150], help='epoch choice to load checkpoint')
    # params for inversion:
    parser.add_argument('--num_inversion_batches', type=int, default=5)
    parser.add_argument('--num_img_per_invbatch', type=int, default=1)
    parser.add_argument('--inv_goal', type=str, default='l2', choices=['l2', 'l1', 'sim'], help='the loss function used in GI attacks')
    parser.add_argument('--inv_iterations', type=int, default=24000)
    parser.add_argument('--inv_optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--inv_lr', type=float, default=0.1, help='from Geiping: 0.1 for GS and 1e-4 for DLG')
    parser.add_argument('--tv_loss', type=float, default=0., help='tv loss coefficient')
    # some actions:
    parser.add_argument('--init', type=str, default='kaiming', choices=['uniform', 'kaiming', 'xavier', 'normal'], help='model initialization')
    # parser.add_argument('--trained_model', action='store_true', help='whether to use uniform initialization of model')
    # params for modifying gradients:
    parser.add_argument('--stage', type=str, default='defend', choices=['pure', 'defend'], help='whether to modify the gradients')
    parser.add_argument('--modify', type=str, default='noise', choices=['noise', 'scale', 'zeros', 'random', 'svd', 'clipsvd', 'pga', 'soteria', 'dp'], help='use which method to modify the gradients; "zero" means set grad as zeros; "random" means set grad as random noise')
    # params for noise control:
    parser.add_argument('--energy', type=float, default=1, help='noise energy control or the scaling factor when modify==scale; should be deactive if args.project is True; also the noise multiplier of DP')
    parser.add_argument('--var', type=str, default='1e-3', help='noise energy control or the scaling factor when modify==scale; should be deactive if args.project is True; also the noise multiplier of DP')
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
    model = baseline_utils.load_model(args.model, num_classes=args.num_classes, num_channels=args.num_channels, data_shape=data_shape)
    model.apply(model_init(args.init))
    model.to(**setup)
    criterion = nn.CrossEntropyLoss()

    # lpips:
    alexnet = lpips.LPIPS(net='alex').to(setup['device'])
    vgg = lpips.LPIPS(net='vgg').to(setup['device'])
    squeeze = lpips.LPIPS(net='squeeze').to(setup['device'])

    result_dir = 'results_invert_inits'
    result_dir = os.path.join(data_root, result_dir)
    wandb.init(project='I2F', name='invert-init', config=vars(args), mode='online')

    print(args)
    main()


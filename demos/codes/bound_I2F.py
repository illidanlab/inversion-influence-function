'''
    Calculate the I2F with noise into gradient.
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

import clip, lpips
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

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

def batch_recover(model:nn.Module, ground_truth:torch.Tensor, labels:torch.Tensor, results_dir, log_dir, images_dir, figure_dir, i):
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
    # defend:
    elif args.stage == 'defend':
        input_gradient, noise, def_stats = defend(args, model, ground_truth, labels, data_shape, setup)
        # compute I2F:
        '''
            ||g_0||: input_grad_norm;
            \epsilon: MSE;
            ||J*\delta||: JtDelta;
            ||\delta||: noise_norm;
        '''
        model.zero_grad()
        grad_gt = deepcopy(ground_truth)
        grad_gt.requires_grad = True
        loss = criterion(model(grad_gt), labels)
        JtDelta = utils.jvp(loss, [grad_gt], list(model.parameters()), noise)  # shape: d_x
        JtDelta = flat_recover_vector(JtDelta, func='flat')[0].detach()
        JtDelta = JtDelta.pow(2).sum().sqrt()

        _inputs = deepcopy(ground_truth)
        _inputs.requires_grad = True
        max_eigen_value = utils.max_eigen_val(model, _inputs, labels, setup['device'])

        input_grad_norm = def_stats['InputNorm']
        noise_norm = def_stats['NoiseNorm']
        noisy_input_grad_norm = def_stats['NoisyInputNorm']
        bound_coefs = {
            'JtDelta': JtDelta.item(),
            'NoiseNorm': noise_norm,
            'InputNorm': input_grad_norm,
            'NoisyInputNorm': noisy_input_grad_norm,
            'MaxEigenValue': max_eigen_value.item(),
        }
            
        input_grad_norm, noise_norm, norm_ratio, noisy_input_grad_norm = \
            def_stats['InputNorm'], def_stats['NoiseNorm'], def_stats['NormRatio'], def_stats['NoisyInputNorm']

    config = dict(signed=False if args.inv_goal == 'l2' else True,
                boxed=False if args.inv_goal == 'l2' else True,
                cost_fn=args.inv_goal,
                indices='def' if args.dataset == 'imagenet' else 'def',
                weights='equal',
                lr=args.inv_lr,
                optim=args.inv_optimizer,
                restarts=1,
                max_iterations=args.inv_iterations,
                total_variation=args.tv_loss,
                init='randn',
                filter= 'none' if args.dataset == 'imagenet' else 'none',
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
    output, inv_stats = rec_machine.reconstruct(input_gradient, labels, img_shape=ground_truth.shape[1:], mask=None)
    end_time = (time.time() - start_time) / 60 # minute
    
    test_mse = (output.detach() - ground_truth).pow(2).mean() # original of Geiping
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds) # original of Geiping
    mean_ssim, batch_ssims = ssim_batch(output, ground_truth) # original of Geiping
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()

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
        'InputGradNorm': input_grad_norm,
        'NoiseNorm': noise_norm,
        'NormRatio': norm_ratio,
        'batch': i,
        'alexnet': lpips_alexnet.item(),
        'vgg': lpips_vgg.item(),
        'squeeze': lpips_squeeze.item(),
    }
    if args.stage == 'defend':
        bound_coefs['MSE'] = test_mse.item()
        bound_coefs.update(inv_metrics)

    print(f"Rec. loss: {inv_stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | Time: {end_time:4.2f}")
    step_info = ''
    if args.stage == 'defend':
        step_info = f'JtDelta: {JtDelta.item()} | MaxEigenValue: {max_eigen_value.item()} | MSE: {test_mse.item()}' \
                f" | vgg: {lpips_vgg.item()} | SSIM: {mean_ssim}\n"
    return output, inv_stats, inv_metrics, step_info, norm_ratio, bound_coefs

def invert():
    invert_result_dir = os.path.join(result_dir, '-%s-%s-seed%d'%(args.model, args.dataset, args.seed))
    os.makedirs(invert_result_dir, exist_ok=True)
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
    inv_idx_iterator = range(args.num_inversion_batches)
    for i in inv_idx_iterator:
        images_dir = os.path.join(invert_result_dir, 'images/', 'batch%d'%i)
        img_pure_dir = os.path.join(invert_result_dir, 'figures/', 'batch%d'%i)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(img_pure_dir, exist_ok=True)

        img, label = test_set[i]
        mixed_labels = torch.as_tensor((label,), device=setup['device'])
        mixed_img = img.to(**setup).unsqueeze(0)

        ground_truth_denormalized = torch.clamp(mixed_img * ds + dm, 0, 1)
        torchvision.utils.save_image(ground_truth_denormalized, os.path.join(images_dir, 'img%d_gt.png'%(i)), nrow=5)

        start_time = time.time()
        '''
            start batch recover and defense:
        '''
        output, inv_stats, inv_metrics, step_info, norm_ratio, bound_coefs = \
            batch_recover(model, mixed_img, mixed_labels, invert_result_dir, 'log_dir', images_dir, img_pure_dir, i)

        end_time = (time.time() - start_time) / 60 # minute
        if args.normalize:
            output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        else:
            output_denormalized = torch.clamp(output, 0, 1)
        torchvision.utils.save_image(output_denormalized, os.path.join(images_dir, 'img%d_output.png'%(i)), nrow=5)
        
        recover_info = f"Batch: {i} | Idx: {i} | Rec. loss: {inv_metrics['RecLoss']:2.4f} | MSE: {inv_metrics['MSE']:2.4f} | SSIM: {inv_metrics['SSIM']:2.4f}\n"
        print(recover_info)
        log.write(recover_info)
        bound_log.write(step_info)
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
    parser.add_argument('--model', type=str, default='ResNet152', )
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--epoch', type=int, default=0, choices=[0, 50, 100, 150], help='epoch choice to load checkpoint')
    parser.add_argument('--use_wandb', action='store_true')
    # params for inversion:
    parser.add_argument('--num_inversion_batches', type=int, default=20)
    parser.add_argument('--num_img_per_invbatch', type=int, default=1)
    parser.add_argument('--indices', type=str, default='def')
    parser.add_argument('--inv_goal', type=str, default='sim', choices=['l2', 'l1', 'sim'], help='the loss function used in GI attacks')
    parser.add_argument('--inv_iterations', type=int, default=24000)
    parser.add_argument('--inv_optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'])
    parser.add_argument('--inv_lr', type=float, default=0.1, help='from Geiping: 0.1 for GS and 1e-4 for DLG')
    parser.add_argument('--tv_loss', type=float, default=0, help='tv loss coefficient')
    # some actions:
    parser.add_argument('--uniform_init', action='store_true', help='whether to use uniform initialization of model')
    parser.add_argument('--trained_model', action='store_true', help='whether to use uniform initialization of model')
    # params for modifying gradients:
    parser.add_argument('--stage', type=str, default='defend', choices=['pure', 'defend'], help='whether to modify the gradients')
    parser.add_argument('--modify', type=str, default='noise', choices=['noise', 'pruning'], help='use which method to modify the gradients; "zero" means set grad as zeros; "random" means set grad as random noise')
    # params for noise generate:
    parser.add_argument('--mean', type=str, default='0', choices=['0', 'batch'], help='the mean of sampling noise')
    parser.add_argument('--std', type=str, default='1e-3', help='std of sampling noise; can be a scalar (e.g., 1)')
    parser.add_argument('--var', type=str, default='1e-3', help='variance of sampling noise; can be a scalar (e.g., 1)')
    # params for noise control:
    parser.add_argument('--energy', type=float, default=1, help='noise energy control or the scaling factor when modify==scale; should be deactive if args.project is True; also the noise multiplier of DP')
    args = parser.parse_args()

    '''env settings:'''
    set_random_seed(args.seed)
    defs = inversefed.training_strategy('conservative')
    setup = inversefed.utils.system_startup(gpu=args.gpu)
    if args.inv_iterations % 10 == 0:
        args.inv_iterations += 1
    args.std = np.sqrt(float(args.var))

    '''set dataset:'''
    # number of classes, img size, data shape:
    args.num_classes = 10
    if args.dataset == 'cifar100':
        args.num_classes = 100
    if args.dataset in ['cifar10', 'cifar100']:
        resize = (32, 32)
        data_shape = (3, 32, 32)
    elif args.dataset == 'mnist':
        resize = (28, 28)
        data_shape = (1, 28, 28)
    elif args.dataset == 'imagenet':
        args.num_classes = 1000
        resize = (224, 224)
        data_shape = (3, 224, 224)
    # num of img channels:
    if args.dataset in ['mnist', 'fmnist', 'mmnist']: args.num_channels = 1
    else: args.num_channels = 3
    # data loaders
    args.train_bsz = 1024
    args.train_lr = 0.1
    args.normalize = True
    if args.model == 'ResNet18': args.normalize = True
    if args.inv_goal == 'sim': args.normalize = True
    if args.dataset != 'imagenet':
        train_set, test_set, trainloader, testloader, dm, ds = baseline_utils.load_datasets(args, args.dataset, val=False, resize=resize, normalize=args.normalize)
    else:
        loss_fn, trainset, validset, testset, trainloader, validloader =  inversefed.construct_dataloaders('ImageNet', defs, 
                                                                            data_path=os.path.join('/localscratch2/hbzhang/data', "ILSVRC2012"))
        test_set = validloader.dataset
    dm = torch.as_tensor(dm, **setup)[:, None, None]
    ds = torch.as_tensor(ds, **setup)[:, None, None]

    '''set modules:'''
    if args.dataset == 'imagenet':
        if args.model == 'ResNet18': model = torchvision.models.resnet18(pretrained=False)
        elif args.model == 'ResNet50': model = torchvision.models.resnet50(pretrained=False)
        elif args.model == 'ResNet101': model = torchvision.models.resnet101(pretrained=True)
        elif args.model == 'ResNet152': model = torchvision.models.resnet152(pretrained=True)
    else:
        model = baseline_utils.load_model('vit', num_classes=args.num_classes, data_shape=data_shape, num_channels=args.num_channels)
    model.to(**setup)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    wandb.init(project='I2F', name='bound-I2F', config=vars(args), mode='online')
    result_dir = os.path.join(data_root, 'result_bound_I2F')

    # lpips:
    alexnet = lpips.LPIPS(net='alex').to(setup['device'])
    vgg = lpips.LPIPS(net='vgg').to(setup['device'])
    squeeze = lpips.LPIPS(net='squeeze').to(setup['device'])

    print(args)
    main()


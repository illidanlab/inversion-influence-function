"""Mechanisms for image reconstruction from parameter gradients."""
import numpy as np
import torch
import torchvision
from collections import defaultdict, OrderedDict
from inversefed.nn import MetaMonkey
from inversefed.metrics import total_variation as TV
from inversefed.metrics import InceptionScore
from inversefed.medianfilt import MedianPool2d
from inversefed.metrics import psnr
from metrics import ssim_batch
from copy import deepcopy

from matplotlib import pyplot as plt

import time, os

DEFAULT_CONFIG = dict(signed=False,
                    boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=3000,
                      total_variation=1e-1,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss',
                      results_dir='',
                      log_dir='',
                      images_dir='images_dir',
                      figure_dir='figure_dir',
                      random_idx=-1,
                      state='',)

def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, ground_truth, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1, init_img=None):
        """
            Initialize with algorithm setup.
            Ground-truth here is the masked ground-truth gradients.
            The same mask should be used in the estimated gradients.
        """
        self.config = _validate_config(config)
        self.model = model
        self.ground_truth = ground_truth
        self.mask = None
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.dm, self.ds = mean_std
        self.num_images = num_images
        self.init_img = init_img

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.iDLG = True
        self.scores = {
            'RecLoss': [[[] for j in range(num_images+1)] for _ in range(self.config['restarts'])],
            'MSE': [[[] for j in range(num_images+1)] for _ in range(self.config['restarts'])],
            'SSIM': [[[] for j in range(num_images+1)] for _ in range(self.config['restarts'])],
            'FMSE': [[[] for j in range(num_images+1)] for _ in range(self.config['restarts'])],
            'PSNR': [[[] for j in range(num_images+1)] for _ in range(self.config['restarts'])],
            'TVLoss': [[[] for j in range(num_images+1)] for _ in range(self.config['restarts'])],
        }
        self.iterations = [[] for _ in range(self.config['restarts'])]

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None, mask=None):
        """Reconstruct image from gradient."""
        self.mask = mask
        start_time = time.time()
        if eval:
            self.model.eval()
        stats = defaultdict(list)
        # initialize the recovery image:
        x = self._init_images(img_shape)
        save_init = torch.clamp(x.squeeze(0) * self.ds + self.dm, 0, 1)
        torchvision.utils.save_image(save_init, os.path.join(self.config['images_dir'], 'img%d_init.png'%(self.config['random_idx'])))
        scores = torch.zeros(self.config['restarts'])

        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
                self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False
            self.labels = labels

        try:
            for trial in range(self.config['restarts']):
                x_trial, labels = self._run_trial(x[trial], input_data, labels, trial, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        else:
            print('Choosing optimal result ...')
            # accept nan values, which means the attack totally failed
            # scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        for name in self.scores.keys():
            self._plot_metric(name, optimal_index)
        return x_optimal.detach(), stats

    def _init_images(self, img_shape):
        if self.init_img is not None:
            return self.init_img
        # self.ground_truth is the image
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'original':
            init_img = self.ground_truth.unsqueeze(0)
            return init_img
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, trial, dryrun=False):
        log_file = os.path.join(self.config['figure_dir'], 'output_trial%d.log' % trial)
        log = open(log_file, 'w+')
        log.write('Iter | Rec Loss | Grad Norm | MSE | SSIM | TV Loss\n')
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'lbfgs':
                optimizer = torch.optim.LBFGS([x_trial, labels])
            else:
                raise ValueError()
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'lbfgs':
                optimizer = torch.optim.LBFGS([x_trial])
            else:
                raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            flag = False
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels)
                # rec_loss, grad_norm, grad_trial, rec_loss_grad = optimizer.step(closure)
                rec_loss = optimizer.step(closure)

                rec_loss_grad = x_trial.grad.detach().data
                grad_norm = torch.norm(x_trial.grad.data.detach())
                grad_trial = torch.autograd.grad(self.loss_fn(self.model(x_trial), labels), self.model.parameters())

                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config['boxed']:
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)
                    if (iteration + 1 == max_iterations) or (iteration + 1) % 5 == 0 or (iteration == 0):
                        if ((iteration + 1) % 500 == 0) or (iteration == 0):
                            print(f'It: {(iteration + 1)}. Rec. loss: {rec_loss.item():2.4f}. Grad norm: {grad_norm.item()}')
                        batch_MSE = (x_trial - self.ground_truth).pow(2).mean().item()
                        batch_TVLoss = TV(x_trial).detach().item()
                        mean_ssim, batch_ssims = ssim_batch(x_trial, self.ground_truth)
                        MSEs = []
                        TVLosses = []

                        self.scores['RecLoss'][trial][-1].append(rec_loss.item())
                        self.scores['FMSE'][trial][-1].append((self.model(x_trial) - self.model(self.ground_truth)).pow(2).mean().item())
                        self.scores['MSE'][trial][-1].append(batch_MSE)
                        self.scores['SSIM'][trial][-1].append(mean_ssim)
                        self.scores['PSNR'][trial][-1].append(psnr(x_trial, self.ground_truth, factor=1 / ds))
                        self.scores['TVLoss'][trial][-1].append(batch_TVLoss)
                        self.iterations[trial].append(iteration)
                        for i in range(self.num_images):
                            img_MSE = (x_trial[i].unsqueeze(0) - self.ground_truth[i].unsqueeze(0)).pow(2).mean().item()
                            img_TVLoss = TV(x_trial[i].unsqueeze(0)).detach().item()
                            MSEs.append(img_MSE)
                            TVLosses.append(img_TVLoss)
                            self.scores['RecLoss'][trial][i].append(rec_loss.item())
                            self.scores['FMSE'][trial][i].append((self.model(x_trial[i].unsqueeze(0)) - self.model(self.ground_truth[i].unsqueeze(0))).pow(2).mean().item())
                            self.scores['MSE'][trial][i].append(img_MSE)
                            self.scores['SSIM'][trial][i].append(batch_ssims[i].item())
                            self.scores['PSNR'][trial][i].append(psnr(x_trial[i].unsqueeze(0), self.ground_truth[i].unsqueeze(0), factor=1 / ds))
                            self.scores['TVLoss'][trial][i].append(img_TVLoss)
                        MSEs.append(batch_MSE)
                        TVLosses.append(batch_TVLoss)
                        log.write('{} | {} | {} | {} | {} | {}\n'.format(iteration, rec_loss.item(), grad_norm.item(), MSEs, batch_ssims, TVLosses))
                        log.flush()

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()
                    
                    if (iteration) <= 50000:
                        if (iteration) % 100 == 0:
                            self._save_img(x_trial, iteration)
                            former_grad = torch.clamp(rec_loss_grad, 0, 1)
                            flag = True
                        if flag:
                            flag = False
                    elif (iteration + 1) % 10000 == 0:
                        former_grad = torch.clamp(rec_loss_grad, 0, 1)
                        flag = True
                        self._save_img(x_trial, iteration)
                    if flag:
                        flag = False
                        
                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            log.close()
            pass
        log.close()
        return x_trial.detach(), labels

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            out = self.model(x_trial)
            if len(out) != x_trial.shape[0]:
                out = out[0]
            # loss = self.loss_fn(self.model(x_trial), label)
            loss = self.loss_fn(out, label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.mask if self.mask is not None else self.config['weights'], )

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            rec_loss.backward()
            # show the gradients of reconstruction loss w.r.t. the x_trial
            # grad_norm = torch.norm(x_trial.grad).data.detach()
            # print('grad norm: ', grad_norm)
            if self.config['signed']:
                x_trial.grad.sign_()
            # return rec_loss, grad_norm, gradient, x_trial.grad.data.detach()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights= self.mask if self.mask is not None else self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(x_optimal).softmax(dim=1)
        loss = self.loss_fn(self.model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights= self.mask if self.mask is not None else self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats
    
    def _plot_metric(self, name, optimal_index):
        # plot metrics of all images in one figure:
        legends = []
        for i in range(self.num_images):
            if name == 'RecLoss' or 'MSE':
                plt.plot(self.iterations[optimal_index], np.log10(self.scores[name][optimal_index][i]))
            else:
                plt.plot(self.iterations[optimal_index], self.scores[name][optimal_index][i])
            legends.append('class%d' % self.labels[i].cpu().detach().numpy())
        if name == 'RecLoss' or 'MSE':
            plt.plot(self.iterations[optimal_index], np.log10(self.scores[name][optimal_index][-1]))
        else:
            plt.plot(self.iterations[optimal_index], self.scores[name][optimal_index][-1])
        legends.append('batch' % self.labels[i].cpu().detach().numpy())
        plt.ylabel(name)
        plt.xlabel('Iterations')
        plt.legend(legends)
        plt.savefig(os.path.join(self.config['figure_dir'], '%s.png'%name))
        # plt.show()
        plt.close()

        # plot all metrics in multi-figures:
        # if (name == 'MSE') or (name == 'PSNR'):
        if name in ['MSE', 'PSNR', 'SSIM']:
            for i in range(self.num_images):
                plt.plot(self.iterations[optimal_index], self.scores[name][optimal_index][i])
                plt.ylabel(name)
                plt.xlabel('Iterations')
                plt.savefig(os.path.join(self.config['figure_dir'], '%s_img%d.png'% (name, i)))
                # plt.show()
                plt.close()
            plt.plot(self.iterations[optimal_index], self.scores[name][optimal_index][-1])
            plt.ylabel(name)
            plt.xlabel('Iterations')
            plt.savefig(os.path.join(self.config['figure_dir'], '%s_batch.png'% (name)))
            # plt.show()
            plt.close()

    def _save_img(self, img, iteration):
        img = img.detach()
        if self.config['cost_fn'] == 'sim':
            img = torch.clamp(img * self.ds + self.dm, 0, 1)
        else:
            img = torch.clamp(img, 0, 1)
        torchvision.utils.save_image(img, os.path.join(self.config['images_dir'], '%s_%d.png'%(self.config['state'], iteration)))
        # save diff:
        diff = img - self.ground_truth
        if self.config['cost_fn'] == 'sim':
            diff = torch.clamp(diff * self.ds + self.dm, 0, 1)
        else:
            diff = torch.clamp(diff, 0, 1)
        torchvision.utils.save_image(diff, os.path.join(self.config['images_dir'], '%s_%d.png'%('diff', iteration)))

class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config=DEFAULT_CONFIG, num_images=1, use_updates=True, batch_size=0):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size

    def _gradient_closure(self, optimizer, x_trial, input_parameters, labels):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr,
                                    use_updates=self.use_updates,
                                    batch_size=self.batch_size)
            rec_loss = reconstruction_costs([parameters], input_parameters,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights= self.mask if self.mask is not None else self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_parameters, labels):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            return reconstruction_costs([parameters], input_parameters,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights= self.mask if self.mask is not None else self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)


def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True, batch_size=0):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            idx = i % (inputs.shape[0] // batch_size)
            outputs = patched_model(inputs[idx * batch_size:(idx + 1) * batch_size], patched_model.parameters)
            labels_ = labels[idx * batch_size:(idx + 1) * batch_size]
        loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    elif weights == 'equal':
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / (pnorm[0].sqrt()) / (pnorm[1].sqrt())

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)
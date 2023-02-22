import math
import sys
import os
import time
from tqdm import tqdm
import logging

import warnings
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils.deit_util as utils
from utils import AverageMeter, to_device

from torchmeta.utils.gradient_based import gradient_update_parameters
from collections import OrderedDict


def train_one_epoch(data_loader: Iterable,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    device: torch.device,
                    loss_scaler = None,
                    fp16: bool = False,
                    max_norm: float = 0, # clip_grad
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    writer: Optional[SummaryWriter] = None,
                    set_training_mode=True,
                    maml = None,
                    ):

    global_step = epoch * len(data_loader)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    model.train(set_training_mode)

    for batch in metric_logger.log_every(data_loader, print_freq, header): # data_loader.next() -> sptTensor, sptLabel, x, y
        batch = to_device(batch, device)
        SupportTensor, SupportLabel, x, y = batch
#         import pdb
#         pdb.set_trace()
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        # forward
            # maml은 다르게
        if maml:
            with torch.cuda.amp.autocast(fp16):
                loss = 0
                for task_idx, (train_input, train_target, test_input, test_target) in enumerate(zip(SupportTensor, SupportLabel, x, y)):
                    train_logit = model(train_input)
                    inner_loss = F.cross_entropy(train_logit, train_target)
                    inner_loss.requires_grad = True

                    model.zero_grad()
                    params = gradient_update_parameters(model,
                                                        inner_loss,
                                                        step_size=maml['step_size'],
                                                        first_order=maml['first_order']
                                                        )

                    test_logit = model(test_input, params=params) # output
                    loss += F.cross_entropy(test_logit, test_target)

#                     with torch.no_grad():
#                         accuracy += get_accuracy(test_logit, test_target)

                loss.div_(maml['batch_size'])
#                 accuracy.div_(args.batch_size)
                loss_value = loss.item() # TODO:May cause error
                loss.requires_grad = True
                loss.backward()
                optimizer.step()
        else:
            with torch.cuda.amp.autocast(fp16):
                output = model(SupportTensor, SupportLabel, x)

            output = output.view(x.shape[0] * x.shape[1], -1)
            y = y.view(-1)
            loss = criterion(output, y)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()

            if fp16:
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order)
            else:
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
        metric_logger.update(n_ways=SupportLabel.max()+1)
        metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])

        # tensorboard
        if utils.is_main_process() and global_step % print_freq == 0:
            writer.add_scalar("train/loss", scalar_value=loss_value, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=lr, global_step=global_step)

        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(data_loaders, model, criterion, device, seed=None, ep=None, maml=None):
    if isinstance(data_loaders, dict):
        test_stats_lst = {}
        test_stats_glb = {}

        for j, (source, data_loader) in enumerate(data_loaders.items()):
            print(f'* Evaluating {source}:')
            seed_j = seed + j if seed else None
            test_stats = _evaluate(data_loader, model, criterion, device, seed_j, maml=maml)
            test_stats_lst[source] = test_stats
            test_stats_glb[source] = test_stats['acc1']

        # apart from individual's acc1, accumulate metrics over all domains to compute mean
        for k in test_stats_lst[source].keys():
            test_stats_glb[k] = torch.tensor([test_stats[k] for test_stats in test_stats_lst.values()]).mean().item()

        return test_stats_glb
    elif isinstance(data_loaders, torch.utils.data.DataLoader): # when args.eval = True
        return _evaluate(data_loaders, model, criterion, device, seed, ep, maml=maml)
    else:
        warnings.warn(f'The structure of {data_loaders} is not recognizable.')
        return _evaluate(data_loaders, model, criterion, device, seed, maml=maml)


@torch.no_grad()
def _evaluate(data_loader, model, criterion, device, seed=None, ep=None, maml=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    metric_logger.add_meter('acc5', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if seed is not None:
        data_loader.generator.manual_seed(seed)

    for ii, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if ep is not None:
            if ii > ep:
                break

        batch = to_device(batch, device)
        SupportTensor, SupportLabel, x, y = batch
        # compute output
        if maml:
            with torch.cuda.amp.autocast():
                # Finetune
                SupportTensor, SupportLabel, x, y = SupportTensor.squeeze(), SupportLabel.squeeze(), x.squeeze(), y.squeeze()
                train_logit = model(SupportTensor)
                inner_loss = F.cross_entropy(train_logit, SupportLabel)

                model.zero_grad()
                inner_loss.requires_grad = True
                params = OrderedDict(model.meta_named_parameters())
#                 import pdb
#                 pdb.set_trace()
#                 grads = torch.autograd.grad(inner_loss,
#                                             params.values(),
#                                             create_graph=False,
#                                             )
#                 updated_params = OrderedDict()
#                 for (name, param), grad in zip(params.items(), grads):
#                     updated_params[name] = param - maml['step_size'] * grad
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=maml['step_size'],
                                                    first_order=maml['first_order'],
                                                    )
                # Get output
                output = model(x, params=params)
                loss = F.cross_entropy(output, y)

#                 with torch.no_grad():
#                     accuracy += get_accuracy(test_logit, test_target)
        else:
            with torch.cuda.amp.autocast():
                output = model(SupportTensor, SupportLabel, x)

                output = output.view(x.shape[0] * x.shape[1], -1)
                y = y.view(-1)
                loss = criterion(output, y)
        acc1, acc5 = accuracy(output, y, topk=(1, 5))

        batch_size = x.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.update(n_ways=SupportLabel.max()+1) # TODO:
        metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    ret_dict['acc_std'] = metric_logger.meters['acc1'].std

    return ret_dict

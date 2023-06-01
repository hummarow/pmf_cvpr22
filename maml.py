import sys
import datetime
import random
import numpy as np
import time
import tqdm
import torch
import torch.backends.cudnn as cudnn
import json
import gc
import umap
import mem
import matplotlib.pyplot as plt

from datetime import timedelta
from collections import defaultdict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from engine import train_one_epoch
import utils.deit_util as utils
from utils import AverageMeter, to_device
from datasets import get_loaders, get_sets
from utils.args import get_args_parser
from models import get_model
from models.meta import Meta


def main(args):
#     utils.init_distributed_mode(args)
    args.distributed = True

    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    if utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "log.txt").open("a") as f:
            f.write(" ".join(sys.argv) + "\n")


    plot_dir = Path(args.plot_dir)
    if utils.is_main_process():
        plot_dir.mkdir(parents=True, exist_ok=True)

    ##############################################
    # Data loaders
    args.choose_train = False
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    data_loader_train, data_loader_val = get_loaders(args, num_tasks, global_rank)
    ##############################################
    # Mixup regularization (by default OFF)
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nClsEpisode)

    ##############################################
    # Model Config
    conv_kernel = 3
    conv_stride = 1
    conv_pad = 0
    pool_kernel = 2
    pool_stride = 2
    pool_pad = 0

    model_config_no_classifier = [
        # [ch_out, ch_in, kernel, kernel, stride, pad]
        ("conv2d", [32, 3, conv_kernel, conv_kernel, conv_stride, conv_pad]),
        ("relu", [True]),  # [inplace]
        ("bn", [32]),  # [ch_out]
        # [kernel, stride, padding]
        ("max_pool2d", [pool_kernel, pool_stride, pool_pad]),
    ]
    model_config_no_classifier.extend([
        ("conv2d", [32, 32, conv_kernel, conv_kernel, conv_stride, conv_pad]),
        ("relu", [True]),
        ("bn", [32]),
        ("max_pool2d", [pool_kernel, pool_stride, pool_pad]),
    ]*3) # Due to different input channel of conv2d layer
    model_config_no_classifier.extend([
        ("flatten", []),
    ])
    ##############################################
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))

    # Calculate linear layer shape
    if args.eval:
        _, _, dset = get_sets(args)
    else:
        _, dset, _ = get_sets(args)
    linear_shape = dset[list(dset.keys())[0]][0][0].shape[2]
    for _ in range(4):
        linear_shape -= 2
        linear_shape //= 2
    linear_shape **= 2
    linear_config = [
        ("linear", [args.n_way, 32 * linear_shape]),
    ]
# 
#     model = Meta(args, model_config_no_classifier)
#     # Append linear layer to the model
#     model.net.append(linear_config)
#     model.to(device)
# 
#     del model
    params_all = []
    num_all_parameters = 0
#     max_accuracy = acc
    for i, (train_source, train_loader) in enumerate(data_loader_train.items()):
#         model_config = [
#             ("conv2d", [32, 3, 3, 3, 1, 0]),  # [ch_out, ch_in, kernel, kernel, stride, pad]
#             ("relu", [True]),  # [inplace]
#             ("bn", [32]),  # [ch_out]
#             ("max_pool2d", [2, 2, 0]),  # [kernel, stride, padding]
#             ("conv2d", [32, 32, 3, 3, 1, 0]),
#             ("relu", [True]),
#             ("bn", [32]),
#             ("max_pool2d", [2, 2, 0]),
#             ("conv2d", [32, 32, 3, 3, 1, 0]),
#             ("relu", [True]),
#             ("bn", [32]),
#             ("max_pool2d", [2, 2, 0]),
#             ("conv2d", [32, 32, 3, 3, 1, 0]),
#             # ("conv2d", [32, 32, 3, 3, 1, 1]),
#             ("relu", [True]),
#             ("bn", [32]),
#             ("max_pool2d", [2, 1, 0]),
#             # ("max_pool2d", [2, 2, 0]),
#             ("flatten", []),
#             # ("linear", [args.n_way, 32 * 5 * 5]),
#             ("linear", [args.n_way, 3872]),
#         ]
#         model = Meta(args, model_config)
        model = Meta(args, model_config_no_classifier)
        # Append linear layer to the model
        model.net.append(linear_config)
        model.to(device)
        for epoch in range(args.start_epoch, args.epochs):
            header = 'Epoch: [{}]'.format(epoch)
            print(header)
            start_time = time.monotonic()
            print_freq = 10
            # for batch in metric_logger.log_every(data_loader_train, print_freq, header):
            acc = 0
            for episode, batch in enumerate(train_loader):
#                 torch.cuda.empty_cache()
#                 gc.collect()
#                 mem_usage = mem.get_memory_usage()
#                 max_pair = max(mem_usage.items(), key=lambda x: x[1])
#                 var_name = mem.get_variable_name(max_pair[0])
#                 print(var_name)
#                 batch = to_device(batch, device)
#                 spt_xs, spt_ys, qry_xs, qry_ys = batch
#                 acc_per_episode = model(spt_xs, spt_ys, qry_xs, qry_ys)

#                 print("Before")
#                 print(torch.cuda.memory_reserved())
                acc_per_episode = train(batch, model, device)
#                 print("After")
#                 print(torch.cuda.memory_reserved())
                acc += acc_per_episode
                if episode % 20 == 0:
                    print("Episode: [{}]".format(epoch * len(train_loader) + episode))
                    print("Train Acc: {:.2f}%".format(acc_per_episode*100))
                if episode % 100 == 0:
                    # Evaluate
                    val_acc = 0
                    params = []
                    tsne_ranges = {}
                    param_per_epoch = defaultdict(list)
                    for j, (source, val_loader) in enumerate(data_loader_val.items()):
                        val_acc_per_loader = 0
                        start_index = len(params)
                        for batch in val_loader:
                            batch = to_device(batch, device)
                            spt_xs, spt_ys, qry_xs, qry_ys = batch
                            for task_idx, (spt_x, spt_y, qry_x, qry_y) in enumerate(zip(spt_xs, spt_ys, qry_xs, qry_ys)):
                                val_acc_per_loader += model.finetunning(spt_x, spt_y, qry_x, qry_y)
                            # Get finetuned parameters per source
                            params.extend(model.finetuned_parameter_list)
                            param_per_epoch[source].extend(model.finetuned_parameter_list)
                            tsne_ranges[source] = slice(start_index, len(params))
                        val_acc_per_loader /= (len(val_loader) * len(spt_xs))
                        print('{} acc: {:.2f}%'.format(source, val_acc_per_loader*100))
                        val_acc += val_acc_per_loader
                    val_acc /= len(data_loader_val)
                    print('Average val acc: {}%'.format(val_acc*100))
                    # Save finetuned parameters 
                    if num_all_parameters == 0:
                        num_all_parameters = len(params)+1
                    print('Plotting parameters T-SNE')
                    # Color list
                    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'blueviolet', 'magenta', 'peru', 'lime']
                    # Add meta-parameter to the end
                    params.append(np.concatenate([torch.flatten(p.detach().cpu()).numpy() for p in model.net.parameters()]).flatten())
                    param_per_epoch['Meta-Parameter'].append(np.concatenate([torch.flatten(p.detach().cpu()).numpy() for p in model.net.parameters()]).flatten())
                    params_all.append(param_per_epoch)

                    embedding = umap.UMAP(random_state=1004).fit_transform(np.array(params))
                    for i, source in enumerate(tsne_ranges.keys()):
                        xs = embedding[tsne_ranges[source], 0]
                        ys = embedding[tsne_ranges[source], 1]
                        plt.scatter(xs, ys, c=colors[i], label=source)
                    plt.scatter(embedding[-1, 0], embedding[-1, 1], c='k', label='Meta-Parameter')
                    plt.legend()
                    plt.show()
                    plt.savefig('{}/{}_umap.png'.format(args.plot_dir, epoch*len(train_loader) + episode), dpi=400)
                    plt.clf()
                    embedding = umap.UMAP(random_state=1005).fit_transform(np.array(params))
                    for i, source in enumerate(tsne_ranges.keys()):
                        xs = embedding[tsne_ranges[source], 0]
                        ys = embedding[tsne_ranges[source], 1]
                        plt.scatter(xs, ys, c=colors[i], label=source)
                    plt.scatter(embedding[-1, 0], embedding[-1, 1], c='k', label='Meta-Parameter')
                    plt.legend()
                    plt.show()
                    plt.savefig('{}/{}_umap_1.png'.format(args.plot_dir, epoch*len(train_loader) + episode), dpi=400)
                    plt.clf()

                    tsne = TSNE(n_components=2, random_state=1004)
                    embedding = tsne.fit_transform(np.array(params))
                    for i, source in enumerate(tsne_ranges.keys()):
                        xs = embedding[tsne_ranges[source], 0]
                        ys = embedding[tsne_ranges[source], 1]
                        plt.scatter(xs, ys, c=colors[i], label=source)
                    plt.scatter(embedding[-1, 0], embedding[-1, 1], c='k', label='Meta-Parameter')
                    plt.legend()
                    plt.show()
                    plt.savefig('{}/{}.png'.format(args.plot_dir, epoch*len(train_loader) + episode), dpi=400)
                    plt.clf()
                    del tsne
                    # Test 3d
                    tsne = TSNE(n_components=3, random_state=1004)
                    embedding = tsne.fit_transform(np.array(params))
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    for i, source in enumerate(tsne_ranges.keys()):
                        xs = embedding[tsne_ranges[source], 0]
                        ys = embedding[tsne_ranges[source], 1]
                        zs = embedding[tsne_ranges[source], 2]
                        ax.scatter(xs, ys, zs, c=colors[i], label=source)
                    ax.scatter(embedding[-1, 0], embedding[-1, 1], c='k', label='Meta-Parameter')
                    plt.legend()
                    plt.show()
                    plt.savefig('{}/{}_3d.png'.format(args.plot_dir, epoch*len(train_loader) + episode), dpi=400)
                    plt.clf()
                    del tsne
                    embedding = umap.UMAP(n_components=3, random_state=1004).fit_transform(np.array(params))
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    for i, source in enumerate(tsne_ranges.keys()):
                        xs = embedding[tsne_ranges[source], 0]
                        ys = embedding[tsne_ranges[source], 1]
                        zs = embedding[tsne_ranges[source], 2]
                        ax.scatter(xs, ys, zs, c=colors[i], label=source)
                    ax.scatter(embedding[-1, 0], embedding[-1, 1], c='k', label='Meta-Parameter')
                    plt.legend()
                    plt.show()
                    plt.savefig('{}/{}_3d_umap.png'.format(args.plot_dir, epoch*len(train_loader) + episode), dpi=400)
                    plt.clf()

            end_time = time.monotonic()
            acc /= (len(train_loader) * len(spt_xs))
            print("Acc: {:.2f}%,\tElapsed time: {}".format(acc*100, timedelta(end_time-start_time)))
        del model
    # apply t-SNE on the concatenated numpy array
    # params = np.concatenate(params)
#     tsne = TSNE(n_components=2, perplexity=5.0)
#     params = np.array(params)
#     embedding = tsne.fit_transform(params)

def train(batch, model, device):
    batch = to_device(batch, device)
    spt_xs, spt_ys, qry_xs, qry_ys = batch
    acc_per_episode = model(spt_xs, spt_ys, qry_xs, qry_ys)
    return acc_per_episode

def evaluate(data_loaders, model, criterion, device, seed=None, ep=None, maml=None):
    print('Test Accuracy')
    acc = 0
    for j, (source, loader) in enumerate(data_loaders.items()):
        acc_per_loader = 0
        for batch in loader:
            batch = to_device(batch, device)
            spt_xs, spt_ys, qry_xs, qry_ys = batch
            for task_idx, (spt_x, spt_y, qry_x, qry_y) in enumerate(zip(spt_xs, spt_ys, qry_xs, qry_ys)):
                acc_per_loader += model.finetunning(spt_x, spt_y, qry_x, qry_y)
        acc_per_loader /= (len(loader) * len(spt_xs))
        print(' * {} acc: {:.2f}%'.format(source, acc_per_loader*100))
        acc += acc_per_loader
    acc /= len(data_loader_val)
    print('Average test acc: {:.2f}%'.format(acc*100))
    return acc

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)

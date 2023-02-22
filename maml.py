import sys
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from engine import train_one_epoch, evaluate
import utils.deit_util as utils
from utils import AverageMeter, to_device
from datasets import get_loaders, get_sets
from utils.args import get_args_parser
from models import get_model
from models.meta import Meta


def main(args):
#     utils.init_distributed_mode(args)
    args.distributed = False

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

    ##############################################
    # Data loaders
    args.choose_train = True
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
        ("conv2d", [32, 3, conv_kernel, conv_kernel, conv_stride, conv_pad]),  # [ch_out, ch_in, kernel, kernel, stride, pad]
        ("relu", [True]),  # [inplace]
        ("bn", [32]),  # [ch_out]
        ("max_pool2d", [pool_kernel, pool_stride, pool_pad]),  # [kernel, stride, padding]
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
#     model = Meta(args, model_config)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
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
    model = Meta(args, model_config_no_classifier)
    # Append linear layer to the model
    model.net.append(linear_config)
    model.to(device)

    for i, (source, data_loader) in enumerate(data_loader_val.items()):
        pass
        # Get input shape to set linear layer
#         linear_shape = dset[source][0][0].shape[2]
#         for _ in range(4):
#             linear_shape -= 2
#             linear_shape //= 2
#         linear_shape **= 2
# 
#         linear_config = [
#             ("linear", [args.n_way, 32 * linear_shape]),
#         ]
#         # Append linear layer to the model
#         model.net.append(linear_config)
#         model.to(device)

        # for ii, batch in enumerate(metric_logger.log_every(data_loader, 10)):
#         for ii, batch in enumerate(data_loader):
#             batch = to_device(batch, device)
#             spt_xs, spt_ys, qry_xs, qry_ys = batch
#             for task_idx, (spt_x, spt_y, qry_x, qry_y) in enumerate(zip(spt_xs, spt_ys, qry_xs, qry_ys)):
#                 acc = model.finetunning(spt_x, spt_y, qry_x, qry_y)
#         model.net.pop()
    if args.eval:
        return

    
    params = []
    start_time = time.time()
#     max_accuracy = acc
    for i, (train_source, train_loader) in enumerate(data_loader_train.items()):
        model = Meta(args, model_config_no_classifier)
        # Append linear layer to the model
        model.net.append(linear_config)
        model.to(device)
        for epoch in range(args.start_epoch, args.epochs):
            header = 'Epoch: [{}]'.format(epoch)
            print_freq = 10
            # for batch in metric_logger.log_every(data_loader_train, print_freq, header):
            for batch in train_loader:
                batch = to_device(batch, device)
                spt_xs, spt_ys, qry_xs, qry_ys = batch
                acc = model(spt_xs, spt_ys, qry_xs, qry_ys)
            
            for j, (source, val_loader) in enumerate(data_loader_val.items()):
                for batch in val_loader:
                    batch = to_device(batch, device)
                    spt_xs, spt_ys, qry_xs, qry_ys = batch
                    for task_idx, (spt_x, spt_y, qry_x, qry_y) in enumerate(zip(spt_xs, spt_ys, qry_xs, qry_ys)):
                        acc = model.finetunning(spt_x, spt_y, qry_x, qry_y)
        

        for param in model.net.parameters().cpu():
            params.append(param.detach().numpy())

        del model
    # apply t-SNE on the concatenated numpy array
    # params = np.concatenate(params)
#     tsne = TSNE(n_components=2, perplexity=5.0)
#     params = np.array(params)
#     embedding = tsne.fit_transform(params)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)

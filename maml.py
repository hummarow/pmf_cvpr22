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


class Trainer(object):
    def __init__(self, args):
        args.distributed = True
        self.device = torch.device(args.device)
        self.seed = args.seed + utils.get_rank()
        args.seed = self.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        cudnn.benchmark = True

        self.plot_dir = Path(args.plot_dir)

        if utils.is_main_process():
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Data loaders
        self.num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        (self.data_loader_train, self.outer_support_set_train), (
            self.data_loader_val,
            self.outer_support_set_val,
        ) = get_loaders(args, self.num_tasks, global_rank)

        # Mixup regularization (by default OFF)
        self.mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.nClsEpisode,
            )

        # Model Config
        _conv_kernel = 3
        _conv_stride = 1
        _conv_pad = 0
        _pool_kernel = 2
        _pool_stride = 2
        _pool_pad = 0

        self.model_config_no_classifier = [
            # [ch_out, ch_in, kernel, kernel, stride, pad]
            ("conv2d", [32, 3, _conv_kernel, _conv_kernel, _conv_stride, _conv_pad]),
            ("relu", [True]),  # [inplace]
            ("bn", [32]),  # [ch_out]
            # [kernel, stride, padding]
            ("max_pool2d", [_pool_kernel, _pool_stride, _pool_pad]),
        ]
        self.model_config_no_classifier.extend(
            [
                ("conv2d", [32, 32, _conv_kernel, _conv_kernel, _conv_stride, _conv_pad]),
                ("relu", [True]),
                ("bn", [32]),
                ("max_pool2d", [_pool_kernel, _pool_stride, _pool_pad]),
            ]
            * 3
        )  # Due to different input channel of conv2d layer
        self.model_config_no_classifier.extend(
            [
                ("flatten", []),
            ]
        )

        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        self.metric_logger.add_meter("n_ways", utils.SmoothedValue(window_size=1, fmt="{value:d}"))
        self.metric_logger.add_meter("n_imgs", utils.SmoothedValue(window_size=1, fmt="{value:d}"))

        # Calculate linear layer shape
        if args.eval:
            _, _, _dset, _, _, _ = get_sets(args)
        else:
            _, _dset, _, _, _, _ = get_sets(args)
        _linear_shape = _dset[list(_dset.keys())[0]][0][0].shape[2]
        for _ in range(4):
            _linear_shape -= 2
            _linear_shape //= 2
        _linear_shape **= 2

        self.linear_config = [
            ("linear", [args.n_way, 32 * _linear_shape]),
        ]

        self.args = args
        print(self.args)

    def _train_one_batch(self, batch, model, device, phi=None):
        batch = to_device(batch, device)
        spt_xs, spt_ys, qry_xs, qry_ys = batch
        acc_per_episode = model(spt_xs, spt_ys, qry_xs, qry_ys, phi=phi)
        return acc_per_episode

    def train(self):
        for i, (train_source, train_loader) in enumerate(self.data_loader_train.items()):
            model = Meta(args, self.model_config_no_classifier)
            # Append linear layer to the model
            model.net.append(self.linear_config)
            model.to(self.device)
            for epoch in range(args.start_epoch, args.epochs):
                header = "Epoch: [{}]".format(epoch)
                print(header)
                start_time = time.monotonic()
                print_freq = 10
                # for batch in metric_logger.log_every(self.data_loader_train, print_freq, header):
                acc = 0
                for episode, batch in enumerate(train_loader):
                    acc_per_episode = self._train_one_batch(batch, model, self.device)
                    acc += acc_per_episode

                    if episode % 20 == 0:
                        print("Episode: [{}]".format(epoch * len(train_loader) + episode))
                        print("Train Acc: {:.2f}%".format(acc_per_episode * 100))
                    if episode % 100 == 0:
                        # Evaluate
                        val_acc = 0
                        params = []
                        tsne_ranges = {}
                        param_per_epoch = defaultdict(list)
                        for j, (source, val_loader) in enumerate(self.data_loader_val.items()):
                            val_acc_per_loader = 0
                            start_index = len(params)
                            for batch in val_loader:
                                batch = to_device(batch, self.device)
                                spt_xs, spt_ys, qry_xs, qry_ys = batch
                                for task_idx, (spt_x, spt_y, qry_x, qry_y) in enumerate(
                                    zip(spt_xs, spt_ys, qry_xs, qry_ys)
                                ):
                                    val_acc_per_loader += model.finetunning(
                                        spt_x, spt_y, qry_x, qry_y
                                    )
                                # Get finetuned parameters per source
                                params.extend(model.finetuned_parameter_list)
                                param_per_epoch[source].extend(model.finetuned_parameter_list)
                                tsne_ranges[source] = slice(start_index, len(params))
                            val_acc_per_loader /= len(val_loader) * len(spt_xs)
                            print("{} acc: {:.2f}%".format(source, val_acc_per_loader * 100))
                            val_acc += val_acc_per_loader
                        val_acc /= len(self.data_loader_val)
                        print("Average val acc: {}%".format(val_acc * 100))

                        # Save finetuned parameters
                        # if num_all_parameters == 0:
                        #     num_all_parameters = len(params) + 1

                end_time = time.monotonic()
                acc /= len(train_loader) * len(spt_xs)
                print(
                    "Acc: {:.2f}%,\tElapsed time: {}".format(
                        acc * 100, timedelta(end_time - start_time)
                    )
                )
            del model

    def train_2tier(self):
        assert self.args.choose_train == True
        for epoch in range(args.start_epoch, args.epochs):
            meta_learner = Meta(args, self.model_config_no_classifier)
            meta_learner.net.append(self.linear_config)  # Append linear layer to the meta_learner
            meta_learner.to(self.device)
            loss = []
            finetuned_meta_parameters = []
            acc = {}  # source: acc
            # Iterate over outer-support-set to get fintuned meta-parameters per source
            for i, (train_source, outer_support_loader) in enumerate(
                self.outer_support_set_train.items()
            ):
                params_per_source = []
                batch = next(iter(outer_support_loader))
                batch = to_device(batch, self.device)
                # TODO: number of tasks in the outer-support-set IS ALREADY 1.
                #       Check why it is the case.
                spt_xs, spt_ys, _, _ = batch  # Outer support set

                finetuned_meta_parameters = meta_learner.finetune_without_query(spt_xs, spt_ys)
                finetuned_meta_parameters = list(zip(*finetuned_meta_parameters))
                finetuned_meta_parameter = []
                for params in finetuned_meta_parameters:
                    average_params = torch.mean(torch.stack(params), axis=0).tolist()
                    finetuned_meta_parameter.append(average_params)

                # finetuned_meta_parameter = torch.mean(
                #     meta_learner.finetune_without_query(spt_xs, spt_ys), axis=0
                # )

                train_loader = self.data_loader_train[train_source]

                batch = next(iter(train_loader))
                batch = to_device(batch, self.device)
                # inner step
                spt_xs, spt_ys, qry_xs, qry_ys = batch
                finetuned_parameters = meta_learner.finetune_without_query(
                    spt_xs, spt_ys, finetuned_meta_parameter
                )
                loss_per_source, acc_per_source = meta_learner.query(
                    qry_xs, qry_ys, finetuned_parameters
                )
                loss.append(loss_per_source)
                acc[train_source] = acc_per_source
            loss = torch.mean(torch.stack(loss))
            meta_learner.meta_optim.zero_grad()
            loss.backward()
            meta_learner.meta_optim.step()
            print("Epoch: [{}]".format(epoch))
            print("Loss: {:.2f}".format(loss))
            # Print accuracy per source and average score
            for source, acc_per_source in acc.items():
                print("{} acc: {:.2f}%".format(source, acc_per_source * 100))
            print("Average acc: {:.2f}%".format(np.mean(list(acc.values())) * 100))

    def evaluate(self):
        pass

    # TODO: 'params' as a dictionary of parameters per source.
    # The parameters are accessed via index, which is not straightforward.
    # additional variables like 'tsne_ranges' will be removed.
    # It is better to use a dictionary.
    # <BJK, 230621, d:within 2 weeks, p:3>
    def plot_parameters(self, params: list):
        # NOTE: This code originally plotted every parameters of all epochs.
        #       The code will be modified to plot only the requested parameters (The last epoch).
        print("Plotting parameters T-SNE")
        colors = ["b", "g", "r", "c", "m", "y", "blueviolet", "magenta", "peru", "lime"]

        # Add meta-parameter to the end
        params.append(
            np.concatenate(
                [torch.flatten(p.detach().cpu()).numpy() for p in model.net.parameters()]
            ).flatten()
        )
        param_per_epoch["Meta-Parameter"].append(
            np.concatenate(
                [torch.flatten(p.detach().cpu()).numpy() for p in model.net.parameters()]
            ).flatten()
        )
        params_all.append(param_per_epoch)

        embedding = umap.UMAP(random_state=1004).fit_transform(np.array(params))
        for i, source in enumerate(tsne_ranges.keys()):
            xs = embedding[tsne_ranges[source], 0]
            ys = embedding[tsne_ranges[source], 1]
            plt.scatter(xs, ys, c=colors[i], label=source)
        plt.scatter(embedding[-1, 0], embedding[-1, 1], c="k", label="Meta-Parameter")
        plt.legend()
        plt.show()
        plt.savefig(
            "{}/{}_umap.png".format(args.plot_dir, epoch * len(train_loader) + episode), dpi=400
        )
        plt.clf()
        embedding = umap.UMAP(random_state=1005).fit_transform(np.array(params))
        for i, source in enumerate(tsne_ranges.keys()):
            xs = embedding[tsne_ranges[source], 0]
            ys = embedding[tsne_ranges[source], 1]
            plt.scatter(xs, ys, c=colors[i], label=source)
        plt.scatter(embedding[-1, 0], embedding[-1, 1], c="k", label="Meta-Parameter")
        plt.legend()
        plt.show()
        plt.savefig(
            "{}/{}_umap_1.png".format(args.plot_dir, epoch * len(train_loader) + episode), dpi=400
        )
        plt.clf()

        tsne = TSNE(n_components=2, random_state=1004)
        embedding = tsne.fit_transform(np.array(params))
        for i, source in enumerate(tsne_ranges.keys()):
            xs = embedding[tsne_ranges[source], 0]
            ys = embedding[tsne_ranges[source], 1]
            plt.scatter(xs, ys, c=colors[i], label=source)
        plt.scatter(embedding[-1, 0], embedding[-1, 1], c="k", label="Meta-Parameter")
        plt.legend()
        plt.show()
        plt.savefig("{}/{}.png".format(args.plot_dir, epoch * len(train_loader) + episode), dpi=400)
        plt.clf()
        del tsne
        # Test 3d
        tsne = TSNE(n_components=3, random_state=1004)
        embedding = tsne.fit_transform(np.array(params))
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for i, source in enumerate(tsne_ranges.keys()):
            xs = embedding[tsne_ranges[source], 0]
            ys = embedding[tsne_ranges[source], 1]
            zs = embedding[tsne_ranges[source], 2]
            ax.scatter(xs, ys, zs, c=colors[i], label=source)
        ax.scatter(embedding[-1, 0], embedding[-1, 1], c="k", label="Meta-Parameter")
        plt.legend()
        plt.show()
        plt.savefig(
            "{}/{}_3d.png".format(args.plot_dir, epoch * len(train_loader) + episode), dpi=400
        )
        plt.clf()
        del tsne
        embedding = umap.UMAP(n_components=3, random_state=1004).fit_transform(np.array(params))
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for i, source in enumerate(tsne_ranges.keys()):
            xs = embedding[tsne_ranges[source], 0]
            ys = embedding[tsne_ranges[source], 1]
            zs = embedding[tsne_ranges[source], 2]
            ax.scatter(xs, ys, zs, c=colors[i], label=source)
        ax.scatter(embedding[-1, 0], embedding[-1, 1], c="k", label="Meta-Parameter")
        plt.legend()
        plt.show()
        plt.savefig(
            "{}/{}_3d_umap.png".format(args.plot_dir, epoch * len(train_loader) + episode), dpi=400
        )
        plt.clf()


def evaluate(data_loaders, model, criterion, device, seed=None, ep=None, maml=None):
    print("Test Accuracy")
    acc = 0
    for j, (source, loader) in enumerate(data_loaders.items()):
        acc_per_loader = 0
        for batch in loader:
            batch = to_device(batch, device)
            spt_xs, spt_ys, qry_xs, qry_ys = batch
            for task_idx, (spt_x, spt_y, qry_x, qry_y) in enumerate(
                zip(spt_xs, spt_ys, qry_xs, qry_ys)
            ):
                acc_per_loader += model.finetunning(spt_x, spt_y, qry_x, qry_y)
        acc_per_loader /= len(loader) * len(spt_xs)
        print(" * {} acc: {:.2f}%".format(source, acc_per_loader * 100))
        acc += acc_per_loader
    acc /= len(data_loader_val)
    print("Average test acc: {:.2f}%".format(acc * 100))
    return acc


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.two_tier:
        args.choose_train = True

    trainer = Trainer(args)
    # trainer.train()
    trainer.train_2tier()

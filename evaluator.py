import sys
import random
import numpy as np
import time
import tqdm
import torch
import torch.backends.cudnn as cudnn
import json
import umap
import matplotlib.pyplot as plt

from datetime import timedelta
from collections import defaultdict
from itertools import chain
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE

from timm.data import Mixup

from engine import train_one_epoch
import utils.deit_util as utils
from utils import AverageMeter, to_device
from datasets import get_loaders, get_sets
from utils.args import get_args_parser
from models import get_model
from models.meta import Meta


class Evaluator(object):
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
        self.output_dir = Path(args.output_dir)
        if utils.is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(" ".join(sys.argv) + "\n")
        self.model_description_json = {}
        self.writer = SummaryWriter(log_dir=str(self.output_dir))

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

        # self.metric_logger = utils.MetricLogger(delimiter="  ")
        # self.metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        # self.metric_logger.add_meter("n_ways", utils.SmoothedValue(window_size=1, fmt="{value:d}"))
        # self.metric_logger.add_meter("n_imgs", utils.SmoothedValue(window_size=1, fmt="{value:d}"))

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

    def train_2tier(self):
        assert self.args.choose_train == True
        meta_learner = Meta(args, self.model_config_no_classifier)
        meta_learner.net.append(self.linear_config)  # Append linear layer to the meta_learner
        meta_learner.to(self.device)

        for epoch in range(args.start_epoch, args.epochs):
            # Iterate for minimum length of the outer-support-set
            num_episodes = min([len(loader) for loader in self.outer_support_set_train.values()])
            for episode in range(num_episodes):
                param_log = defaultdict(list)
                param_log["finetuned_meta_parameters"] = {}
                param_log["global"].append(
                    np.concatenate(
                        [
                            torch.flatten(p.detach().cpu()).numpy()
                            for p in meta_learner.net.parameters()
                        ]
                    ).flatten()
                )

                meta_learner.train()
                loss = []
                finetuned_meta_parameters = []
                acc = {}  # source: acc
                # Iterate over outer-support-set to get fintuned meta-parameters per source
                episode_over_total_epochs = episode + num_episodes * epoch
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
                        average_params = torch.mean(torch.stack(params), axis=0)
                        finetuned_meta_parameter.append(average_params)

                    param_log["finetuned_meta_parameters"][train_source] = [
                        np.concatenate(
                            [
                                torch.flatten(p.detach().cpu()).numpy()
                                for p in finetuned_meta_parameter
                            ]
                        ).flatten()
                    ]

                    train_loader = self.data_loader_train[train_source]

                    batch = next(iter(train_loader))
                    batch = to_device(batch, self.device)
                    # inner step
                    spt_xs, spt_ys, qry_xs, qry_ys = batch
                    if args.one_tier:
                        finetuned_meta_parameter = None
                    finetuned_parameters = meta_learner.finetune_without_query(
                        spt_xs, spt_ys, phi=finetuned_meta_parameter
                    )
                    for param in finetuned_parameters:
                        param_log[train_source].append(
                            np.concatenate(
                                [torch.flatten(p.detach().cpu()).numpy() for p in param]
                            ).flatten()
                        )

                    loss_per_source, acc_per_source = meta_learner.query(
                        qry_xs, qry_ys, finetuned_parameters
                    )
                    loss.append(loss_per_source)
                    acc[train_source] = acc_per_source
                    self.writer.add_scalar(
                        "Loss/train/{}".format(train_source),
                        loss_per_source,
                        episode_over_total_epochs,
                    )
                    self.writer.add_scalar(
                        "Acc/train/{}".format(train_source),
                        acc_per_source,
                        episode_over_total_epochs,
                    )
                loss = torch.mean(torch.stack(loss))
                meta_learner.meta_optim.zero_grad()
                loss.backward()
                meta_learner.meta_optim.step()
                average_accuarcy = np.mean(list(acc.values()))
                self.writer.add_scalar("Loss/train/avereage", loss, episode_over_total_epochs)
                self.writer.add_scalar(
                    "Acc/train/average", average_accuarcy, episode_over_total_epochs
                )
                if episode_over_total_epochs % 10 == 0:
                    print("Episode: [{}]".format(episode_over_total_epochs))
                    print("Loss: {:.2f}".format(loss))
                    # Print accuracy per source and average score
                    for source, acc_per_source in acc.items():
                        print("{} acc: {:.2f}%".format(source, acc_per_source * 100))
                    print("Average acc: {:.2f}%".format(average_accuarcy * 100))
                    # Plot parameters
                    # self.plot_parameters(param_log, episode_over_total_epochs)
                if episode_over_total_epochs % 100 == 0:
                    self.evaluate(meta_learner)

    def evaluate(self, meta_learner):
        print("Evaluate")
        meta_learner.eval()
        # Evaluate
        val_acc = 0
        val_loss = 0
        params = []
        tsne_ranges = {}
        param_per_epoch = defaultdict(list)
        for i, (val_source, outer_support_loader) in enumerate(self.outer_support_set_val.items()):
            params_per_source = []
            batch = next(iter(outer_support_loader))
            batch = to_device(batch, self.device)
            outer_spt_xs, outer_spt_ys, _, _ = batch  # Outer support set

            finetuned_meta_parameters = meta_learner.finetune_without_query(
                outer_spt_xs, outer_spt_ys
            )
            finetuned_meta_parameters = list(zip(*finetuned_meta_parameters))
            finetuned_meta_parameter = []
            for params in finetuned_meta_parameters:
                average_params = torch.mean(torch.stack(params), axis=0)
                finetuned_meta_parameter.append(average_params)

            val_loader = self.data_loader_val[val_source]

            batch = next(iter(val_loader))
            batch = to_device(batch, self.device)
            # inner step
            spt_xs, spt_ys, qry_xs, qry_ys = batch
            if args.one_tier:
                finetuned_meta_parameter = None
            finetuned_parameters = meta_learner.finetune_without_query(
                spt_xs, spt_ys, phi=finetuned_meta_parameter
            )
            # finetuned_parameters = meta_learner.finetune_without_query(spt_xs, spt_ys)
            val_loss_per_source, val_acc_per_source = meta_learner.query(
                qry_xs, qry_ys, finetuned_parameters
            )
            # val_acc_per_source /= len(val_loader) * len(spt_xs)
            self.writer.add_scalar(
                "Loss/val/{}".format(val_source), val_loss_per_source, episode_over_total_epochs
            )
            self.writer.add_scalar(
                "Acc/val/{}".format(val_source), val_acc_per_source, episode_over_total_epochs
            )
            print("{} acc: {:.2f}%".format(val_source, val_acc_per_source * 100))
            val_acc += val_acc_per_source
            val_loss += val_loss_per_source
        val_acc /= len(self.data_loader_val)
        val_loss /= len(self.data_loader_val)
        self.writer.add_scalar("Loss/val/average", val_loss, episode_over_total_epochs)
        self.writer.add_scalar("Acc/val/average", val_acc, episode_over_total_epochs)
        print("Average val acc: {}%".format(val_acc * 100))
        print("Average val loss: {}".format(val_loss))

    def save_parameters(self, params: list, epoch: int, episode: int):
        pass

    def plot_parameters(self, params: dict, timestep, mode="tsne"):
        # NOTE: This code originally plotted every parameters of all epochs.
        #       The code will be modified to plot only the requested parameters (The last epoch).
        print("Plotting parameters T-SNE")
        colors = ["b", "g", "r", "c", "m", "y", "blueviolet", "magenta", "peru", "lime"]
        if mode == "tsne":
            model = TSNE(n_components=2, random_state=1004)
        elif mode == "umap":
            model = umap.UMAP(random_state=1004)
        # Convert parameters to a flat list
        param_list = []
        color_range = {}
        meta_color_range = {}
        color_dict = {}
        for key in params.keys():
            if type(params[key]) == list:
                color_range[key] = slice(len(param_list), len(param_list) + len(params[key]))
                param_list += params[key]
                color_dict[key] = color_dict.get(key, colors[len(color_dict)])
            elif type(params[key]) == dict:
                for source in params[key].keys():
                    meta_color_range[source] = slice(
                        len(param_list), len(param_list) + len(params[key][source])
                    )
                    param_list += params[key][source]
                    color_dict[source] = color_dict.get(source, colors[len(color_dict)])

        param_list = np.array(param_list)
        embedding = model.fit_transform(param_list)

        # embedding = defaultdict(list)

        for i, (key, value) in enumerate(params.items()):
            if type(value) == list:
                xs, ys = embedding[color_range[key], 0], embedding[color_range[key], 1]
                plt.scatter(xs, ys, c=color_dict[key], label=key)
            elif type(value) == dict:
                for j, (source, source_value) in enumerate(value.items()):
                    xs, ys = (
                        embedding[meta_color_range[source], 0],
                        embedding[meta_color_range[source], 1],
                    )
                    plt.scatter(
                        xs,
                        ys,
                        c=color_dict[source],
                        label=source,
                        marker="x",
                        linewidth=1,
                        edgecolor="k",
                    )
        plt.legend(
            bbox_to_anchor=(1.04, 1),
            borderaxespad=0,
        )
        plt.savefig(
            "{}/{}_{}{}.png".format(
                args.plot_dir, timestep, mode, ("_one" if args.one_tier else "")
            ),
            dpi=400,
            bbox_inches="tight",
        )
        plt.clf()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.two_tier:
        args.choose_train = True

    evaluator = Evaluator(args)

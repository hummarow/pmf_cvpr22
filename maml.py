import sys
import random
import numpy as np
import time
import tqdm
import torch
import torch.backends.cudnn as cudnn
import json
import umap
import copy
import shlex
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


class Trainer(object):
    def __init__(self, args):
        args.distributed = True
        self.device = torch.device(args.device)
        self.seed = args.seed + utils.get_rank()
        self.seed = random.choice(range(self.seed, self.seed + 10))
        args.seed = self.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        cudnn.benchmark = True

        self.plot_dir = Path(args.plot_dir)
        self.log_dir = Path(args.log_dir)
        self.output_dir = Path(args.output_dir)
        if utils.is_main_process():
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / "log.txt").open("a") as f:
                if not args.eval:
                    f.write(" ".join(sys.argv) + "\n")
        self.model_description_json = {}
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        if utils.is_main_process():
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Data loaders
        self.num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        (self.data_loader_train, self.outer_support_set_train), (
            self.data_loader_val,
            self.outer_support_set_val,
        ) = get_loaders(args, self.num_tasks, global_rank)
        # (_, _), (
        #     self.data_loader_test,
        #     self.outer_support_set_test,
        # ) = get_loaders(args, self.num_tasks, global_rank)
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
        args.linear_shape = _linear_shape

        self.args = args
        print(self.args)

    def _train_one_batch(self, batch, model, device, phi=None):
        batch = to_device(batch, device)
        spt_xs, spt_ys, qry_xs, qry_ys = batch
        acc_per_episode = model(spt_xs, spt_ys, qry_xs, qry_ys, phi=phi)
        return acc_per_episode

    def train(self):
        for i, (train_source, train_loader) in enumerate(self.data_loader_train.items()):
            if args.choose_train:
                print("Training on {}".format(train_source))
            model = Meta(args, self.model_config_no_classifier)
            # Append linear layer to the model
            model.net.append(self.linear_config)
            model.to(self.device)
            for epoch in range(args.start_epoch, args.epochs):
                header = "Epoch: [{}]".format(epoch)
                print(header)
                start_time = time.monotonic()
                print_freq = 10
                acc = 0
                # for batch in metric_logger.log_every(self.data_loader_train, print_freq, header):
                print(len(train_loader))
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
        meta_learner = Meta(args, self.model_config_no_classifier)
        meta_learner.net.append(self.linear_config)  # Append linear layer to the meta_learner
        meta_learner.to(self.device)
        outer_update_fn = (
            meta_learner.contrastive
            if args.train_method == "contrastive"
            else meta_learner.finetune_without_query
        )
        inner_update_fn = meta_learner.finetune_without_query

        val_losses = []
        val_accs = []

        for epoch in range(args.start_epoch, args.epochs):
            # Iterate for minimum length of the outer-support-set
            num_episodes = min([len(loader) for loader in self.outer_support_set_train.values()])
            print(epoch, num_episodes)
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
                average_outer_loss = 0
                average_inner_loss = 0
                params_per_source = {}
                for i, (train_source, outer_support_loader) in enumerate(
                    self.outer_support_set_train.items()
                ):
                    print(train_source)
                    # 230801 Copy the model for each source
                    learner = copy.deepcopy(meta_learner)
                    learner.train()

                    outer_update_fn = (
                        learner.contrastive
                        if args.train_method == "contrastive"
                        else learner.finetune_without_query
                    )
                    inner_update_fn = learner.finetune_without_query

                    # print(train_source)
                    if args.one_tier:
                        finetuned_meta_parameter = None
                    else:
                        batch = next(iter(outer_support_loader))
                        batch = to_device(batch, self.device)
                        outer_spt_xs, outer_spt_ys, _, _ = batch  # Outer support set
                        breakpoint()
                        finetuned_meta_parameters, outer_loss = outer_update_fn(
                            outer_spt_xs, outer_spt_ys
                        )

                        # Get average of finetuned meta-parameters
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
                    breakpoint()
                    finetuned_parameters, inner_loss = inner_update_fn(
                        spt_xs, spt_ys, phi=finetuned_meta_parameter, inner=True
                    )
                    for param in finetuned_parameters:
                        param_log[train_source].append(
                            np.concatenate(
                                [torch.flatten(p.detach().cpu()).numpy() for p in param]
                            ).flatten()
                        )

                    loss_per_source, acc_per_source = learner.query(
                        qry_xs, qry_ys, finetuned_parameters, eval=False
                    )
                    # loss_per_source, acc_per_source = meta_learner.query(
                    #     qry_xs,
                    #     qry_ys,
                    #     [finetuned_meta_parameter] * len(finetuned_parameters),
                    #     eval=False,
                    # )

                    learner.meta_optim.zero_grad()
                    loss_per_source.backward()

                    learner.meta_optim.step()
                    params_per_source[train_source] = learner.net.parameters()

                    loss.append(loss_per_source)
                    acc[train_source] = acc_per_source
                    self.writer.add_scalar(
                        "Loss/train/{}".format(train_source),
                        loss_per_source,
                        episode_over_total_epochs,
                    )
                    self.writer.add_scalar(
                        "Outer/train/{}".format(train_source),
                        outer_loss,
                        episode_over_total_epochs,
                    )
                    self.writer.add_scalar(
                        "Inner/train/{}".format(train_source),
                        inner_loss,
                        episode_over_total_epochs,
                    )
                    self.writer.add_scalar(
                        "Acc/train/{}".format(train_source),
                        acc_per_source,
                        episode_over_total_epochs,
                    )
                    average_outer_loss += outer_loss
                    average_inner_loss += inner_loss
                loss = torch.mean(torch.stack(loss))
                # meta_learner.meta_optim.zero_grad()
                # breakpoint()
                # loss.backward()
                # 수동으로
                # meta_learner.meta_optim.step()

                # Average updated parameters
                average_parameter = []
                for params in zip(*params_per_source.values()):
                    average_params = torch.mean(torch.stack(params), axis=0)
                    average_parameter.append(average_params)
                # Insert the average_parameter into meta_learner.net
                for i, param in enumerate(meta_learner.net.parameters()):
                    param.data = average_parameter[i].data

                average_outer_loss /= len(self.outer_support_set_train)
                average_inner_loss /= len(self.outer_support_set_train)
                average_accuarcy = np.mean(list(acc.values()))
                self.writer.add_scalar("Loss/train/average", loss, episode_over_total_epochs)
                self.writer.add_scalar(
                    "Outer/train/average", average_outer_loss, episode_over_total_epochs
                )
                self.writer.add_scalar(
                    "Inner/train/average", average_inner_loss, episode_over_total_epochs
                )
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
                if episode_over_total_epochs % 100 == 0:
                    val_acc, val_loss = self.evaluate(meta_learner, episode_over_total_epochs)
                    if not val_accs or val_acc > max(val_accs):
                        torch.save(
                            meta_learner.state_dict(),
                            self.output_dir / "best_model.pth",
                        )

                    val_accs.append(val_acc)
                    # Plot parameters
                    # self.plot_parameters(param_log, episode_over_total_epochs)

    def evaluate(self, meta_learner, epoch):
        print("Evaluate")
        meta_learner.eval()
        # Evaluate
        val_acc = 0
        val_loss = 0
        params = []
        tsne_ranges = {}
        param_per_epoch = defaultdict(list)
        outer_update_fn = (
            meta_learner.contrastive
            if args.train_method == "contrastive"
            else meta_learner.finetune_without_query
        )
        inner_update_fn = meta_learner.finetune_without_query
        outer_average_loss = 0
        inner_average_loss = 0
        for i, (val_source, outer_support_loader) in enumerate(self.outer_support_set_val.items()):
            params_per_source = []
            batch = next(iter(outer_support_loader))
            batch = to_device(batch, self.device)
            outer_spt_xs, outer_spt_ys, _, _ = batch  # Outer support set

            finetuned_meta_parameters, outer_loss = outer_update_fn(outer_spt_xs, outer_spt_ys)

            # finetuned_meta_parameters = meta_learner.finetune_without_query(
            #     outer_spt_xs,
            #     outer_spt_ys,
            # )
            outer_average_loss += outer_loss
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
            # 이런 식으로 해도 되나?
            finetuned_parameters, inner_loss = inner_update_fn(
                spt_xs, spt_ys, phi=finetuned_meta_parameter, inner=True
            )
            inner_average_loss += inner_loss
            # finetuned_parameters = meta_learner.finetune_without_query(spt_xs, spt_ys)
            val_loss_per_source, val_acc_per_source = meta_learner.query(
                qry_xs, qry_ys, finetuned_parameters, eval=True
            )
            # val_acc_per_source /= len(val_loader) * len(spt_xs)
            if not args.eval:
                self.writer.add_scalar("Loss/val/{}".format(val_source), val_loss_per_source, epoch)
                self.writer.add_scalar("Outer/val/{}".format(val_source), outer_loss, epoch)
                self.writer.add_scalar("Inner/val/{}".format(val_source), inner_loss, epoch)
                self.writer.add_scalar("Acc/val/{}".format(val_source), val_acc_per_source, epoch)
            print("{} acc: {:.2f}%".format(val_source, val_acc_per_source * 100))
            val_acc += val_acc_per_source
            val_loss += val_loss_per_source
        val_acc /= len(self.data_loader_val)
        val_loss /= len(self.data_loader_val)
        if not args.eval:
            self.writer.add_scalar("Loss/val/average", val_loss, epoch)
            self.writer.add_scalar("Outer/val/average", outer_average_loss, epoch)
            self.writer.add_scalar("Inner/val/average", inner_average_loss, epoch)
            self.writer.add_scalar("Acc/val/average", val_acc, epoch)
        print("Average val acc: {}%".format(val_acc * 100))
        print("Average val loss: {}".format(val_loss))
        return val_acc, val_loss

    def save_parameters(self, params: list, epoch: int, episode: int):
        pass

    def plot_parameters(self, params: dict, timestep, mode="tsne"):
        # NOTE: This code originally plotted every parameters of all epochs.
        #       The code will be modified to plot only the requested parameters (The last epoch).
        print("Plotting parameters T-SNE")
        colors = ["b", "g", "r", "c", "m", "y", "blueviolet", "magenta", "peru", "lime"]
        model = TSNE(n_components=3, random_state=1004, perplexity=5)
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

        model = TSNE(n_components=3, random_state=1004, perplexity=5)
        embedding = model.fit_transform(param_list)
        x, y, z = zip(*embedding)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for i, (key, value) in enumerate(params.items()):
            if type(value) == list:
                xs, ys, zs = x[color_range[key]], y[color_range[key]], z[color_range[key]]
                ax.scatter(xs, ys, zs, c=color_dict[key], label=key)
            elif type(value) == dict:
                for j, (source, source_value) in enumerate(value.items()):
                    xs, ys, zs = (
                        x[meta_color_range[source]],
                        y[meta_color_range[source]],
                        z[meta_color_range[source]],
                    )
                    ax.scatter(
                        xs,
                        ys,
                        zs,
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
            "{}/{}.png".format(args.plot_dir, timestep),
            dpi=400,
            bbox_inches="tight",
        )
        plt.clf()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    args.output_dir += args.name
    args.plot_dir += args.name
    args.log_dir += args.name
    if args.eval:
        with open(Path(args.log_dir) / "log.txt", "r") as fp:
            shline = fp.readlines()[-1]
            shline = " ".join(shlex.split(shline)[1:])
            args = parser.parse_args(shlex.split(shline))
            args.output_dir += args.name
            args.plot_dir += args.name
            args.log_dir += args.name
            args.eval = True
    if args.two_tier:
        args.choose_train = True

    trainer = Trainer(args)
    # trainer.train_separately(args)
    if args.eval:
        # Load model
        print("Load model")
        model = Meta(args, trainer.model_config_no_classifier)
        model.net.append(trainer.linear_config)
        model.to(trainer.device)
        model.load_state_dict(torch.load(trainer.output_dir / "best_model.pth"))
        model.eval()
        trainer.evaluate(model, 0)
    else:
        if args.two_tier:
            trainer.train_2tier()
        else:
            trainer.train()

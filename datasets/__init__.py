import os
import random
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

from .samplers import RASampler
from .episodic_dataset import EpisodeDataset, EpisodeJSONDataset
from .meta_val_dataset import MetaValDataset
from .meta_h5_dataset import FullMetaDatasetH5
from .meta_dataset.utils import Split


def get_sets(args):
    if args.dataset == "cifar_fs":
        from .cifar_fs import dataset_setting
    elif args.dataset == "cifar_fs_elite":  # + elite data augmentation
        from .cifar_fs_elite import dataset_setting
    elif args.dataset == "mini_imagenet":
        from .mini_imagenet import dataset_setting
    elif args.dataset == "meta_dataset":
        if args.eval:
            trainSet = valSet = None
            testSet = FullMetaDatasetH5(args, Split.TEST)
            if args.two_tier:
                testOuterSupportSet = FullMetaDatasetH5(args, Split.TEST)
        else:
            # Not checked when multiprocess.
            trainSet = {}
            if args.two_tier:
                trainOuterSupportSet = {}
            if args.choose_train:
                sources = args.base_sources
                for source in sources:
                    args.base_sources = [source]
                    trainSet[source] = FullMetaDatasetH5(args, Split.TRAIN)
                    if args.two_tier:
                        trainOuterSupportSet[source] = FullMetaDatasetH5(args, Split.TRAIN)
                args.base_sources = sources
            else:
                trainSet["combined"] = FullMetaDatasetH5(args, Split.TRAIN)
                if args.two_tier:
                    trainOuterSupportSet["combined"] = FullMetaDatasetH5(args, Split.TRAIN)
            valSet = {}
            if args.two_tier:
                valOuterSupportSet = {}
            for source in args.val_sources:
                valSet[source] = MetaValDataset(
                    os.path.join(
                        args.data_path, source, f"val_ep{args.nValEpisode}_img{args.image_size}.h5"
                    ),
                    num_episodes=args.nValEpisode,
                )
                if args.two_tier:
                    valOuterSupportSet[source] = MetaValDataset(
                        os.path.join(
                            args.data_path,
                            source,
                            f"val_ep{args.nValEpisode}_img{args.image_size}.h5",
                        ),
                        num_episodes=args.nValEpisode,
                    )
            testSet = None
            if args.two_tier:
                testOuterSupportSet = None
        if not args.two_tier:
            trainOuterSupportSet = valOuterSupportSet = testOuterSupportSet = None
        return (
            trainSet,
            valSet,
            testSet,
            trainOuterSupportSet,
            valOuterSupportSet,
            testOuterSupportSet,
        )
    elif args.dataset == "bscdfsl":
        if args.eval:
            trainSet = valSet = None

        return trainSet, valSet, testSet
    else:
        raise ValueError(f"{args.dataset} is not supported.")

    # If not meta_dataset
    (
        trainTransform,
        valTransform,
        inputW,
        inputH,
        trainDir,
        valDir,
        testDir,
        episodeJson,
        nbCls,
    ) = dataset_setting(args.nSupport, args.img_size)

    trainSet = EpisodeDataset(
        imgDir=trainDir,
        nCls=args.nClsEpisode,
        nSupport=args.nSupport,
        nQuery=args.nQuery,
        transform=trainTransform,
        inputW=inputW,
        inputH=inputH,
        nEpisode=args.nEpisode,
    )

    outerSupportSet = EpisodeDataset(
        imgDir=trainDir,
        nCls=args.nClsEpisode,
        nSupport=args.nSupport,
        nQuery=0,  # NOTE: no query set
        transform=valTransform,
        inputW=inputW,
        inputH=inputH,
        nEpisode=args.nEpisode,
    )
    valSet = EpisodeJSONDataset(episodeJson, valDir, inputW, inputH, valTransform)

    testSet = EpisodeDataset(
        imgDir=testDir,
        nCls=args.nClsEpisode,
        nSupport=args.nSupport,
        nQuery=args.nQuery,
        transform=valTransform,
        inputW=inputW,
        inputH=inputH,
        nEpisode=args.nEpisode,
    )

    return trainSet, valSet, testSet, None, None, None


def task_collate(samples):
    support_images, support_labels, query_images, query_labels = zip(*samples)
    return (support_images, support_labels, query_images, query_labels)


def get_loaders(args, num_tasks, global_rank):
    # datasets
    if args.eval:
        _, _, dataset_vals, _, _, outer_support_set_vals = get_sets(args)
    else:
        (
            dataset_trains,
            dataset_vals,
            _,
            outer_support_set_trains,
            outer_support_set_vals,
            _,
        ) = get_sets(args)
    # Worker init function
    if "meta_dataset" in args.dataset:  # meta_dataset & meta_dataset_h5
        # worker_init_fn = partial(worker_init_fn_, seed=args.seed)
        # worker_init_fn = lambda _: np.random.seed()
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    else:
        worker_init_fn = None

    # Val loader
    # NOTE: meta-dataset has separate val-set per domain
    if not isinstance(dataset_vals, dict):
        dataset_vals = {"single": dataset_vals}
        if args.two_tier:
            outer_support_set_vals = {"single": outer_support_set_vals}

    data_loader_val = {}
    if args.two_tier:
        outer_support_data_loader_val = {}
    else:
        outer_support_data_loader_val = None

    for j, (source, dataset_val) in enumerate(dataset_vals.items()):
        if args.distributed:
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print(
                        "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                        "This will slightly alter validation results as extra duplicate entries are added to achieve "
                        "equal num of samples per-process."
                    )
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
                )
                if args.two_tier:
                    outer_support_set_vals[source] = torch.utils.data.DistributedSampler(
                        outer_support_set_vals[source],
                        num_replicas=num_tasks,
                        rank=global_rank,
                        shuffle=False,
                    )
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                if args.two_tier:
                    outer_support_set_vals[source] = torch.utils.data.SequentialSampler(
                        outer_support_set_vals[source]
                    )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            if args.two_tier:
                outer_support_set_vals[source] = torch.utils.data.SequentialSampler(
                    outer_support_set_vals[source]
                )

        generator = torch.Generator()
        generator.manual_seed(args.seed + 10000 + j)

        data_loader_val[source] = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            # batch_size=args.task_num,  # Number of tasks when evaluating need to be checked again.
            batch_size=1,
            num_workers=3,  # more workers can take too much CPU
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        if args.two_tier:
            outer_support_data_loader_val[source] = torch.utils.data.DataLoader(
                outer_support_set_vals[source],
                batch_size=1,
                num_workers=3,  # more workers can take too much CPU
                pin_memory=args.pin_mem,
                drop_last=False,
                worker_init_fn=worker_init_fn,
                generator=generator,
            )

    if "single" in dataset_vals:
        data_loader_val = data_loader_val["single"]

    if args.eval:
        return None, data_loader_val

    # Train loader
    data_loader_train = {}
    if args.two_tier:
        outer_support_data_loader_train = {}
    else:
        outer_support_data_loader_train = None
    for j, (source, dataset_train) in enumerate(dataset_trains.items()):
        if args.choose_train:
            sampler_train = torch.utils.data.SequentialSampler(dataset_trains)
            if args.two_tier:
                outer_support_sampler_train = torch.utils.data.RandomSampler(
                    outer_support_set_trains
                )
        else:
            if args.distributed:
                if args.repeated_aug:  # (by default OFF)
                    sampler_train = RASampler(
                        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                    )
                    if args.two_tier:
                        outer_support_sampler_train = RASampler(
                            outer_support_set_trains[source],
                            num_replicas=num_tasks,
                            rank=global_rank,
                            shuffle=True,
                        )
                else:
                    sampler_train = torch.utils.data.DistributedSampler(
                        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                    )
                    if args.two_tier:
                        outer_support_sampler_train = torch.utils.data.DistributedSampler(
                            outer_support_set_trains[source],
                            num_replicas=num_tasks,
                            rank=global_rank,
                            shuffle=True,
                        )
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                if args.two_tier:
                    outer_support_sampler_train = torch.utils.data.RandomSampler(
                        outer_support_set_trains[source]
                    )

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        data_loader_train[source] = torch.utils.data.DataLoader(
            dataset_train,
            # sampler=sampler_train,
            batch_size=args.task_num,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            generator=generator,
            collate_fn=task_collate,
        )
        if args.two_tier:
            outer_support_data_loader_train[source] = torch.utils.data.DataLoader(
                outer_support_set_trains[source],
                sampler=outer_support_sampler_train,
                batch_size=1,  # One finetuned_meta_parameter per source
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
                worker_init_fn=worker_init_fn,
                generator=generator,
                collate_fn=task_collate,
            )

    return (data_loader_train, outer_support_data_loader_train), (
        data_loader_val,
        outer_support_data_loader_val,
    )


def get_bscd_loader(dataset="EuroSAT", test_n_way=5, n_shot=5, image_size=224):
    iter_num = 600
    n_query = 15
    few_shot_params = dict(n_way=test_n_way, n_support=n_shot)

    if dataset == "EuroSAT":
        from .cdfsl.EuroSAT_few_shot import SetDataManager
    elif dataset == "ISIC":
        from .cdfsl.ISIC_few_shot import SetDataManager
    elif dataset == "CropDisease":
        from .cdfsl.CropDisease_few_shot import SetDataManager
    elif dataset == "ChestX":
        from .cdfsl.ChestX_few_shot import SetDataManager
    else:
        raise ValueError(f"Datast {dataset} is not supported.")

    datamgr = SetDataManager(image_size, n_eposide=iter_num, n_query=n_query, **few_shot_params)
    novel_loader = datamgr.get_data_loader(aug=False)

    def _loader_wrap():
        for x, y in novel_loader:
            SupportTensor = x[:, :n_shot].contiguous().view(1, test_n_way * n_shot, *x.size()[2:])
            QryTensor = x[:, n_shot:].contiguous().view(1, test_n_way * n_query, *x.size()[2:])
            SupportLabel = torch.from_numpy(np.repeat(range(test_n_way), n_shot)).view(
                1, test_n_way * n_shot
            )
            QryLabel = torch.from_numpy(np.repeat(range(test_n_way), n_query)).view(
                1, test_n_way * n_query
            )

            yield SupportTensor, SupportLabel, QryTensor, QryLabel

    class _DummyGenerator:
        def manual_seed(self, seed):
            pass

    class _Loader(object):
        def __init__(self):
            self.iterable = _loader_wrap()
            # NOTE: the following are required by engine.py:_evaluate()
            self.dataset = self
            self.generator = _DummyGenerator()

        def __len__(self):
            return len(novel_loader)

        def __iter__(self):
            return self.iterable

    return _Loader()

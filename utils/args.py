import argparse
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser("Few-shot learning script", add_help=False)
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    # General
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.set_defaults(fp16=True)
    parser.add_argument("--output_dir", default="saved_model/", help="path where to save")
    parser.add_argument("--plot_dir", default="images/", help="path where to save pictures")
    parser.add_argument("--device", default="cuda", help="cuda:gpu_id for single GPU training")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--choose_train", action="store_true")
    parser.add_argument("--two_tier", action="store_true")
    parser.add_argument("--train_method", default="supervised", type=str)  # supervised, contrastive

    # Dataset parameters
    parser.add_argument(
        "--data-path",
        # default="/datasets01/imagenet_full_size/061417/",
        default="/home/bjk/hdd1/records/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--pretrained-checkpoint-path",
        default=".",
        type=str,
        help="path which contains the directories pretrained_ckpts and pretrained_ckpts_converted",
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar_fs_elite", "cifar_fs", "mini_imagenet", "meta_dataset"],
        default="meta_dataset",
        help="Which few-shot dataset.",
    )

    # Few-shot parameters (Mini-ImageNet & CIFAR-FS)
    parser.add_argument(
        "--nClsEpisode", default=5, type=int, help="Number of categories in each episode."
    )
    parser.add_argument(
        "--nSupport", default=1, type=int, help="Number of samples per category in the support set."
    )
    parser.add_argument(
        "--nQuery", default=15, type=int, help="Number of samples per category in the query set."
    )
    parser.add_argument(
        "--nValEpisode", default=120, type=int, help="Number of episodes for validation."
    )
    parser.add_argument(
        "--nEpisode", default=2000, type=int, help="Number of episodes for training / testing."
    )

    # MetaDataset parameters
    parser.add_argument(
        "--image_size", type=int, default=128, help="Images will be resized to this value"
    )
    parser.add_argument(
        "--base_sources",
        nargs="+",
        default=[
            "aircraft",
            "cu_birds",
            "dtd",
            "fungi",
            "ilsvrc_2012",
            "omniglot",
            "quickdraw",
            "vgg_flower",
        ],
        help="List of datasets to use for training",
    )
    parser.add_argument(
        "--val_sources",
        nargs="+",
        default=[
            "aircraft",
            "cu_birds",
            "dtd",
            "fungi",
            "ilsvrc_2012",
            "omniglot",
            "quickdraw",
            "vgg_flower",
        ],
        help="List of datasets to use for validation",
    )
    parser.add_argument(
        "--test_sources",
        nargs="+",
        default=[
            "traffic_sign",
            "mscoco",
            "ilsvrc_2012",
            "omniglot",
            "aircraft",
            "cu_birds",
            "dtd",
            "quickdraw",
            "fungi",
            "vgg_flower",
        ],
        help="List of datasets to use for meta-testing",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether or not to shuffle data for TFRecordDataset",
    )
    parser.add_argument(
        "--train_transforms",
        nargs="+",
        default=["random_resized_crop", "jitter", "random_flip", "to_tensor", "normalize"],
        help="Transforms applied to training data",
    )
    parser.add_argument(
        "--test_transforms",
        nargs="+",
        default=["resize", "center_crop", "to_tensor", "normalize"],
        help="Transforms applied to test data",
    )
    parser.add_argument(
        "--num_ways", type=int, default=None, help="Set it if you want a fixed # of ways per task"
    )
    parser.add_argument(
        "--num_support",
        type=int,
        default=None,
        help="Set it if you want a fixed # of support samples per class",
    )
    parser.add_argument(
        "--num_query",
        type=int,
        default=None,
        help="Set it if you want a fixed # of query samples per class",
    )
    parser.add_argument("--min_ways", type=int, default=5, help="Minimum # of ways per task")
    parser.add_argument(
        "--max_ways_upper_bound", type=int, default=50, help="Maximum # of ways per task"
    )
    parser.add_argument("--max_num_query", type=int, default=10, help="Maximum # of query samples")
    parser.add_argument(
        "--max_support_set_size", type=int, default=500, help="Maximum # of support samples"
    )
    parser.add_argument(
        "--max_support_size_contrib_per_class",
        type=int,
        default=100,
        help="Maximum # of support samples per class",
    )
    parser.add_argument(
        "--min_examples_in_class",
        type=int,
        default=0,
        help="Classes that have less samples will be skipped",
    )
    parser.add_argument(
        "--min_log_weight",
        type=float,
        default=np.log(0.5),
        help="Do not touch, used to randomly sample support set",
    )
    parser.add_argument(
        "--max_log_weight",
        type=float,
        default=np.log(2),
        help="Do not touch, used to randomly sample support set",
    )
    parser.add_argument(
        "--ignore_bilevel_ontology",
        action="store_true",
        help="Whether or not to use superclass for BiLevel datasets (e.g Omniglot)",
    )
    parser.add_argument(
        "--ignore_dag_ontology",
        action="store_true",
        help="Whether to ignore ImageNet DAG ontology when sampling \
                              classes from it. This has no effect if ImageNet is not  \
                              part of the benchmark.",
    )
    parser.add_argument(
        "--ignore_hierarchy_probability",
        type=float,
        default=0.0,
        help="if using a hierarchy, this flag makes the sampler \
                              ignore the hierarchy for this proportion of episodes \
                              and instead sample categories uniformly.",
    )

    # CDFSL parameters
    parser.add_argument(
        "--test_n_way", default=5, type=int, help="class num to classify for testing (validation) "
    )
    parser.add_argument(
        "--n_shot",
        default=5,
        type=int,
        help="number of labeled data in each class, same as n_support",
    )
    parser.add_argument(
        "--cdfsl_domains",
        nargs="+",
        default=["EuroSAT", "ISIC", "CropDisease", "ChestX"],
        help="CDFSL datasets",
    )

    # Model params
    # parser.add_argument(
    #     "--arch", default="dino_base_patch16_224", type=str, help="Architecture of the backbone."
    # )
    parser.add_argument("--arch", default="maml", type=str, help="Architecture of the backbone.")
    parser.add_argument("--patch_size", default=16, type=int, help="Patch resolution of the model.")
    parser.add_argument(
        "--pretrained_weights", default="", type=str, help="Path to pretrained weights to evaluate."
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument("--unused_params", action="store_true")
    parser.add_argument("--no-pretrain", action="store_true")

    # MAML params
    parser.add_argument("--step_size", default=0.01, type=float, help="Inner update step_size.")
    parser.add_argument("--num_steps", default=5, type=int, help="Number of inner gradient steps")
    parser.add_argument(
        "--first_order", action="store_true", help="To use first_order or not (second_order)."
    )

    # Deployment params
    parser.add_argument(
        "--deploy",
        type=str,
        default="vanilla",
        help="Which few-shot model to be deployed for meta-testing.",
    )
    parser.add_argument("--num_adapters", default=1, type=int, help="Number of adapter tokens")
    parser.add_argument(
        "--ada_steps", default=40, type=int, help="Number of feature adaptation steps"
    )
    parser.add_argument(
        "--ada_lr", default=5e-2, type=float, help="Learning rate of feature adaptation"
    )
    parser.add_argument(
        "--aug_prob",
        default=0.9,
        type=float,
        help="Probability of applying data augmentation during meta-testing",
    )
    parser.add_argument(
        "--aug_types",
        nargs="+",
        default=["color", "translation"],
        help="color, offset, offset_h, offset_v, translation, cutout",
    )

    # Other model parameters
    parser.add_argument("--img-size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
    )
    parser.add_argument(
        "--drop-path", type=float, default=0.1, metavar="PCT", help="Drop path rate (default: 0.1)"
    )

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.set_defaults(model_ema=False)
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help="")
    parser.add_argument("--model-ema-force-cpu", action="store_true", default=False, help="")

    # Optimizer parameters
    parser.add_argument(
        "--opt", default="adamw", type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw"'
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, metavar="LR", help="learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR (step scheduler)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.0, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")'
    )
    parser.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0.0, help="mixup alpha, mixup enabled if > 0. (default: 0.8)"
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=0.0,
        help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Distillation parameters
    parser.add_argument(
        "--teacher-model",
        default="regnety_160",
        type=str,
        metavar="MODEL",
        help='Name of teacher model to train (default: "regnety_160"',
    )
    parser.add_argument("--teacher-path", type=str, default="")
    parser.add_argument(
        "--distillation-type", default="none", choices=["none", "soft", "hard"], type=str, help=""
    )
    parser.add_argument("--distillation-alpha", default=0.5, type=float, help="")
    parser.add_argument("--distillation-tau", default=1.0, type=float, help="")

    # Misc
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist-eval", action="store_true", default=False, help="Enabling distributed evaluation"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--epoch", type=int, help="epoch number", default=6)
    parser.add_argument("--n_way", type=int, help="n way", default=5)
    parser.add_argument("--k_spt", type=int, help="k shot for support set", default=1)
    parser.add_argument("--k_qry", type=int, help="k shot for query set", default=15)
    parser.add_argument("--imgsz", type=int, help="imgsz", default=84)
    parser.add_argument("--imgc", type=int, help="imgc", default=3)
    parser.add_argument(
        "--task_num",
        type=int,
        help="meta batch size, namely task num",
        default=4,
    )
    parser.add_argument(
        "--meta_lr",
        type=float,
        help="meta-level outer learning rate",
        default=0.01,
    )
    parser.add_argument(
        "--update_lr",
        type=float,
        help="task-level inner update learning rate",
        default=0.01,
    )
    # inner_update_lr only for 2-tier
    parser.add_argument(
        "--inner_update_lr",
        type=float,
        help="task-level inner update learning rate",
        default=0.001,
    )
    parser.add_argument(
        "--update_step",
        type=int,
        help="task-level inner update steps",
        default=5,
    )
    # inner_update_step only for 2-tier
    parser.add_argument(
        "--inner_update_step",
        type=int,
        help="task-level inner update steps",
        default=5,
    )
    parser.add_argument(
        "--episode",
        type=int,
        help="Number of training episodes per epoch",
        default=10000,
    )
    parser.add_argument(
        "--repeat",
        type=int,
        help="number of validations",
        default=5,
    )
    parser.add_argument(
        "--update_step_test",
        type=int,
        help="update steps for finetunning",
        default=10,
    )
    parser.add_argument("--log_dir", type=str, help="log directory for tensorboard", default="log/")
    # Augmentation
    parser.add_argument(
        "--traditional_augmentation",
        "--trad_aug",
        action="store_true",
        help="train with augment data in traditional way",
        default=False,
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="add augmentation and measure weight distance between original data and augmented data",
        default=False,
    )
    parser.add_argument(
        "--qry_aug",
        action="store_true",
        help="use augmented query set when meta-updating parameters",
        default=False,
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="add random horizontal flip augmentation",
        default=False,
    )
    # Regularizer
    parser.add_argument("--reg", type=float, help="coefficient for regularizer", default=0.01)
    parser.add_argument(
        "--rm_augloss",
        action="store_true",
        default=False,
    )
    # proximal regularizer for imaml
    parser.add_argument(
        "--prox_lam",
        type=float,
        help="Lambda for imaml proximal regularizer",
        default=0,
    )
    parser.add_argument(
        "--prox_task",
        type=int,
        help="Apply proximal regularizer at task 0 (original), task 1 (augmented), or 2 (both)",
        default=-1,
    )
    # Chaser loss for bmaml
    parser.add_argument(
        "--chaser_lam",
        type=float,
        help="Lambda for bmaml chaser loss",
        default=0,
    )
    parser.add_argument(
        "--chaser_task",
        type=int,
        help="Apply proximal regularizer at task 0 (original), task 1 (augmented), or 2 (both)",
        default=-1,
    )
    parser.add_argument(
        "--chaser_lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--bmaml",
        action="store_true",
        help="Bmaml loss only",
        default=False,
    )
    # TEMP
    parser.add_argument(
        "--one_tier",
        action="store_true",
    )
    return parser

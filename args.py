from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    # Execution mode
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full'],
        default='train',
        help=("train: performs training and validation; test: tests the model "
              "found in \"--save-dir\" with name \"--name\" on \"--dataset\"; "
              "full: combines train and test modes. Default: train"))
    parser.add_argument(
        "--resume",
        action='store_true',
        help=("The model found in \"--save-dir/--name/\" and filename "
              "\"--name.h5\" is loaded."))
    parser.add_argument(
        "--model",
        choices=['enet', 'm_unet', 'pspnet'],
        default='enet',
        help="Model to use. Default: enet")
    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="The batch size. Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs. Default: 300")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=5e-4,
        help="The learning rate. Default: 5e-4")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.5")
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=100,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 100")
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4")

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=['camvid', 'cityscapes', 'darea'],
        default='camvid',
        help="Dataset to use. Default: camvid")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/CamVid",
        help="Path to the root directory of the selected dataset. "
        "Default: data/CamVid")
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="The image height. Default: 360")
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="The image width. Default: 480")
    parser.add_argument(
        "--weighing",
        choices=['enet', 'mfb', 'none'],
        default='ENet',
        help="The class weighing technique to apply to the dataset. "
        "Default: enet")
    parser.add_argument(
        "--with-unlabeled",
        dest='ignore_unlabeled',
        action='store_false',
        help="The unlabeled class is not ignored.")

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. Default: 4")
    parser.add_argument(
        "--print-step",
        action='store_true',
        help="Print loss every step")
    parser.add_argument(
        "--imshow-batch",
        action='store_true',
        help=("Displays batch images when loading the dataset and making "
              "predictions."))
    parser.add_argument(
        "--device",
        default='cuda',
        help="Device on which the network will be trained. Default: cuda")

    # Storage settings
    parser.add_argument(
        "--name",
        type=str,
        default='ENet',
        help="Name given to the model when saving. Default: ENet")
    parser.add_argument(
        "--save-dir",
        type=str,
        default='save',
        help="The directory where models are saved. Default: save")
    # parser.add_argument(
    #     "--ckpt",
    #     type=str,
    #     default="/media/gnort/HDD/Work_New/PyTorch-ENet/save/ENet_DArea/ENet",
    #     help="Checkpoint path")
    parser.add_argument(
        "--mask-out-dir",
        type=str,
        default='save/test_img/m_unet/',
        help="The directory where mask images are saved. "
        "Default: save/test_img/m_unet")
    parser.add_argument(
        "--save-interval-freq",
        type=int,
        default=0,
        help="The number of intervals between each save. Default: 0 "
        "(no saving during training)")
    parser.add_argument(
        "--save-all-checkpoints",
        action='store_true',
        default=False,
        help="Saves all checkpoints instead of only the last one."
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=10,
        help="The number of epochs between each validation. Default: 10")
    parser.add_argument(
        "--checkpoint-metric-max",
        action='store_true',
        default=True,
        help="The metric to be maximized during checkpoint saving. Default: True")
    return parser.parse_args()

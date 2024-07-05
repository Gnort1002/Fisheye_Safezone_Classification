import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models.enet import ENet
from models.mobile_unet import MobileUNet
from models.pspnet import PSPNet
from engine.train import Train
from metric.iou import IoU
from args import get_arguments
import utils
from data_loader.dataset_loader import load_dataset
# Get the arguments
args = get_arguments()

device = torch.device(args.device)


def main_train(
        train_loader,
        val_loader,
        class_weights,
        class_encoding,
        model_name,
        save_dir):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    # Intialize ENet
    if model_name.lower() == 'enet':
        model = ENet(num_classes).to(device)
    if model_name.lower() == 'm_unet':
        model = MobileUNet(num_classes).to(device)
    if model_name.lower() == 'pspnet':
        model = PSPNet(layers=50, classes=num_classes).to(device)
    # Check if the network architecture is correct
    print(model)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ENet authors used Adam as the optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0

    # Start Training
    print()
    train = Train(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optimizer,
        criterion=criterion, 
        metric=metric,
        device=device)
    train.run()
    return model


if __name__ == '__main__':
    # Fail fast if the dataset directory doesn't exist
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)

    # Import the requested dataset
    if args.dataset.lower() == 'camvid':
        from data import CamVid as dataset
    elif args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    elif args.dataset.lower() == 'darea':
        from data import Drivable_Area as dataset
    else:
        # Should never happen...but just in case it does
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(
            args.dataset))

    loaders, w_class, class_encoding = load_dataset(dataset, args)
    train_loader, val_loader, test_loader = loaders

    model = main_train(
        train_loader,
        val_loader,
        w_class,
        class_encoding,
        args.model,
        args.save_dir)

import torch
import numpy as np
import os
import cv2
from typing import Dict
from utils import (logger)
CHECKPOINT_EXTN = "pt"
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor


def get_model_state_dict(model: torch.nn.Module) -> Dict:
    """Returns `state_dict` of a given model.

    Args:
        model: A torch model (it can be also a wrapped model, e.g., with DDP).

    Returns:
        `state_dict` of the model. If model is an EMA instance,
        the `state_dict` corresponding to EMA parameters is
            returned.
    """
    return model.state_dict()


def get_training_state(
    iterations: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
) -> Dict:
    """Create a checkpoint dictionary that includes all required
        states to resume the training from its current state.

    Args:
        iterations: An integer denoting training iteration number. Each
            iteration corresponds to forward-backward passes on a batch
            with all GPUs.
        epoch: An integer denoting epoch number.
        model: The model being trained.
        optimizer: Optimizer object, which possibly store training
            optimization state variables.
        best_metric: Best observed value of the tracking validation metric.
            For example, best top-1 validation accuracy
            that is observed until the current iteration.
        gradient_scaler: `GradScaler` object storing required automatic mixed 
            precision state.
        model_ema: EMA model to be stored in the checkpoint.

    Returns:
        A dictionary that includes all required states to resume the training 
            from its current state.
    """
    model_state = get_model_state_dict(model)
    training_state = {
        "iterations": iterations,
        "epoch": epoch,
        "state_dict": model_state,
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric,

    }
    return training_state


def save_checkpoint(model, optimizer, epoch, miou, args):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - args (``ArgumentParser``): An instance of ArgumentParser which contains
    the arguments used to train ``model``. The arguments are written to a text
    file in ``args.save_dir`` named "``args.name``_args.txt".

    """
    name = args.name
    save_dir = args.save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)

    # Save arguments
    summary_filename = os.path.join(save_dir, name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(miou))


def load_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        folder_dir: str,
        filename: str):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.

    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.

    """
    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    # breakpoint()
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    return model, optimizer, epoch, miou


def create_overlayed_image(
        image: np.ndarray,
        mask: np.ndarray,
        mask_color=(0, 255, 0),
        alpha=0.5,
        out_image_path=None):
    """Creates an overlayed image from an image and a mask.

    Keyword arguments:
    - image (``numpy.ndarray``): The image.
    - mask (``numpy.ndarray``): The mask.
    - mask_color (``tuple``): The color of the mask.

    Returns:
    The overlayed image.

    """
    # image = cv2.resize(image, mask.shape[:2])
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored[np.where(
        (mask_colored == [255, 255, 255]).all(axis=2)
        )] = mask_color
    # breakpoint()
    mask_colored = cv2.resize(mask_colored, image.shape[:2][::-1])
    masked_overlay = cv2.addWeighted(
        image,
        1 - alpha,
        mask_colored,
        alpha, 0)
    if out_image_path is not None:
        cv2.imwrite(out_image_path, masked_overlay)
    return masked_overlay


def save_interval_checkpoint(
    iterations: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    save_dir: str,
    *args,
    **kwargs
) -> None:
    """Save current iteration training checkpoint.

    Args:
        iterations: An integer denoting training iteration number
        Each iteration corresponds to forward-backward passeson
        a batch with all GPUs.
        epoch: An integer denoting epoch number.
        model: The model being trained.
        optimizer: Optimizer object, which possibly store
        training optimization state variables.
        best_metric: Best observed value of the tracking validation metric.
        For example, best top-1 validation accuracy
            that is observed until the current iteration.
        save_dir: Path to a directory to save checkpoints.
        gradient_scaler: `GradScaler` object storing required automatic 
        mixed precision state.
        model_ema: EMA model to be stored in the checkpoint.
    """
    checkpoint = get_training_state(
        iterations, epoch, model, optimizer, best_metric
    )
    ckpt_str = "{}/training_checkpoint".format(save_dir)
    ckpt_fname = "{}_{}_{}.{}".format(
        ckpt_str, epoch, iterations, CHECKPOINT_EXTN)
    torch.save(checkpoint, ckpt_fname)


def save_checkpoint_2(
    args,
    iterations: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    is_best: bool,
    save_dir: str,
    max_ckpt_metric: bool = False,
    k_best_checkpoints: int = -1,
    save_all_checkpoints: bool = False,
) -> None:
    """Save checkpoints corresponding to the current state of the training.

    Args:
        iterations: An integer denoting training iteration number.
            Each iteration corresponds to forward-backward passes
            on a batch with all GPUs.
        epoch: An integer denoting epoch number.
        model: The model being trained.
        optimizer: Optimizer object, which possibly store training
        optimization state variables.
        best_metric: Best observed value of the tracking validation metric.
        For example, best top-1 validation accuracy
            that is observed until the current iteration.
        is_best: A boolean demonstrating whether the current model obtains the
            best validation metric compared to the previously saved checkpoints
        save_dir: Path to a directory to save checkpoints.
        gradient_scaler: `GradScaler` object storing required automatic mixed
            precision state.
        model_ema: EMA model to be stored in the checkpoint.
        is_ema_best: A boolean demonstrating whether the current EMA model
            obtains the best validation metric compared to the previously
            saved checkpoints.
        ema_best_metric: Best observed value of the tracking validation metric 
            by the EMA model.
        max_ckpt_metric: A boolean demonstrating whether the tracking
            validation metric is higher the better, or lower the better.
        k_best_checkpoints: An integer k determining number of top
            (based on validation metric) checkpoints to keep. If
            `k_best_checkpoints` is smaller than 1,
            only best checkpoint stored.
        save_all_checkpoints: If True, will save model_state checkpoints
            (main model and its EMA) for all epochs.
    """
    checkpoint = get_training_state(
        iterations, epoch, model, optimizer, best_metric
    )
    model_state = checkpoint.get("model_state_dict")
    ckpt_str = "{}/checkpoint".format(save_dir)

    if is_best:
        best_model_fname = "{}_best.{}".format(ckpt_str, CHECKPOINT_EXTN)
        if os.path.isfile(best_model_fname):
            os.remove(best_model_fname)
        torch.save(model_state, best_model_fname)
        logger.log(
            "Best checkpoint with score {:.2f} saved at {}".format(
                best_metric, best_model_fname
            )
        )
        summary_filename = os.path.join(save_dir, 'config_summary.txt')
        with open(summary_filename, 'w') as summary_file:
            sorted_args = sorted(vars(args))
            summary_file.write("ARGUMENTS\n")
            for arg in sorted_args:
                arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
                summary_file.write(arg_str)

            summary_file.write("\nBEST VALIDATION\n")
            summary_file.write("Epoch: {0}\n". format(epoch))
            summary_file.write("Mean IoU: {0}\n". format(best_metric))

    # ckpt_fname = "{}/training_checkpoint_last.{}".format(
    #     save_dir, CHECKPOINT_EXTN)
    # torch.save(checkpoint, ckpt_fname)

    ckpt_fname = "{}_last.{}".format(ckpt_str, CHECKPOINT_EXTN)
    torch.save(model_state, ckpt_fname)

    if save_all_checkpoints:
        ckpt_fname = "{}_epoch{}.{}".format(ckpt_str, epoch, CHECKPOINT_EXTN)
        torch.save(model_state, ckpt_fname)


def save_training_log(
        loss: float,
        metric: float,
        epoch: int,
        output_file,
        debug: bool = True,
        training: bool = True):
    """Saves the training log to a text file.

    Keyword arguments:
    - loss (``float``): The loss value.
    - metric (``float``): The metric value.
    - epoch (``int``): The epoch number.
    - output_file (``file``): The text file where the log is saved.
    - debug (``bool``): If True, the log is also printed to the console.
    - training (``bool``): If True, the log is saved as a training log.

    """
    if training:
        if debug:
            print(">>>> [Epoch: {0:d}] Training".format(epoch))
            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | "
                  "Mean IoU: {2:.4f}".format(epoch, loss, metric))
        output_file.write("Epoch {0:d} Training: Loss: {1:.4f} | "
                          "mIoU: {2:.4f}\n".format(epoch, loss, metric))
    else:
        if debug:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))
            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | "
                  "Mean IoU: {2:.4f}".format(epoch, loss, metric))
            output_file.write("Epoch {0:d} Validation: Loss: {1:.4f} | "
                              "mIoU: {2:.4f}\n".format(epoch, loss, metric))
      

def get_tensor_sizes(data: Union[Dict, Tensor]) -> Union[List[str], List[Tuple[int]]]:
    """Utility function for extracting tensor shapes (for printing purposes only)."""
    if isinstance(data, Dict):
        tensor_sizes = []
        for k, v in data.items():
            size_ = get_tensor_sizes(v)
            if size_:
                tensor_sizes.append(f"{k}: {size_}")
        return tensor_sizes
    elif isinstance(data, Tensor):
        return [*data.shape]
    else:
        return []


def model_info(model, img_size):
    model.eval()
    overall_params = sum([p.numel() for p in model.parameters()])
    print("{:<20} = {:>8.3f} M".format("Overall parameters", overall_params / 1e6))
    try:
        # Compute FLOPs using FVCore
        try:
            input_fvcore = torch.rand(1, 3, *img_size)
        except NotImplementedError:
            logger.warning(
                "Profiling not available, dummy_input_and_label not implemented for this model."
            )
            return

        # compute flops using FVCore
        from fvcore.nn import FlopCountAnalysis, flop_count_table

        flop_analyzer = FlopCountAnalysis(model, input_fvcore)
        logger.log("FVCore Analysis:")
        # print input sizes
        input_sizes = get_tensor_sizes(input_fvcore)
        logger.log("Input sizes: {}".format(input_sizes))
        print(flop_count_table(flop_analyzer))

        logger.warning(
            "\n** Please be cautious when using the results in papers. "
            "Certain operations may or may not be accounted in FLOP computation in FVCore. "
            "Therefore, you want to manually ensure that FLOP computation is correct."
        )
    except:
        logger.ignore_exception_with_warning(
            "Unable to compute FLOPs using FVCore. Please check"
        )


from .utils import (
    save_checkpoint_2,
    save_training_log,
    save_interval_checkpoint
)
# from time import time
import sys
sys.path.append("..")
from utils import (
    logger
    )
import torch
import argparse
import torch.utils.data
import torch.nn as nn
import os


class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(
            self,
            args: argparse.Namespace,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            optim: torch.optim.Optimizer,
            criterion: torch.nn.modules.loss,
            metric,
            device: torch.device,
            start_epoch: int = 0,
            start_iteration: int = 0,
            ):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.start_epoch = start_epoch
        self.train_iterations = start_iteration
        self.save_interval_freq = args.save_interval_freq
        self.save_location = args.save_dir
        self.best_metric = 0.0

    def train_epoch(self, epoch, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()

        # epoch_start_time = time.time()
        # batch_load_start = time.time()
        for step, batch_data in enumerate(self.train_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # batch_load_toc = time.time() - batch_load_start
            # batch_size = get_batch_size(inputs)
            # Forward propagation
            outputs = self.model(inputs)

            # Loss computation
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            # save the checkpoint every N updates
            if (
                self.save_interval_freq > 0
                and (self.train_iterations % self.save_interval_freq) == 0
            ):

                save_interval_checkpoint(
                    iterations=self.train_iterations,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    best_metric=loss.item(),
                    save_dir=self.save_location,
                )
                logger.info(
                    "Checkpoints saved after {} updates at: {}".format(
                        self.train_iterations, self.save_location
                    ),
                    print_line=True,
                )
            self.train_iterations += 1
            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.train_loader), self.metric.value()

    def val_epoch(self, iteration_loss=False):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        self.model.eval()
        epoch_loss = 0.0
        self.metric.reset()

        for step, batch_data in enumerate(self.val_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation
            outputs = self.model(inputs)

            # Loss computation
            loss = self.criterion(outputs, labels)

            with torch.no_grad():
                # Forward propagation
                outputs = self.model(inputs)

                # Loss computation
                loss = self.criterion(outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            # save the checkpoint every N updates

            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.val_loader), self.metric.value()

    def run(self):
        # train_start_time = time.time()
        output_file_path = os.path.join(self.args.save_dir, 'results.txt')
        output_file = open(output_file_path, 'w')
        output_file.write("History:\n")
        output_file.close()
        max_epochs = self.args.epochs
        for epoch in range(self.start_epoch, max_epochs):
            output_file = open(output_file_path, 'a')
            max_checkpoint_metric = self.args.checkpoint_metric_max,
            is_best = False
            train_loss, train_ckpt_metric = self.train_epoch(
                epoch, self.args.print_step)
            save_training_log(
                loss=train_loss,
                metric=train_ckpt_metric[1],
                epoch=epoch + 1,
                output_file=output_file,
                training=True,
                debug=True)
            if (epoch + 1) % self.args.val_interval == 0 or epoch + 1 == self.args.epochs:
                val_loss, val_ckpt_metric = self.val_epoch(
                    self.args.print_step)
                save_training_log(
                    loss=train_loss,
                    metric=train_ckpt_metric[1],
                    epoch=epoch + 1,
                    output_file=output_file,
                    training=False,
                    debug=True)
                        
                if max_checkpoint_metric:
                    is_best = val_ckpt_metric[1] >= self.best_metric
                    self.best_metric = max(
                        val_ckpt_metric[1], self.best_metric)
                else:
                    is_best = val_ckpt_metric[1] <= self.best_metric
                    self.best_metric = min(
                        val_ckpt_metric[1], self.best_metric)
            output_file.close()
            save_checkpoint_2(
                args=self.args,
                iterations=self.train_iterations,
                epoch=epoch+1,
                model=self.model,
                optimizer=self.optim,
                best_metric=self.best_metric,
                is_best=is_best,
                save_dir=self.save_location,
                max_ckpt_metric=max_checkpoint_metric,
                save_all_checkpoints=self.args.save_all_checkpoints,
                )
            logger.info(
                "Checkpoints saved at: {}".format(self.save_location),
                print_line=True,
            )
            

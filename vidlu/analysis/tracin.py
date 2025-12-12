import os
import warnings
from pathlib import Path
from functools import partial
import typing as T
import logging

import torch
import numpy as np
from tqdm import tqdm

from vidlu.training import Trainer, IterState
from vidlu.training.checkpoint_manager import CheckpointManager
from vidlu.data import Dataset
import vidlu.torch_utils as vtu
import vidlu.utils.context as vuc
import vidlu.modules.utils as vmu


def get_grads(trainer: Trainer, trainer_state: dict, batch, training_mode=False):
    """Computes the gradients for each example in a batch during a training step.

    Args:
        trainer (Trainer): The trainer object containing the model and optimizer.
        trainer_state (dict): The state dictionary of the trainer.
        batch: A training batch.

    Returns:
        List[dict]: A list where each entry contains a dictionary mapping parameter IDs
        to their respective gradients for a single example.
    """

    def get_single_grad(x):
        trainer.load_state_dict(trainer_state)

        grad = dict()
        num_backward_calls = 0

        def store_grad():
            nonlocal grad, num_backward_calls
            num_backward_calls += 1
            if num_backward_calls > 1:
                raise RuntimeError(
                    "Multiple backward calls within a training step are not supported")
            grad = {id(p): p.grad.detach().clone() for name, p in trainer.model.named_parameters()}

        with (vuc.preserve_attribute([trainer.optimizer], 'step'),
              vuc.preserve_attribute([trainer.optimizer], 'zero_grad')):
            trainer.optimizer.step = store_grad

            trainer.model.eval()
            if not training_mode:
                trainer.model = vmu.KeepingTrainingMode(trainer.model)
            trainer.training.step(trainer.training, x)
            if not training_mode:
                trainer.model = trainer.model.module
        trainer.optimizer.zero_grad()

        return grad

    return [get_single_grad([x[i:i + 1] for x in batch]) for i in range(len(batch[0]))]


def get_param_update(trainer: Trainer, trainer_state: dict, batch):
    """Computes the parameter updates for each example in a batch during a training step.

    The updates include effects from the optimizer such as learning rate scaling, momentum,
    and normalization.

    Args:
        trainer (Trainer): The trainer object containing the model and optimizer.
        trainer_state (dict): The state dictionary of the trainer.
        batch: A training batch.

    Returns:
        List[dict]: A list where each entry contains a dictionary mapping parameter names
        to their respective updates for a single example.
    """

    def get_single_param_update(x):
        params_update = None

        opt_step = trainer.optimizer.step

        @torch.no_grad()
        def opt_step_wrapper():
            nonlocal params_update

            opt_step()
            params_with = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}

            trainer.load_state_dict(trainer_state)
            for name, p in trainer.model.named_parameters():
                p.grad = torch.zeros_like(p)
            opt_step()
            params_without = {name: p.detach().clone() for name, p in
                              trainer.model.named_parameters()}

            params_update = {name: params_with[name] - p for name, p in params_without.items()}

        with vuc.switch_attribute(trainer.optimizer, 'step', opt_step_wrapper):
            trainer.load_state_dict(trainer_state)

            trainer.model.eval()
            trainer.model = vmu.KeepingTrainingMode(trainer.model)
            trainer.training.step(trainer.training, x)
            trainer.model = trainer.model.module

        return params_update

    return [get_single_param_update([x[i:i + 1] for x in batch]) for i in range(len(batch[0]))]


def save_training_images(training_data, output_dir):
    """Saves images from training data to the specified directory.

    Args:
        training_data: The training dataset containing image data and labels.
        output_dir: Directory to save the images.

    Returns:
        List of relative paths to the saved images.
    """
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    image_paths = []

    for i, (image_tensor, _) in enumerate(training_data):
        image = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        image_filename = f"image_{i}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)
        image_paths.append(os.path.relpath(image_path, start=output_dir))

    return image_paths


import plotly.graph_objects as go


def plot_tracin_scores(scores, training_data, output_path, checkpoint_index):
    """Plots TracIn scores using Plotly and saves the result as an HTML file.

    Args:
        scores: The matrix of TracIn scores.
        training_data: The training dataset for generating hover images.
        output_path: Path to save the HTML file.
        checkpoint_index: Index of the current checkpoint.
    """

    scores_np = scores

    # Create hover text with score and coordinates
    hover_text = []
    for i in range(len(training_data)):
        row = []
        for j in range(len(training_data)):
            row.append(f"Score: {scores_np[i, j]:.2f}<br>Indices: ({i}, {j})")
        hover_text.append(row)

    # Create a Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=scores_np,
        colorscale='RdBu',
        zmid=0
    ))

    fig.update_layout(
        title=f"TracIn at Checkpoint {checkpoint_index}",
        xaxis_title="Training Examples",
        yaxis_title="Training Examples",
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickmode='linear')
    )

    fig.write_html(os.path.join(output_path, f"checkpoint_{checkpoint_index:04d}.html"))


def compute_tracin_scores(trainer: Trainer, checkpoints: T.Iterable[dict],
                          training_data: T.Sequence | None,
                          test_data: T.Sequence | None = None,
                          batch_size: int = 16,
                          output_dir: str | os.PathLike = '~/data/stuff/tracin'):
    """Computes TracIn scores for the given training data.

    Args:
        training_data (T.Sequence | None): A `Dataset` instance or data compatible with `Dataset`.

    Returns:
        torch.Tensor: A matrix containing influence scores between training examples.
    """
    assert test_data is None  # TODO: Extend to handle test data.

    training_data = (training_data if isinstance(training_data, Dataset) else
                     Dataset(training_data)).enumerate()

    make_training_data_loader = partial(
        trainer.get_data_loader, shuffle=False, sampler=None, drop_last=False,
        batch_size=batch_size)

    with vtu.preserve_state(trainer):
        scores = torch.zeros([len(training_data), len(training_data)],
                             device=next(trainer.model.parameters()).device)
        for ic, trainer_state in enumerate(checkpoints):
            trainer.load_state_dict(trainer_state)
            param_id_to_lr = {id(p): param_group['lr']
                              for param_group in trainer.optimizer.param_groups
                              for p in param_group['params']}

            for indices1, batch1 in tqdm(make_training_data_loader(training_data)):
                grads1 = get_grads(trainer, trainer_state, batch1)

                for indices2, batch2 in tqdm(
                        make_training_data_loader(training_data[indices1[0]:])):
                    trainer.load_state_dict(trainer_state)
                    grads2 = get_grads(trainer, trainer_state, batch2)

                    with torch.no_grad():
                        for i1, g1 in zip(indices1.cpu().numpy(), grads1):
                            for i2, g2 in zip(indices2.cpu().numpy(), grads2):
                                if test_data is None and i1 > i2:
                                    continue

                                influence = sum([
                                    g1[param_id].flatten().dot(g2[param_id].flatten()).mul_(lr)  # TODO: check 1.0 instead of lr
                                    for param_id, lr in param_id_to_lr.items()])

                                scores[i1, i2] += influence

                                if test_data is None and i1 != i2:
                                    scores[i2, i1] = scores[i1, i2]

            import matplotlib.pyplot as plt
            figure = plt.figure()
            axes = figure.add_subplot(111)
            vmax = torch.max(scores.abs())
            caxes = axes.matshow(scores.cpu().numpy(), cmap='RdBu', vmin=-vmax, vmax=vmax)
            plt.xticks(list(range(0, scores.shape[0], 5)), rotation=90)
            plt.yticks(list(range(0, scores.shape[0], 5)))
            figure.colorbar(caxes)
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}/{ic:04d}.png')
            plt.close()
            plot_tracin_scores(scores.cpu().numpy(), training_data.data, output_dir, ic)

            # scores *= 0

    return scores


class PeriodicModelRecorder:
    def __init__(self, trainer: Trainer, checkpoints_root: str | os.PathLike, experiment_name: str,
                 eval_epochs: T.Sequence | None = None,
                 eval_iters: T.Sequence | None = None, dir_name='tracin'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if (eval_epochs is None) == (eval_iters is None):
            warnings.warn("Either `eval_epochs` or `eval_iters` must be provided, but not both.")
            self.logger.warning(
                "Invalid evaluation configuration: either eval_epochs or eval_iters must be provided")

        self.trainer = trainer
        self.eval_epochs = eval_epochs
        self.eval_iters = eval_iters

        self.cpman = CheckpointManager(Path(checkpoints_root) / dir_name, experiment_name,
                                       state_loaded=True, n_recent_kept=float('inf'),
                                       start_mode='resume_or_start')
        if len(self.cpman) > 0:
            self.cpman.load()

        checkpoint_event = trainer.training.epoch_completed if self.eval_epochs else trainer.training.iter_completed
        self.training_handler_handle = checkpoint_event.add_handler(self.on_iter_completed)

    def on_iter_completed(self, iter_state: IterState):
        """Callback called at specified training iterations or epochs to save checkpoints."""
        eval_iters, iter = (
            self.eval_iters, iter_state.abs_iteration) if self.eval_iters is not None else (
            self.eval_epochs, iter_state.epoch)
        if iter in eval_iters:
            self.logger.info(
                f"Saving checkpoint at {'iteration' if self.eval_iters else 'epoch'} {iter}")
            self.cpman.save(self.trainer.model.state_dict(), index=iter)

    def __del__(self):
        self.training_handler_handle.remove()


class TracIn:
    """Implements the TracIn influence function for computing the impact of training examples on model updates.

    Args:
        trainer (Trainer): The trainer object managing model training.
        experiment_dir (os.PathLike): Directory for storing experiment checkpoints.
        eval_epochs (Optional[Sequence]): Epochs at which to evaluate model influence.
        eval_iters (Optional[Sequence]): Iterations at which to evaluate model influence.
        dir (Path): Directory to store TracIn data.
        cpman (CheckpointManager): A `CheckpointManager` instance.
    """

    def __init__(self, trainer: Trainer, checkpoints_root: str | os.PathLike, experiment_name: str,
                 training_data: Dataset, test_data: Dataset = None,
                 eval_epochs: T.Sequence | None = None,
                 eval_iters: T.Sequence | None = None, dir_name='tracin'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if (eval_epochs is None) == (eval_iters is None):
            warnings.warn("Either `eval_epochs` or `eval_iters` must be provided, but not both.")
            self.logger.warning(
                "Invalid evaluation configuration: either eval_epochs or eval_iters must be provided")

        self.trainer = trainer
        self.eval_epochs = eval_epochs
        self.eval_iters = eval_iters
        self.training_data = training_data
        self.test_data = test_data

        self.cpman = CheckpointManager(Path(checkpoints_root) / dir_name, experiment_name,
                                       n_recent_kept=float('inf'), start_mode='resume_or_start',
                                       separately_saved_state_parts=('model',))
        if len(self.cpman) > 0:
            self.cpman.load()

        checkpoint_event = trainer.training.epoch_completed if self.eval_epochs else trainer.training.iter_completed
        self.training_handler_handle = checkpoint_event.add_handler(self.on_iter_completed)
        self.training_completed_handler_handle = trainer.training.completed.add_handler(
            self.on_training_completed)

    def on_iter_completed(self, iter_state: IterState):
        """Callback called at specified training iterations or epochs to save checkpoints."""
        eval_iters, iter = (
            self.eval_iters, iter_state.abs_iteration) if self.eval_iters is not None else (
            self.eval_epochs, iter_state.epoch)
        if iter in eval_iters:
            self.logger.info(
                f"Saving checkpoint at {'iteration' if self.eval_iters else 'epoch'} {iter}")
            self.cpman.save(self.trainer.state_dict(), index=iter)

    def on_training_completed(self, iter_state: IterState):
        compute_tracin_scores(trainer=self.trainer,
                              checkpoints=(x[0] for x in self.cpman.load_all()),
                              training_data=self.training_data, test_data=self.test_data)

    def __del__(self):
        self.training_handler_handle.remove()
        self.training_completed_handler_handle.remove()

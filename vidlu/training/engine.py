import logging
import time
from collections import defaultdict
import warnings

from ignite._utils import _to_hours_mins_secs

from vidlu.utils.misc import Event


class State(object):
    """An object that is used to pass internal and user-defined state between event handlers"""

    def __init__(self, **kwargs):
        self.reset(**kwargs)

    def reset(self, **kwargs):
        self.iteration = 0
        self.epoch = 0
        self.output = None
        self.batch = None
        self.batch_count = None
        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Engine(object):
    # Copied from Ignite and modified
    """Runs a given process_function over each batch of a dataset, emitting events as it goes.

    Args:
        process_function (Callable): A function receiving a handle to the engine and the current batch
            in each iteration, and returns data to be stored in the engine's state

    Example usage:

    .. code-block:: python

        def train_and_store_loss(engine, batch):
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            return loss.item()

        engine = Engine(train_and_store_loss)
        engine.run(data_loader)

        # Loss value is now stored in `engine.state.output`.

    """

    def __init__(self, process_function):
        self._event_handlers = defaultdict(list)
        self._logger = logging.getLogger(__name__ + "." + type(self).__name__)
        self._logger.addHandler(logging.NullHandler())
        self._process_function = process_function
        self.should_terminate = False
        self.should_terminate_single_epoch = False
        self.state = State()

        # events
        self.started = Event()
        self.completed = Event()
        self.epoch_started = Event()
        self.epoch_completed = Event()
        self.iteration_started = Event()
        self.iteration_completed = Event()

        if self._process_function is None:
            raise ValueError("Engine must be given a processing function in order to run")

    def terminate(self):
        """Sends terminate signal to the engine, so that it terminates completely the run after the
        current iteration
        """
        self._logger.info(
            "Terminate signaled. Engine will stop after current iteration is finished")
        self.should_terminate = True

    def terminate_epoch(self):
        """Sends terminate signal to the engine, so that it terminates the current epoch after the
        current iteration
        """
        self._logger.info("Terminate current epoch is signaled. "
                          "Current epoch iteration will stop after current iteration is finished")
        self.should_terminate_single_epoch = True

    def _run_once_on_dataset(self):
        start_time = time.time()

        for batch in self.state.dataloader:
            self.state.iteration += 1
            self.state.batch = batch
            self.iteration_started(self.state)
            self.state.output = self._process_function(self, batch)
            self.iteration_completed(self.state)
            if self.should_terminate or self.should_terminate_single_epoch:
                self.should_terminate_single_epoch = False
                break
        time_taken = time.time() - start_time
        return time_taken

    def run(self, data, max_epochs=1, restart=True):
        """Runs the process_function over the passed data.

        Args:
            data (Iterable): Collection of batches allowing repeated iteration (e.g., list or `DataLoader`)
            max_epochs (int, optional): max epochs to run for (default: 1)

        Returns:
            State: output state
        """

        if restart or self.state is None:
            self.state.reset(metrics={})
        self.state.update(dataloader=data, max_epochs=max_epochs, batch_count=len(data))

        self._logger.info("Engine run starting with max_epochs={}".format(max_epochs))
        start_time = time.time()
        self.started(self.state)
        if self.state.epoch >= max_epochs:
            warnings.warn("All epochs are already completed.")
        while self.state.epoch < max_epochs and not self.should_terminate:
            self.state.epoch += 1
            self.epoch_started(self.state)
            time_taken = self._run_once_on_dataset()
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self._logger.info(
                f"Epoch {self.state.epoch} completed after {hours:02}:{mins:02}:{secs:02}")
            if self.should_terminate:
                break
            self.epoch_completed(self.state)

        self.completed(self.state)
        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)
        self._logger.info(f"Engine run completed after {hours:02}:{mins:02}:{secs:02}")

        return self.state

    def state_dict(self):
        return dict(epoch=self.state.epoch)

    def load_state_dict(self, state_dict):
        self.state.epoch = state_dict['epoch']

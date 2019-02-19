import logging
import signal

import torch

from vidlu.utils.misc import try_input

class TerminateOnKey:

    def __init__(self):
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.StreamHandler())

    def __call__(self, engine):
        self._logger.warning("")
        engine.terminate()

import dataclasses as dc
import os

import vidlu.experiments as ve
import vidlu.modules as vm


@dc.dataclass
class ExperimentExtension:
    experiment: ve.TrainingExperiment

    def __post_init__(self):
        self.__enter__()

    def attach(self):
        raise NotImplementedError()

    def detach(self):
        for h in self._handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach()

    def __del__(self):
        self.detach()


@dc.dataclass
class ActivationStorer(ExperimentExtension):
    dir: os.PathLike

    def attach(self):
        # TODO: collect intermediate outputs
        model_wiot = vm.with_intermediate_outputs(self.experiment.model)
        raise NotImplementedError()

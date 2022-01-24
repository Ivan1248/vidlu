import torch
import hashlib
import warnings

from . import elements as E
from vidlu.utils.collections import FSDict


def tensor_hash(x):
    warnings.warn("tensor_hash does not produce a hash persistent over sessions.")
    return hash((tuple(x.view(-1).cpu().numpy()), x.device, x.shape))


class FileCachedModule(E.Module):
    def __init__(self, module, path):
        super().__init__()
        self.module = module
        self.cache = FSDict(path)

    def forward(self, inputs):
        if self.training:
            raise RuntimeError("The cached module has to be in eval mode.")
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor.")
        hashes = list(map(str, map(tensor_hash, inputs)))
        if any(h not in self.cache for h in hashes):
            outputs = self.module(inputs)
            self.cache.update(dict(zip(hashes, outputs)))
        else:
            outputs = torch.stack([self.cache[h] for h in hashes])
        return outputs

    def __del__(self):
        self.cache.delete()

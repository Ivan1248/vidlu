import os
from pathlib import Path
import pickle
from functools import partial

import torch
import numpy as np
import json


class LoadSave:
    @staticmethod
    def save(obj, file):
        raise NotImplemented

    @staticmethod
    def load(file):
        raise NotImplemented


class BinaryLoadSave(LoadSave):
    read_mode = "rb"
    write_mode = "wb"


class BaseTextLoadSave(LoadSave):
    read_mode = "r"
    write_mode = "w"


class TorchLoadSave(BinaryLoadSave):
    save = torch.save
    load = partial(torch.load, weights_only=False)


class NumpyLoadSave(BinaryLoadSave):
    save = np.save
    load = np.load


def pickle_load(path):
    with path.with_suffix('.pkl').open('rb') as f:
        return pickle.load(f)


def pickle_save(obj, path):
    with path.with_suffix('.pkl').open('wb') as f:
        return pickle.dump(obj, f)


class PickleLoadSave:
    save = pickle_save
    load = pickle_load


class JsonLoadSave(BaseTextLoadSave):
    save = json.dump
    load = json.load


def text_load(path):
    return Path(path).read_text() if isinstance(path, os.PathLike) else path.read()


def text_save(obj, path):
    return (Path(path).write_text if isinstance(path, os.PathLike) else path.write)(obj)


class TextLoadSave(BaseTextLoadSave):
    save = text_save
    load = text_load

# class ValueFile:
#     def __init__(self, path, loadsave, atomic=False):
#         self.path = Path(path)
#         self.loadsave = loadsave
#         assert not atomic
#
#     def load(self):
#         return self.loadsave.load(self.path)
#
#     def save(self, obj):
#         return self.loadsave.save(obj, self.path)

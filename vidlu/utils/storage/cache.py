import os
from pathlib import Path
import pickle
import shutil

from vidlu.utils.storage.compressors import DefaultCompressor


class PickleFileAccessor:
    def save(self, cache_path, obj):
        with open(cache_path, 'wb') as cache_file:
            return pickle.dump(obj, cache_file, protocol=4)

    def load(self, cache_path):
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)


class CompressingFileAccessor:
    __slots__ = 'compressor', 'file_accessor'

    def __init__(self, file_accessor_f=PickleFileAccessor, compressor_f=DefaultCompressor):
        self.compressor = compressor_f()
        self.file_accessor = file_accessor_f()

    def load(self, path):
        cobj = self.file_accessor.load(path)
        return self.compressor.decompress(cobj)

    def save(self, path, obj):
        cobj = self.compressor.compress(obj)
        self.file_accessor.save(path, cobj)


class HDDCache:
    def __init__(self, dir, name=None, persistent=False, compressor_f=DefaultCompressor,
                 file_accessor_f=PickleFileAccessor):
        self.dir = Path(dir) if name is None else Path(dir)
        self.persistent = persistent
        self.compressor = compressor_f()
        self.file_accessor = file_accessor_f()
        os.makedirs(self.dir, exist_ok=True)

    def __getitem__(self, key: str):
        cobj = self.file_accessor.load(self._get_path(key))
        return self.compressor.decompress(cobj)

    def __setitem__(self, key: str, obj):
        cobj = self.compressor.compress(obj)
        self.file_accessor.save(self._get_path(key), cobj)

    def __delitem__(self, key: str):
        self._get_path(key).unlink()

    def __del__(self):
        if not self.persistent and self.dir.exists():
            self.delete()

    def clear(self):
        for x in self.cache_dir.iterdir():
            if x.is_dir():
                shutil.rmtree(x)
            else:
                x.unlink()

    def delete(self):
        shutil.rmtree(self.dir)

    def _get_path(self, key: str):
        return self.dir / key

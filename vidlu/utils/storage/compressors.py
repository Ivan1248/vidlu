import PIL.Image as pimg

import blosc
import numpy as np
import torch

from vidlu.transforms.image.format import numpy_to_pil


class CompressedObject:
    def __init__(self, compressor, data):
        self.data = compressor.compress(data)
        self._decompress = compressor.decompress

    def decompress(self):
        return self._decompress(self.data)


class Compressor:
    def compress(self, obj):
        raise NotImplementedError

    def decompress(self, obj):
        raise NotImplementedError


class NonCompressor:
    def compress(self, obj):
        return obj

    def decompress(self, obj):
        return obj


class NumpyCompressor(Compressor):
    def compress(self, arr):
        # cname='zstd'
        c = blosc.compress_ptr(arr.__array_interface__['data'][0], arr.size, arr.dtype.itemsize,
                               clevel=3, cname='lz4', shuffle=blosc.SHUFFLE)
        return c, arr.shape, arr.dtype

    def decompress(self, obj):
        c, shape, dtype = obj
        arr = np.empty(shape, dtype)
        blosc.decompress_ptr(c, arr.__array_interface__['data'][0])
        return arr


class TorchCompressor(Compressor):
    def __init__(self, numpy_compressor_f=NumpyCompressor):
        self.numpy_compressor = numpy_compressor_f()

    def compress(self, obj):
        return self.numpy_compressor.compress(obj.numpy())

    def decompress(self, obj):
        return torch.from_numpy(obj)


class PILCompressor(Compressor):
    def __init__(self, numpy_compressor_f=NumpyCompressor):
        self.numpy_compressor = numpy_compressor_f()

    def compress(self, obj):
        mode = obj.mode
        arr = np.array(obj)
        carr = self.numpy_compressor.compress(arr)
        return carr, mode

    def decompress(self, obj):
        carr, mode = obj
        arr = self.numpy_compressor.decompress(carr)
        return numpy_to_pil(arr, mode=mode)


class DefaultCompressor(Compressor):
    # TODO: torchvision.io.encode_png

    def compress(self, obj):
        compressor = (NumpyCompressor() if isinstance(obj, np.ndarray) else
                      PILCompressor() if isinstance(obj, pimg.Image) else
                      NonCompressor())
        return CompressedObject(compressor, obj)

    def decompress(self, obj):
        return obj.decompress()

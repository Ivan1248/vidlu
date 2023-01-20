import PIL.Image as pimg

import blosc
import numpy as np
import torch
import torchvision.io

from vidlu.transforms.image.format import numpy_to_pil, pil_to_torch, torch_to_pil


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
                               clevel=9, cname='lz4hc', shuffle=blosc.SHUFFLE)
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


class PILPNGCompressor(Compressor):  # TODO
    def compress(self, obj):
        return torchvision.io.encode_png(pil_to_torch(obj).permute(2, 0, 1),
                                         compression_level=9), obj.mode

    def decompress(self, obj):
        # compr. level: size, compr. time, decompr. time, decompr. CPU util.
        # torchvision.io.encode_png
        # 0: 780MiB, 1:49, 16s, 215cpu
        # 1: 369MiB, 1:57, 19s, 206cpu
        # 3: 354MiB, 2:10, 19s, 207cpu
        # 6: 329MiB, 3:28, 18s, 207cpu
        # 9: 322MiB, 11:11, 18s, 208cpu
        # blosc.compress_ptr lz4
        # 0: 1018MiB, 1:42, 13s, 189cpu
        # 3: 697MiB, 1:41, 13s, 188cpu
        # 9: 697MiB, 1:42, 13s, 189cpu
        # blosc.compress_ptr lz4hc
        # 3: 571MiB, 1:46, 13s, 189cpu
        # 9: 550MiB, 1:54, 13s, 191cpu
        content, mode = obj
        return numpy_to_pil(torchvision.io.decode_png(content).permute(1, 2, 0).numpy(), mode=mode)


class DefaultCompressor(Compressor):
    def compress(self, obj):
        compressor = (NumpyCompressor() if isinstance(obj, np.ndarray) else
                      PILCompressor() if isinstance(obj, pimg.Image) else
                      NonCompressor())
        return CompressedObject(compressor, obj)

    def decompress(self, obj):
        return obj.decompress()

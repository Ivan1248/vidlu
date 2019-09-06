import torch

import torch.nn.functional as F



def channel2space(x, s):
    N, C, H, W = x.size()
    x = x.permute(0, 3, 2, 1)  # N, W, H, C
    x = x.reshape((N, W, H * s, int(C / s)))  # N, W, H*scale, C/scale
    x = x.permute(0, 2, 1, 3)  # N, H*scale, W, C/scale
    x = x.reshape((N, H * s, W * s, int(C / s ** 2)))  # N, H*scale, W*scale, C/(scale**2)
    x = x.permute(0, 3, 1, 2)  # N, C/(scale**2), H*scale, W*scale
    return x

def space2channel(x, s):
    N, C, H, W = x.size()  # N, C/(scale**2), H*scale, W*scale
    x = x.permute(0, 2, 3, 1)  # N, C/(scale**2), H*scale, W*scale
    x = x.reshape((N, H // s, W //s, C * s ** 2))  # N, H*scale, W*scale, C/(scale**2)



def space2channel_kernel(w):
    C_, C, H, W = w.size()


def conv1x1channel2space(x, s, w):
    x = F.conv2d(x, w)
    return channel2space(x)

def transposed2x2s(x, s, w):
    return F.conv_transpose2d(x, space2channel(w), stride=s)
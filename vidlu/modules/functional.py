from torch.nn import functional as F


def dimensional_function(f_list, *args, **kwargs):
    return f_list[len(args[0].size()) - 3](*args, **kwargs)


def adaptive_avg_pool(x, output_size):
    return dimensional_function(
        [F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d], x, output_size)


def avg_pool(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return dimensional_function(
        [F.avg_pool1d, F.avg_pool2d, F.avg_pool3d], x, kernel_size=kernel_size, stride=stride,
        padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)


def global_avg_pool(x):
    return adaptive_avg_pool(x, 1).squeeze()

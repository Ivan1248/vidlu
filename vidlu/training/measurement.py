import functools
import os
import re
from unittest.mock import MagicMock
import types

import torch

from vidlu.utils.code_modification import transform_func_code


def get_memory_allocated(empty_cache: bool, reset_peak_stats: bool):
    if empty_cache:
        torch.cuda.empty_cache()
    mem_curr = torch.cuda.memory_allocated() / 1024 ** 2
    mem_max = torch.cuda.max_memory_allocated() / 1024 ** 2
    if reset_peak_stats:
        torch.cuda.reset_peak_memory_stats()
    if empty_cache:
        torch.cuda.empty_cache()
    return mem_curr, mem_max


class MemoryTracker:
    def __init__(self, indices=None):
        self.times = []
        self.mems = []
        self.curr_index = -1
        self.indices = indices
        self.labels = []
        self.finished = False

    def add_measurement(self, label):
        self.curr_index += 1
        t = self.curr_index if self.indices is not None else self.curr_index
        t = label[0]

        mem_curr, mem_max = [m / 1024 ** 2 for m in
                             get_memory_allocated(empty_cache=True, reset_peak_stats=True)]

        self.times.append(t - 1)
        self.mems.append(self.mems[-1] if len(self.mems) > 0 else mem_curr)

        self.times.append(t - 0.5)
        self.mems.append(mem_max)

        self.times.append(t)
        self.mems.append(mem_curr)

        label = ': '.join(list(map(str, label)))
        if len(label) > 100:
            label = label[:97] + '...'
        self.labels.extend(['', label, ''])

    def show_plot(self):
        import matplotlib.pyplot as plt

        print(f"time={repr(self.times)}")
        print(f"mem={repr(self.mems)}")

        times, mems, labels = zip(
            *[(t, m, l)
              for i, (t, m, l) in enumerate(zip(self.times, self.mems, self.labels))
              if len(set(self.mems[i // 3 * 3:i // 3 * 3 + 3])) > 1])
        times = [times[0] - 1, *times, times[-1] + 1]
        mems = [mems[0], *mems, mems[-1]]
        labels = ['', *labels, '']

        w, h = 8, len(self.labels) // 10
        fig, ax = plt.subplots(figsize=(w, h))

        ax.fill_betweenx(times, mems, color='tab:blue', alpha=0.3)

        plt.ylim(times[0], times[-1])
        plt.yticks(times)
        ax.set_yticklabels(labels, fontsize=8)
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")
        ax.tick_params(axis="y", direction="in", pad=-4)
        ax.set_axisbelow(True)

        lastm = -1
        for i, (t, m) in enumerate(zip(times, mems)):
            if m != lastm:
                ax.annotate(f"{m:.0f}", xy=(m + 1, t),
                            bbox=dict(facecolor='white', edgecolor='None', alpha=0.7, pad=-0))
                lastm = m

        ax.set_xlim(left=0)
        ax.set_xlabel(r"memory/MiB", rotation=0)

        ax.invert_yaxis()

        plt.tight_layout()
        plt.show()


def mem_tracking_plot_template(_):
    _mem_tracker = MemoryTracker()
    yield
    _mem_tracker.add_measurement(_)
    yield
    _mem_tracker.show_plot()


def _substitute_del_comments(source_code):
    free_comment_pattern = re.compile(r'# !del +((?:[\w\d]+ *, *)*[\w\d]+)')
    matches = free_comment_pattern.findall(source_code)
    for match in matches:
        variables = match.split(', ')
        replacement = 'del ' + ', '.join(f'{var}' for var in variables)
        replacement += '; ' + ' = '.join(f'{var}' for var in variables) + ' = MagicMock()'
        source_code = source_code.replace(f'# !del {match}', f'{replacement}')
        # source_code = source_code.replace(f'# !del {match}', f'del {replacement}')
    return source_code


def memory_tracking(substitute_del_comments=False,
                    cond=lambda s, *a, **k: bool(int(os.environ.get('VIDLU_MEM_TRACKING', '0')))):
    def memory_tracking_wrapper(func: types.FunctionType):
        transformed_func = transform_func_code(
            func, mem_tracking_plot_template,
            preprocess=_substitute_del_comments if substitute_del_comments else None,
            additional_namespace=dict(MagicMock=MagicMock, MemoryTracker=MemoryTracker))

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if cond(self, *args, **kwargs):
                return transformed_func(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return memory_tracking_wrapper

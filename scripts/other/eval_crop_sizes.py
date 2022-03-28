import _context

from functools import partial

import vidlu.transforms.image as vti
from vidlu.experiments import TrainingExperiment


def run(e: TrainingExperiment, cropping_size=None, padding_sizes=None, padding_alignment='center',
        metric='mIoU'):
    if padding_sizes is None:
        padding_sizes = [cropping_size]
    results = []
    for ps in padding_sizes:
        if cropping_size is None:
            center_crop = crop_size = cropping_size
        else:
            crop_size = cropping_size if cropping_size[0] < ps[0] else ps
            center_crop = partial(vti.center_crop, shape=crop_size)
        print(
            f"Evaluating {crop_size or 'original'} crop padded to {ps or 'original'} with alignment {repr(padding_alignment)}.")
        data_val = e.data.test
        if center_crop is not None:
            data_val = e.data.test.map_fields(dict(image=center_crop, seg_map=center_crop))
        if ps is not None:
            pad_image = partial(vti.pad_to_shape, shape=ps, value='mean',
                                alignment=padding_alignment)
            pad_label = partial(pad_image, value=-1)
            data_val = data_val.map_fields(dict(image=pad_image, seg_map=pad_label))
        result = e.trainer.eval(data_val)
        results.append(result.metrics[metric])
    print(results)

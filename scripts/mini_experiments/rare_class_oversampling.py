import sys
from functools import partial
from pathlib import Path

from IPython.core import ultratb

from vidlu.data import Dataset, Record, datasets
from vidlu.factories import prepare_dataset
from vidlu.data.utils import class_incidence
from vidlu.transforms import jitter
import vidlu.transforms.image as vti
from vidlu.utils.presentation.visualization import show_batch, view_predictions


def set_traceback_format(call_pdb=False, verbose=False):
    sys.excepthook = ultratb.FormattedTB(mode='Verbose' if verbose else 'Plain',
                                         color_scheme='Linux', call_pdb=call_pdb)


def add_class_segment_boxes(ds):
    ds = ds.info_cache_hdd(dict(seg_class_info=class_incidence.seg_class_info), CACHE_DIR,
                           recompute=False, simplify_dataset=lambda ds: ds[:4])
    return ds.enumerate().map_unpack(
        lambda i, r: Record(r,
                            class_seg_boxes=ds.info.seg_class_info['class_segment_boxes'][i]))


set_traceback_format(call_pdb=True, verbose=False)

CACHE_DIR = Path.home() / 'data/cache'
DATASETS_DIR = Path.home() / 'data/datasets'

RARE_CLASSES = [16]

my_dataset = datasets.Cityscapes(DATASETS_DIR / 'Cityscapes', 'train')
ds = Dataset(data=my_dataset, name='my_dataset', info=dict(class_count=25)).map(
    lambda r: Record(image=r.image, seg_map=r.seg_map))

ds = prepare_dataset(ds)

ds = add_class_segment_boxes(ds)

random_scale_crop_f = partial(vti.RandomScaleCrop,
                              rand_crop_fn=partial(jitter.random_crop_overlapping,
                                                   rare_classes=RARE_CLASSES))
from tqdm import tqdm

ds_jittered = ds.permute(seed=1)[:100] \
    .filter(lambda r: any(r.seg_map.eq(c).sum() > 0 for c in RARE_CLASSES),
            progress_bar=partial(tqdm, desc='Filtering images with rare classes for visualization')) \
    .map(jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                         scale_dist="log-uniform",
                                         random_scale_crop_f=random_scale_crop_f))

view_predictions(
    ds_jittered[:, :2].map(lambda r: (r.image.permute(1, 2, 0).numpy(), r.seg_map.numpy())),
    infer=None)

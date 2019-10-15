from pathlib import Path
from torchvision.transforms import Compose
import numpy as np
from tqdm import tqdm

from _context import vidlu, dirs
from vidlu.experiments import get_prepared_data_for_trainer

from vidlu.libs.swiftnet.data import Cityscapes as IKCityscapes

from vidlu.libs.swiftnet.data.transform import *
from vidlu.libs.swiftnet.data.mux.transform import *

data_path = Path("/home/igrubisic/data/datasets/Cityscapes")

# mydata = Cityscapes(data_path, subset="val")
mydata = get_prepared_data_for_trainer("cityscapes{train,val}", dirs.DATASETS, dirs.CACHE,
                                       "standardize").test
ts = (2048, 1024)
ikdata = IKCityscapes(
    data_path,
    subset="val",
    transforms=Compose([Open(),
                        RemapLabels(IKCityscapes.map_to_id, IKCityscapes.num_classes),
                        Pyramid(alphas=[1.]),
                        SetTargetSize(target_size=ts, target_size_feats=(ts[0] // 4, ts[1] // 4)),
                        Normalize(scale=255, mean=IKCityscapes.mean, std=IKCityscapes.std),
                        # Tensor(),
                        ]))
breakpoint()

for my, ik in tqdm(zip(mydata, ikdata), total=len(mydata)):
    myx = np.array(my.x)
    myy = np.array(my.y, np.int64)
    ikx = np.array(ik['image']).transpose(2, 0, 1)
    iky = np.array(ik['labels'], dtype=np.int64)
    iky[iky == 19] = -1
    assert np.all(myx == ikx) and np.all(myy == iky)

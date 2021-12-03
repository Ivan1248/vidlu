import numpy as np
import torch

import vidlu.transforms.image as vti
import vidlu.data.types as dt


def prepare_element(x, key_or_type):
    dom = dt.from_key.get(key_or_type, None) if isinstance(key_or_type, str) else key_or_type
    if dom is None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x
    if dom is dt.Image:
        x = vti.hwc_to_chw(vti.to_torch(x)).to(dtype=torch.float) / 255
    elif dom in (dt.ClassLabel, dt.SegMap):
        x = torch.tensor(x, dtype=torch.int64)  # the loss supports only int64
    return dom(x)

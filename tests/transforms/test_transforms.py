import pytest

import numpy as np
import torch
import PIL.Image as pimg

from vidlu.transforms import image_transformer


class TestData:
    def test_transforms(self):
        im = np.random.randint(0, 255, (30, 20, 3), dtype=np.uint8)
        imtn = image_transformer(im)
        imtp = imtn.to_pil()
        imtn_f = imtn.transform(lambda x: x.astype(np.float32))
        imtt = imtn.to_torch()
        imtt.to_numpy()
        assert imtt.to_pil().item == imtp.item
        assert np.all(imtt.to_numpy().item == imtn.item)
        assert np.all(imtn.item == imtp.to_torch().to_numpy().item)

        assert imtt.hwc_to_chw().item.shape != imtt.item.shape
        assert torch.all(imtt.hwc_to_chw().chw_to_hwc().item == imtt.item)

        imtt_f = imtt.to_float32()
        mean, std = (127,) * 3, (50,) * 3
        imtt_f_st = imtt_f.standardize(mean, std)
        imtt_f_st_dst = imtt_f_st.destandardize(mean, std)
        assert torch.max(torch.abs(imtt_f.item - imtt_f_st_dst.item)) < 1e-5
        assert torch.all(imtt.item == imtt_f.transform(lambda x: x+0.5).to_uint8().item)

        imtp_cc = imtp.center_crop((40, 10))
        imtn_cc = imtn.center_crop((40, 10))
        assert np.all(imtp_cc.to_numpy().item == imtn_cc.item)
        assert imtp_cc.item == imtn_cc.to_pil().item

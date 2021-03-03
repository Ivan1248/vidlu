import numpy as np
import torch
import torchvision.transforms.functional as F

from vidlu.transforms import image, numpy


class TestData:
    def test_transforms(self):
        imtn = np.random.randint(0, 255, (30, 20, 3), dtype=np.uint8)
        imtp = image.to_pil(imtn)
        imtt = image.to_torch(imtn)
        assert image.to_pil(imtt) == imtp
        assert np.all(image.to_numpy(imtt) == imtn)
        assert np.all(imtn == image.to_numpy(image.to_torch(imtp)))
        imtt_chw = image.hwc_to_chw(imtt)

        assert image.hwc_to_chw(imtt).shape != imtt.shape
        assert torch.all(image.chw_to_hwc(imtt_chw) == imtt)

        imtt_chw_f = imtt_chw.to(torch.float32)
        mean, std = torch.tensor((127,) * 3), torch.tensor((50,) * 3)
        imtt_chw_f_st = image.standardize(imtt_chw_f, mean, std)
        assert torch.all(imtt_chw_f_st == image.Standardize(mean, std)(imtt_chw_f))
        imtt_chw_f_st_dst = image.destandardize(imtt_chw_f_st, mean, std)
        assert torch.max(torch.abs(imtt_chw_f - imtt_chw_f_st_dst)) < 1e-5
        assert torch.all(imtt_chw == (imtt_chw_f + 0.5).to(torch.uint8))

        imtp_cc = F.center_crop(imtp, (10, 5))
        imtn_cc = numpy.center_crop(imtn, (10, 5))
        assert np.all(image.to_numpy(imtp_cc).shape == imtn_cc.shape)
        assert np.all(image.to_numpy(imtp_cc) == imtn_cc)
        assert imtp_cc == image.to_pil(imtn_cc)

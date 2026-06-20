import numpy as np
import pytest
import torch

from irap_data.image_utils import center_crop, hwc_to_chw_float_tensor, resize_to_cover


def _make_hwc(h, w, c=3, dtype=np.uint8):
    return np.arange(h * w * c, dtype=dtype).reshape(h, w, c)


# --- center_crop ---

def test_center_crop_exact_size():
    img = _make_hwc(288, 384)
    out = center_crop(img, (384, 288))
    assert out.shape == (288, 384, 3)
    np.testing.assert_array_equal(out, img)


def test_center_crop_larger_source():
    img = _make_hwc(300, 400)
    out = center_crop(img, (384, 288))
    assert out.shape == (288, 384, 3)
    # center pixel should be preserved
    out_cy, out_cx = 288 // 2, 384 // 2
    src_cy, src_cx = (300 - 288) // 2 + out_cy, (400 - 384) // 2 + out_cx
    np.testing.assert_array_equal(out[out_cy, out_cx], img[src_cy, src_cx])


def test_center_crop_error_on_smaller_source_width():
    img = _make_hwc(288, 300)
    with pytest.raises(ValueError, match="smaller than target"):
        center_crop(img, (384, 288))


def test_center_crop_error_on_smaller_source_height():
    img = _make_hwc(200, 384)
    with pytest.raises(ValueError, match="smaller than target"):
        center_crop(img, (384, 288))


# --- hwc_to_chw_float_tensor ---

def test_hwc_to_chw_float_tensor_shape():
    img = _make_hwc(288, 384)
    t = hwc_to_chw_float_tensor(img)
    assert t.shape == (3, 288, 384)


def test_hwc_to_chw_float_tensor_dtype_range():
    img = np.array([[[0, 128, 255]]], dtype=np.uint8)
    t = hwc_to_chw_float_tensor(img)
    assert t.dtype == torch.float32
    assert float(t[0, 0, 0]) == pytest.approx(0.0)
    assert float(t[2, 0, 0]) == pytest.approx(1.0)
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


def test_hwc_to_chw_float_tensor_layout():
    img = _make_hwc(4, 5)
    t = hwc_to_chw_float_tensor(img)
    # channel 0 of tensor should match [:, :, 0] of source
    np.testing.assert_allclose(
        t[0].numpy(), img[:, :, 0].astype(np.float32) / 255.0
    )


# --- resize_to_cover ---

def test_resize_to_cover_already_large_enough():
    img = _make_hwc(288, 384)
    out = resize_to_cover(img, (384, 288))
    assert out.shape == (288, 384, 3)
    # no resize → same pixels
    np.testing.assert_array_equal(out, img)


def test_resize_to_cover_small_source():
    img = _make_hwc(100, 200)
    out = resize_to_cover(img, (384, 288))
    h, w = out.shape[:2]
    assert w >= 384 and h >= 288


def test_resize_to_cover_preserves_aspect():
    # 200×100 → cover 384×288: scale = max(384/200, 288/100) = max(1.92, 2.88) = 2.88
    # new size = ceil(200*2.88)=576, ceil(100*2.88)=288
    img = _make_hwc(100, 200)
    out = resize_to_cover(img, (384, 288))
    assert out.shape[0] == 288
    assert out.shape[1] >= 384

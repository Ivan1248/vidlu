"""Tests for the Qwen3-VL vision-tower backbone.

Lightweight tests (vision-tower resolution + PIL conversion) always run. The
integration test that loads ``Qwen/Qwen3-VL-8B-Instruct`` runs only when
``IRAP_RUN_HEAVY_TESTS`` is set (needs the 8B weights, ``peft``, GPU).
"""

import os

import numpy as np
import pytest
import torch
from torch import nn

from vidlu_irap_gaim.models.encoders.qwen_vision import (
    Qwen3VLVisionEncoder,
    _resolve_vision_tower,
    _to_pil_uint8,
)

HEAVY = os.environ.get("IRAP_RUN_HEAVY_TESTS")


def test_to_pil_uint8_roundtrip():
    frame = torch.rand(3, 8, 8)
    img = _to_pil_uint8(frame)
    assert img.mode == "RGB"
    assert img.size == (8, 8)  # PIL is (W, H)
    arr = np.asarray(img)
    assert arr.dtype == np.uint8
    # Values track the input (allowing for rounding to 0..255).
    expected = (frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    assert np.array_equal(arr, expected)


def test_to_pil_uint8_clamps():
    frame = torch.tensor([-1.0, 2.0]).view(1, 1, 2).expand(3, 1, 2).contiguous()
    arr = np.asarray(_to_pil_uint8(frame))
    assert arr.min() == 0 and arr.max() == 255


class _Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 2)


class _WithModelVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = _Leaf()
        self.lm_head = nn.Linear(2, 2)


class _WithVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Leaf()


class _NoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Leaf()


def test_resolve_vision_tower_model_visual():
    m = _WithModelVisual()
    tower, path = _resolve_vision_tower(m)
    assert path == "model.visual"
    assert tower is m.model.visual


def test_resolve_vision_tower_visual():
    m = _WithVisual()
    tower, path = _resolve_vision_tower(m)
    assert path == "visual"
    assert tower is m.visual


def test_resolve_vision_tower_missing_raises():
    with pytest.raises(RuntimeError, match="Could not locate the Qwen vision tower"):
        _resolve_vision_tower(_NoTower())


# --- adapter-only checkpointing ---------------------------------------------------

class _FakeQuantLinear(nn.Linear):
    """Stand-in for a bitsandbytes `Linear4bit`.

    Reproduces the part that breaks strict loading: `_save_to_state_dict` writes
    quantization state under the *parameter* name `weight`, and there is no
    `_load_from_state_dict` that would accept those entries back.
    """

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "weight.absmax"] = torch.zeros(2)
        destination[prefix + "weight.quant_state.bitsandbytes__nf4"] = torch.zeros(2)


class _StubVisionEncoder(Qwen3VLVisionEncoder):
    """`Qwen3VLVisionEncoder` with a two-layer stub tower instead of the 8B model."""

    def __init__(self):
        nn.Module.__init__(self)  # skips `_build`, which would download Qwen3-VL
        self.visual = nn.Module()
        self.visual.base = _FakeQuantLinear(4, 4)  # frozen, "quantized"
        self.visual.lora = nn.Linear(4, 4, bias=False)  # trainable adapter
        self.visual.base.requires_grad_(False)
        self._adapter_state_keys = frozenset({"visual.lora.weight"})
        self._set_pixel_stats(None)  # the image processor normalizes


def test_state_dict_holds_only_the_adapter():
    enc = _StubVisionEncoder()
    assert set(enc.state_dict()) == {"visual.lora.weight"}


def test_load_state_dict_roundtrip_is_strict():
    enc = _StubVisionEncoder()
    saved = {k: v.clone() for k, v in enc.state_dict().items()}
    with torch.no_grad():
        enc.visual.lora.weight.add_(1.0)
        base_before = enc.visual.base.weight.clone()

    enc.load_state_dict(saved, strict=True)

    assert torch.equal(enc.visual.lora.weight, saved["visual.lora.weight"])
    assert torch.equal(enc.visual.base.weight, base_before)  # rebuilt, never checkpointed


def test_load_state_dict_accepts_legacy_full_checkpoint():
    """Checkpoints written before adapter-only saving carry base weights + quant state."""
    enc = _StubVisionEncoder()
    legacy = nn.Module.state_dict(enc)  # unfiltered, as the old code saved it
    assert "visual.base.weight.absmax" in legacy  # the keys that made strict loading fail

    enc.load_state_dict(legacy, strict=True)

    assert torch.equal(enc.visual.lora.weight, legacy["visual.lora.weight"])


def test_load_state_dict_still_rejects_unknown_tower_keys():
    enc = _StubVisionEncoder()
    state = dict(enc.state_dict(), **{"visual.bogus.weight": torch.zeros(4)})
    with pytest.raises(RuntimeError, match="visual.bogus.weight"):
        enc.load_state_dict(state, strict=True)


def test_checkpointing_works_as_a_submodule():
    parent = nn.Module()
    parent.frame_encoder = _StubVisionEncoder()
    parent.head = nn.Linear(4, 2)

    state = parent.state_dict()
    assert set(state) == {"frame_encoder.visual.lora.weight", "head.weight", "head.bias"}
    parent.load_state_dict(state, strict=True)


def test_state_dict_rejects_positional_arguments():
    enc = _StubVisionEncoder()
    with pytest.raises(TypeError, match="only keyword arguments"):
        enc.state_dict({}, "prefix.")


# --- trainability -----------------------------------------------------------------

def test_lora_mode_trains_the_adapter_only():
    enc = _StubVisionEncoder()
    enc.set_trainable("lora")
    assert enc.visual.lora.weight.requires_grad
    assert not enc.visual.base.weight.requires_grad


def test_full_finetuning_is_rejected():
    """The base tower is loaded frozen (possibly quantized) and is never checkpointed,
    so unfreezing it would train weights that a resumed run could not restore."""
    enc = _StubVisionEncoder()
    with pytest.raises(ValueError, match="mode='lora'"):
        enc.set_trainable("all")


def test_pool_mode_trains_nothing_in_the_tower():
    enc = _StubVisionEncoder()
    enc.set_trainable("pool")
    assert not any(p.requires_grad for p in enc.parameters())


@pytest.mark.skipif(not HEAVY, reason="set IRAP_RUN_HEAVY_TESTS to load Qwen3-VL weights")
def test_qwen_vision_encoder_integration():
    pytest.importorskip("peft")

    enc = Qwen3VLVisionEncoder(lora_r=8, load_in_4bit=torch.cuda.is_available())
    assert len(enc.lora_parameters()) > 0  # LoRA adapters must be trainable

    x = torch.rand(2, 3, 224, 224)
    fm, pooled = enc(x)
    assert pooled.ndim == 2 and pooled.shape[0] == 2
    assert fm.shape[:2] == (2, pooled.shape[1])

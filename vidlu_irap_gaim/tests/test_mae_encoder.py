"""Tests for the MAE ViT backbone.

The lightweight tests (helper reshape + wiring through ImageSequenceClassifier with
a stub) always run. The integration test that actually downloads
``facebook/vit-mae-base`` runs only when ``IRAP_RUN_HEAVY_TESTS`` is set, since it
pulls a large checkpoint and needs network access.
"""

import os

import pytest
import torch
from torch import nn

from vidlu_irap_gaim.models.classification import ImageSequenceClassifier
from vidlu_irap_gaim.models.encoders.vit import patch_tokens_to_feature_map

HEAVY = os.environ.get("IRAP_RUN_HEAVY_TESTS")


def test_patch_tokens_to_feature_map_grid():
    B, D, patch = 2, 5, 16
    grid = 224 // patch  # 14
    tokens = torch.randn(B, grid * grid, D)
    frame = torch.zeros(B, 3, 224, 224)
    fm = patch_tokens_to_feature_map(tokens, frame, patch)
    assert fm.shape == (B, D, grid, grid)
    # The first token maps to spatial cell (0, 0).
    assert torch.allclose(fm[:, :, 0, 0], tokens[:, 0])


def test_patch_tokens_to_feature_map_bad_count():
    tokens = torch.randn(1, 7, 4)  # 7 is not a valid grid for 224/16
    frame = torch.zeros(1, 3, 224, 224)
    with pytest.raises(ValueError):
        patch_tokens_to_feature_map(tokens, frame, 16)


class _StubMAEEncoder(nn.Module):
    """Mimics MAEEncoder's (feature_map, cls) contract without HF weights."""

    def __init__(self, dim=768, grid=14):
        super().__init__()
        self.proj = nn.Linear(3 * 16 * 16, dim)
        self.grid = grid
        self.dim = dim

    def forward(self, frame):
        B = frame.shape[0]
        patches = frame.unfold(2, 16, 16).unfold(3, 16, 16)  # (B,3,gh,gw,16,16)
        patches = patches.reshape(B, 3, self.grid, self.grid, 16 * 16)
        patches = patches.permute(0, 2, 3, 1, 4).reshape(B, self.grid * self.grid, -1)
        tokens = self.proj(patches)
        feature_map = tokens.transpose(1, 2).reshape(B, self.dim, self.grid, self.grid)
        cls = tokens.mean(dim=1)
        return feature_map, cls

    def pooling_parameters(self):
        return []


def test_mae_style_encoder_wires_into_classifier():
    torch.manual_seed(0)
    model = ImageSequenceClassifier(
        class_counts=(3, 4), sequence_length=2, attention=False,
        encoder_f=_StubMAEEncoder(dim=32),
    )
    x = torch.randn(2, 2, 3, 224, 224)
    outs = model(x)
    assert [o.shape for o in outs] == [(2, 3), (2, 4)]
    # heads sized to sequence_length * pooled_dim = 2 * 32
    assert model.heads[0].in_features == 2 * 32


@pytest.mark.skipif(not HEAVY, reason="set IRAP_RUN_HEAVY_TESTS to download MAE weights")
def test_mae_encoder_integration(tmp_path):
    from vidlu_irap_gaim.models.encoders.mae import mae_vit_encoder

    enc = mae_vit_encoder("facebook/vit-mae-base", params_dir=str(tmp_path))
    enc.eval()
    # Masking must be disabled so every patch token is present and unshuffled.
    assert enc.config.mask_ratio == 0.0

    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        fm1, pooled1 = enc(x)
        fm2, pooled2 = enc(x)

    assert pooled1.shape == (2, enc.config.hidden_size)
    assert fm1.shape[:2] == (2, enc.config.hidden_size)
    assert fm1.shape[-1] == enc.image_size // enc.patch_size
    assert enc.pooling_parameters() == []
    # Determinism proves masking/shuffling is off.
    assert torch.allclose(pooled1, pooled2)
    assert torch.allclose(fm1, fm2)

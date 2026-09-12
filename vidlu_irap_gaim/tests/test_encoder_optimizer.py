"""Tests for `EncoderOptimizerMaker`.

It exists to serve layer-wise LR decay, whose defining property is the ratio between layers'
learning rates. The schedule that has to preserve that ratio is covered by
`tests/optim/test_lr_schedulers.py`.
"""

import pytest
import torch
from torch import nn

from vidlu_irap_gaim.models.classification import ImageSequenceClassifier
from vidlu_irap_gaim.models.encoders import FrameEncoder, PixelStats
from vidlu_irap_gaim.training.optim import EncoderOptimizerMaker


class _TinyEncoder(FrameEncoder):
    """Two "blocks" of decreasing depth, so layer-wise decay has something to order."""

    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([nn.Conv2d(3, dim, 1), nn.Conv2d(dim, dim, 1)])
        self.pool_proj = nn.Conv2d(dim, dim, 1)
        self._set_pixel_stats(PixelStats(mean=(0., 0., 0.), std=(1., 1., 1.)))

    def encode(self, frames):
        x = frames
        for block in self.blocks:
            x = block(x)
        return x, self.pool_proj(x).mean(dim=(2, 3))

    def pool_parameters(self):
        return list(self.pool_proj.parameters())

    def lora_parameters(self):
        return list(self.blocks[1].parameters())

    def last_block_parameters(self):
        return list(self.blocks[1].parameters()) + self.pool_parameters()


def _make_model():
    torch.manual_seed(0)
    model = ImageSequenceClassifier(class_counts=(3, 4), sequence_length=2,
                                    encoder_f=_TinyEncoder)
    model(torch.rand(2, 2, 3, 8, 8))  # builds the heads
    return model


@pytest.mark.parametrize("mode,expected", [
    ("none", set()),
    ("pool", {"frame_encoder.pool_proj.weight", "frame_encoder.pool_proj.bias"}),
    ("lora", {"frame_encoder.blocks.1.weight", "frame_encoder.blocks.1.bias"}),
    ("last_blocks", {"frame_encoder.blocks.1.weight", "frame_encoder.blocks.1.bias",
                     "frame_encoder.pool_proj.weight", "frame_encoder.pool_proj.bias"}),
])
def test_maker_sets_the_requested_trainability(mode, expected):
    model = _make_model()
    EncoderOptimizerMaker(torch.optim.AdamW, trainable=mode, lr=1e-4)(model)
    encoder_trainable = {n for n, p in model.named_parameters()
                         if p.requires_grad and n.startswith("frame_encoder.")}
    assert encoder_trainable == expected


def test_heads_are_always_trained_and_get_their_own_rate():
    model = _make_model()
    optimizer = EncoderOptimizerMaker(torch.optim.AdamW, trainable="none", lr=1e-4,
                                      head_lr_multiplier=10.)(model)
    assert all(p.requires_grad for p in model.head_parameters())
    head_ids = {id(p) for p in model.head_parameters()}
    head_groups = [g for g in optimizer.param_groups if {id(p) for p in g["params"]} == head_ids]
    assert len(head_groups) == 1 and head_groups[0]["lr"] == pytest.approx(1e-3)


def test_maker_runs_before_the_optimizer_sees_the_parameters():
    """The optimizer must receive exactly the parameters the mode makes trainable."""
    model = _make_model()
    optimizer = EncoderOptimizerMaker(torch.optim.AdamW, trainable="pool", lr=1e-4)(model)
    optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert optimized == {id(p) for p in model.parameters() if p.requires_grad}


def test_head_weight_decay_is_separate_from_the_backbones():
    model = _make_model()
    optimizer = EncoderOptimizerMaker(torch.optim.AdamW, trainable="all", lr=1e-4,
                                      weight_decay=0.05, head_weight_decay=1e-4)(model)
    head_ids = {id(p) for p in model.head_parameters()}
    (head_group,) = [g for g in optimizer.param_groups
                     if {id(p) for p in g["params"]} == head_ids]
    assert head_group["weight_decay"] == 1e-4


def test_layer_decay_on_an_encoder_that_does_not_support_it():
    model = _make_model()
    with pytest.raises(NotImplementedError, match="layer-wise LR decay"):
        EncoderOptimizerMaker(torch.optim.AdamW, trainable="all", lr=1e-4,
                              layer_decay=0.65)(model)

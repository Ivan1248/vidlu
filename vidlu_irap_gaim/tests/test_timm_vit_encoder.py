"""Tests for the timm ViT backbones (SigLIP 2, SPAR, DINOv3).

The architecture tests build the real modules with ``pretrained=False``, so they need no
network access and no weights. Only the pretrained-weight test is gated behind
``IRAP_RUN_HEAVY_TESTS``.
"""

import os

import pytest
import torch
from torch import nn

from vidlu_irap_gaim.models.classification import ImageSequenceClassifier
from vidlu_irap_gaim.models.encoders.timm_vit import (SIGLIP2_VARIANT_TO_TIMM_MODEL,
                                                      TimmViTEncoder,
                                                      load_spar_trunk_state_dict)

HEAVY = os.environ.get("IRAP_RUN_HEAVY_TESTS")

# The iRAP pipeline feeds center-cropped 384x288 (W x H) frames, i.e. an 18x24 patch grid
# at patch size 16 – well away from either backbone's pretraining resolution.
FRAME_HW = (288, 384)
PATCH_GRID = (18, 24)


@pytest.fixture(scope="module")
def siglip2():
    return TimmViTEncoder("vit_base_patch16_siglip_512", pretrained=False).eval()


@pytest.fixture(scope="module")
def dinov3():
    return TimmViTEncoder("vit_large_patch16_dinov3", pretrained=False).eval()


# --- the encoder contract ---------------------------------------------------------

@pytest.mark.parametrize("name,dim", [("vit_base_patch16_siglip_512", 768),
                                      ("vit_large_patch16_dinov3", 1024)])
def test_output_widths(name, dim):
    enc = TimmViTEncoder(name, pretrained=False).eval()
    with torch.no_grad():
        feature_map, pooled = enc(torch.rand(2, 3, *FRAME_HW))
    assert pooled.shape == (2, dim)
    assert feature_map.shape == (2, dim, *PATCH_GRID)


def test_siglip2_declares_its_own_normalization(siglip2):
    """mean = std = 0.5, per timm/ViT-B-16-SigLIP2-512's open_clip_config.json."""
    assert siglip2.pixel_stats.mean == (0.5, 0.5, 0.5)
    assert siglip2.pixel_stats.std == (0.5, 0.5, 0.5)


def test_dinov3_declares_imagenet_normalization(dinov3):
    assert dinov3.pixel_stats.mean == pytest.approx((0.485, 0.456, 0.406))
    assert dinov3.pixel_stats.std == pytest.approx((0.229, 0.224, 0.225))


def test_normalization_is_applied_to_inputs(siglip2):
    """Frames arrive in [0, 1]. With mean = std = 0.5 they must be mapped to [-1, 1]."""
    assert siglip2.normalize(torch.ones(1, 3, 2, 2)).unique().tolist() == [1.0]
    assert siglip2.normalize(torch.zeros(1, 3, 2, 2)).unique().tolist() == [-1.0]


def test_dinov3_register_tokens_are_excluded_from_the_feature_map(dinov3):
    # CLS + 4 registers must not end up in the spatial map, or the grid would not fit.
    assert dinov3.model.num_prefix_tokens == 5
    with torch.no_grad():
        feature_map, _ = dinov3(torch.rand(1, 3, *FRAME_HW))
    assert feature_map.shape[-2:] == PATCH_GRID


def test_dinov3_default_pooling_is_cls_plus_avg():
    """DINOv3's own linear evaluation classifies the class token concatenated with the
    average patch token (`dinov3/eval/linear.py`, `use_avgpool`). timm's default head
    averages the patch tokens alone and discards the class token, so the factory
    overrides it – otherwise the probe would not be the protocol DINOv3 reports."""
    from vidlu_irap_gaim.models.encoders.timm_vit import dinov3_vit_encoder

    enc = dinov3_vit_encoder("vit_large_patch16_dinov3", pretrained=False).eval()
    assert enc.pooling == "cls+avg"

    with torch.no_grad():
        _, pooled = enc(torch.rand(2, 3, *FRAME_HW))
        tokens = enc.model.forward_features(enc.normalize_input(torch.zeros(2, 3, *FRAME_HW)))
    assert pooled.shape == (2, 2048)
    # The two halves must be the class token and the patch average, in that order.
    with torch.no_grad():
        _, pooled0 = enc(torch.zeros(2, 3, *FRAME_HW))
    assert torch.allclose(pooled0[:, :1024], tokens[:, 0], atol=1e-5)
    assert torch.allclose(pooled0[:, 1024:], tokens[:, 5:].mean(1), atol=1e-5)


def test_cls_plus_avg_rejected_without_a_class_token(siglip2):
    """SigLIP has no prefix tokens, so silently pooling `tokens[:, 0]` would return a patch."""
    assert siglip2.model.num_prefix_tokens == 0
    with pytest.raises(ValueError, match="needs a class token"):
        TimmViTEncoder("vit_base_patch16_siglip_512", pretrained=False, pooling="cls+avg")


def test_last_blocks_trains_only_the_trailing_blocks():
    enc = TimmViTEncoder("vit_base_patch16_siglip_512", pretrained=False,
                         num_trainable_blocks=2)
    enc.set_trainable("last_blocks")
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert any(n.startswith("model.blocks.11.") for n in trainable)
    assert any(n.startswith("model.blocks.10.") for n in trainable)
    assert not any(n.startswith("model.blocks.9.") for n in trainable)
    assert not any(n.startswith("model.patch_embed.") for n in trainable)
    # The output norm and the pooling head sit downstream of the unfrozen blocks.
    assert any(n.startswith("model.attn_pool.") for n in trainable)


def test_last_blocks_beyond_the_backbones_depth_is_rejected():
    enc = TimmViTEncoder("vit_base_patch16_siglip_512", pretrained=False,
                         num_trainable_blocks=99)
    with pytest.raises(ValueError, match="exceeds"):
        enc.set_trainable("last_blocks")


def test_input_size_resizes_frames():
    enc = TimmViTEncoder("vit_base_patch16_siglip_512", pretrained=False,
                         input_size=256).eval()
    with torch.no_grad():
        feature_map, _ = enc(torch.rand(1, 3, *FRAME_HW))
    assert feature_map.shape[-2:] == (256 // 16, 256 // 16)


# --- trainability -----------------------------------------------------------------

def test_linear_probing_trains_siglips_attention_pooling_head(siglip2):
    siglip2.set_trainable("pool")
    trainable = {n for n, p in siglip2.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("model.attn_pool.") for n in trainable)


def test_linear_probing_trains_nothing_when_pooling_is_parameterless(dinov3):
    # DINOv3 pools by averaging patch tokens, so 'pool' coincides with 'none'.
    assert dinov3.pool_parameters() == []
    dinov3.set_trainable("pool")
    assert not any(p.requires_grad for p in dinov3.parameters())


def test_full_finetuning_trains_everything(siglip2):
    siglip2.set_trainable("all")
    assert all(p.requires_grad for p in siglip2.parameters())


def test_lora_mode_without_adapters_is_rejected(siglip2):
    with pytest.raises(ValueError, match="no parameters for mode='lora'"):
        siglip2.set_trainable("lora")


# --- parameter groups -------------------------------------------------------------

def test_layer_decay_gives_deeper_blocks_larger_learning_rates(siglip2):
    siglip2.set_trainable("all")
    groups = siglip2.param_groups(lr=1e-4, weight_decay=0.05, layer_decay=0.65)
    lrs = sorted({g["lr"] for g in groups})
    assert lrs[-1] == pytest.approx(1e-4)  # the topmost layer trains at the base rate
    # 12 blocks plus the stem, each a factor of `layer_decay` below the next.
    assert lrs[0] == pytest.approx(1e-4 * 0.65 ** 13)
    assert all(p.requires_grad for g in groups for p in g["params"])


def test_layer_decay_excludes_norms_and_embeddings_from_weight_decay(siglip2):
    siglip2.set_trainable("all")
    groups = siglip2.param_groups(lr=1e-4, weight_decay=0.05, layer_decay=0.65)
    assert {g["weight_decay"] for g in groups} == {0.0, 0.05}


def test_param_groups_cover_only_trainable_parameters(siglip2):
    siglip2.set_trainable("pool")
    groups = siglip2.param_groups(lr=1e-3, weight_decay=1e-4, layer_decay=0.65)
    grouped = {id(p) for g in groups for p in g["params"]}
    assert grouped == {id(p) for p in siglip2.parameters() if p.requires_grad}


def test_param_groups_without_layer_decay_is_a_single_group(siglip2):
    siglip2.set_trainable("all")
    (group,) = siglip2.param_groups(lr=1e-4, weight_decay=0.05)
    assert group["lr"] == 1e-4


# --- SPAR checkpoint extraction ---------------------------------------------------

def _trunk_keys():
    return [n for n, _ in TimmViTEncoder("vit_base_patch16_siglip_512",
                                         pretrained=False).model.named_parameters()]


@pytest.mark.parametrize("prefix", ["segmentor.net.model.visual.trunk.",
                                    "model.visual.trunk.",
                                    "visual.trunk."])
def test_spar_extraction_strips_the_wrapper(tmp_path, prefix):
    """SPAR checkpoints wrap the trunk in an open_clip model inside a Lightning segmentor."""
    keys = _trunk_keys()
    checkpoint = {prefix + k: torch.zeros(1) for k in keys}
    checkpoint["model.text.transformer.weight"] = torch.zeros(1)  # the text tower, dropped
    path = tmp_path / "spar.pth"
    torch.save(checkpoint, path)

    assert set(load_spar_trunk_state_dict(path)) == set(keys)


def test_spar_extraction_unwraps_a_lightning_state_dict(tmp_path):
    keys = _trunk_keys()
    path = tmp_path / "spar.ckpt"
    torch.save({"state_dict": {f"segmentor.net.model.visual.trunk.{k}": torch.zeros(1)
                               for k in keys},
                "epoch": 9},
               path)
    assert set(load_spar_trunk_state_dict(path)) == set(keys)


def test_spar_extraction_reports_the_layout_when_no_trunk_is_found(tmp_path):
    path = tmp_path / "spar.pth"
    torch.save({"some.unexpected.layout.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="Top-level key prefixes"):
        load_spar_trunk_state_dict(path)


def test_spar_weights_are_loaded_strictly(tmp_path):
    """A wrong extraction must fail rather than leave the backbone partly random."""
    path = tmp_path / "spar.pth"
    torch.save({"model.visual.trunk.patch_embed.proj.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError):
        TimmViTEncoder("vit_base_patch16_siglip_512", pretrained=False, params_path=path)


def test_spar_round_trip_replaces_the_weights(tmp_path):
    reference = TimmViTEncoder("vit_base_patch16_siglip_512", pretrained=False)
    checkpoint = {f"model.visual.trunk.{k}": v.clone()
                  for k, v in reference.model.state_dict().items()}
    path = tmp_path / "spar.pth"
    torch.save(checkpoint, path)

    enc = TimmViTEncoder("vit_base_patch16_siglip_512", pretrained=False, params_path=path)
    loaded = enc.model.state_dict()
    assert set(loaded) == set(reference.model.state_dict())
    assert all(torch.equal(loaded[k], v) for k, v in reference.model.state_dict().items())


# --- wiring into the classifier ---------------------------------------------------

def test_encoder_wires_into_the_classifier():
    """The heads are sized on the first call from the encoder's actual output, and that
    size carries over to inputs of other resolutions."""
    from functools import partial

    model = ImageSequenceClassifier(
        class_counts=(3, 4), sequence_length=3, attention=False,
        encoder_f=partial(TimmViTEncoder, "vit_base_patch16_siglip_512", pretrained=False))
    assert model.heads is None
    with torch.no_grad():
        outs = model(torch.rand(2, 3, 3, 64, 96))
    assert model.heads[0].in_features == 3 * 768
    assert [tuple(o.shape) for o in outs] == [(2, 3), (2, 4)]
    with torch.no_grad():
        outs = model(torch.rand(1, 3, 3, *FRAME_HW))
    assert [tuple(o.shape) for o in outs] == [(1, 3), (1, 4)]


def test_batched_frame_encoding_matches_per_frame_encoding():
    """Frames are folded into the batch dimension for one encoder call per sequence.

    Not bit-exact: the two calls run at batch sizes 2 and 6, and matmul kernels reduce in
    a different order at different shapes. The tolerance covers that, not a difference in
    what is computed – the ordering of the concatenated features is checked exactly by
    `test_batched_frame_encoding_preserves_frame_order`.
    """
    from functools import partial

    torch.manual_seed(0)
    model = ImageSequenceClassifier(
        class_counts=(3,), sequence_length=3, attention=False,
        encoder_f=partial(TimmViTEncoder, "vit_base_patch16_siglip_512", pretrained=False)).eval()
    encoder = model.frame_encoder
    x = torch.rand(2, 3, 3, 64, 96)
    with torch.no_grad():
        per_frame = torch.cat([encoder(x[:, i])[1] for i in range(3)], dim=1)
        batched = encoder(x.flatten(0, 1))[1].reshape(2, -1)
    assert torch.allclose(per_frame, batched, rtol=1e-4, atol=1e-5)


def test_batched_frame_encoding_preserves_frame_order():
    """The reshape must lay frames out as `[frame 0 | frame 1 | frame 2]` per example,
    matching the per-frame concatenation it replaced. A transposed reshape would
    interleave examples and go unnoticed by a shape check."""
    pooled = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2 * 3, 4)  # (B*S, D)
    seq_feat = pooled.reshape(2, 3 * 4)
    expected = torch.cat([pooled.reshape(2, 3, 4)[:, i] for i in range(3)], dim=1)
    assert torch.equal(seq_feat, expected)


@pytest.mark.skipif(not HEAVY, reason="set IRAP_RUN_HEAVY_TESTS to download SigLIP 2 weights")
def test_pretrained_weights_load(tmp_path):
    enc = TimmViTEncoder(SIGLIP2_VARIANT_TO_TIMM_MODEL["base-512"], pretrained=True).eval()
    untrained = TimmViTEncoder("vit_base_patch16_siglip_512", pretrained=False).eval()
    x = torch.rand(1, 3, *FRAME_HW)
    with torch.no_grad():
        assert not torch.allclose(enc(x)[1], untrained(x)[1])

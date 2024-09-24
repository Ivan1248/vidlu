import typing as T
from functools import partial
from warnings import warn

import torch
import torch.nn.functional as F
from typeguard import typechecked
from torch import nn

import vidlu.modules.components as vmc
import vidlu.modules as vm
from vidlu.modules.other.mask2former.transformer_decoder import \
    MultiScaleMaskedTransformerDecoder
from vidlu.modules.other.mask2former.set_criterion import SetCriterion
from vidlu.data.types import ClassMasks2D

from .models import (SegmentationModel, resnet_v1_backbone, swiftnet_set_mem_efficiency,
                     SwiftNetBase, swiftnet_ladder_f)
from .utils import ladder_input_names
from .initialization import kaiming_resnet


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result


class VidluMultiScaleMaskedTransformerDecoder(vm.Module):
    def __init__(
            self,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            enforce_input_project: bool):
        super().__init__()
        self.module = None
        self.args = self.get_args()

    def build(self, x, mask_features):
        in_channels = mask_features.shape[1]
        if in_channels != self.args.hidden_dim:
            warn(
                f'1×1 convolutions will be used to project to hidden_dim={self.args.hidden_dim} because {in_channels=} is different.')
        self.module = MultiScaleMaskedTransformerDecoder(in_channels=mask_features.shape[1],
                                                         **self.args)

    def forward(self, multi_scale_features, mask_features):
        return self.module(multi_scale_features, mask_features)


maskswifter_transformer_head_f = partial(
    VidluMultiScaleMaskedTransformerDecoder, mask_classification=True, num_queries=100, nheads=8,
    dim_feedforward=2048, dec_layers=9, pre_norm=False, enforce_input_project=False)


@typechecked
class MaskSwifter(SwiftNetBase):
    def __init__(self,
                 backbone_f=resnet_v1_backbone,
                 up_width=128,
                 head_f=maskswifter_transformer_head_f,
                 input_adapter=None,
                 init=kaiming_resnet,
                 laterals=ladder_input_names,
                 # list(f"bulk.unit{i}_{j}" for i, j in zip(range(3), [1] * 3)),
                 lateral_suffix: T.Literal['sum', 'act', ''] = '',
                 stage_count=None,
                 ladder_f=swiftnet_ladder_f,
                 mem_efficiency=1):
        super().__init__(backbone_f=backbone_f,
                         head_f=partial(head_f, hidden_dim=up_width, mask_dim=up_width), init=init,
                         input_adapter=input_adapter, laterals=laterals,
                         lateral_suffix=lateral_suffix, stage_count=stage_count, ladder_f=ladder_f,
                         mem_efficiency=mem_efficiency)

    def post_build(self, *args, **kwargs):
        swiftnet_set_mem_efficiency(self, self.mem_efficiency)
        return True

    def forward(self, x):
        bulk_wio = vm.with_intermediate_outputs(
            self.bulk, [f'ladder.up_blends.{i}' for i in range(len(self.laterals))], inputs=True)

        final_features, multi_scale_features1 = bulk_wio(x)
        multi_scale_features, skips = list(zip(*multi_scale_features1))

        outputs = self.head(multi_scale_features, final_features)
        aux_outputs = outputs.pop('aux_outputs')
        return outputs, aux_outputs

    def forward_sem_seg(self, x, shape=None):
        output, _ = self(x)
        mask_cls_results = output["pred_logits"]
        mask_pred_results = output["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(mask_pred_results, size=shape or x.shape[-2:],
                                          mode="bilinear", align_corners=False)

        del output
        return self.semantic_inference(mask_cls_results, mask_pred_results)

    def semantic_inference(self, mask_class_probs, masks, upsample_masks=True, shape=None):
        if upsample_masks:
            masks = F.interpolate(masks, size=shape, mode="bilinear", align_corners=False)

        # mask_cls: (N, 100, 20)
        # mask_pred (N, 100, 1024, 2048)
        mask_class_probs = F.softmax(mask_class_probs, dim=-1)
        masks = masks.sigmoid()
        # mask_pred /= mask_pred.sum(1, keepdim=True)
        semseg = torch.einsum("nqc,nqhw->nchw", mask_class_probs, masks)
        result = semseg[:, :-1, ...]

        if shape is not None and not upsample_masks:
            result = F.interpolate(result, size=shape, mode="bilinear", align_corners=False)
        return result


def sem_segmentation_to_class_masks(segmentation: torch.Tensor) -> T.List[ClassMasks2D]:
    return [ClassMasks2D(classes=(classes := torch.unique(s).to(segmentation.device)),
                         masks=torch.stack([s == c for c in classes]))
            for s in segmentation]


class MaskFormerOutput(T.TypedDict):
    pred_logits: torch.Tensor
    pred_masks: torch.Tensor


class Mask2FormerSetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        super().__init__()
        self.module = SetCriterion(num_classes, matcher, eos_coef, losses,
                                   num_points, oversample_ratio, importance_sample_ratio)

    def forward(self, output: MaskFormerOutput, aux_outputs: T.List[MaskFormerOutput],
                targets: T.Union[torch.Tensor, T.List[ClassMasks2D]]):
        if isinstance(targets, torch.Tensor):
            targets = sem_segmentation_to_class_masks(targets)
        targets = [dict(labels=tcm.classes, masks=tcm.masks) for tcm in targets]
        outputs = dict(**output, aux_outputs=aux_outputs)

        result = self.module(outputs, targets)

        loss_names = ['loss_ce', 'loss_mask', 'loss_dice']
        main = {k: v for k, v in result.items() if k in loss_names}
        aux = [{name: result[f'{name}_{i}'] for name in loss_names}
               for i in range(len(result) // 3 - 1)]
        return main, aux

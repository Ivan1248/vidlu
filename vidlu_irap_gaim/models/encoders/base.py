"""Frame encoder interface for per-frame vision backbones.

Encoders map a batch of RGB frames in `[0, 1]` to spatial feature maps and pooled vectors:

    forward(frames: (B, 3, H, W)) -> (feature_map: (B, C, H_feat, W_feat), pooled: (B, D))

Key design specifications:
- Input frames are expected in `[0, 1]`; each encoder applies its checkpoint-specific normalization.
- Output widths (`feature_map.shape[1]`, `pooled.shape[1]`) do not depend on the input
  resolution; consumers read them off the output of a first call.
- Trainability modes ('none', 'pool', 'lora', 'last_blocks', 'all') configure parameter gradients.
"""

import abc
import typing as T

import torch
from torch import nn

#: Which parameters of a backbone are trained.
#:
#: - ``'none'``: nothing. The backbone is a fixed feature extractor.
#: - ``'pool'``: only the pooling head (ResNet's SPP, a ViT's attention-pooling head).
#:   Equivalent to ``'none'`` for backbones that pool without parameters (CLS token,
#:   average pooling). This is the linear-probing regime.
#: - ``'lora'``: only low-rank adapters injected into the backbone.
#: - ``'last_blocks'``: only the last few transformer blocks, plus the output norm and
#:   pooling head. The middle ground between probing and full fine-tuning.
#: - ``'all'``: every backbone parameter.
TrainabilityMode = T.Literal["none", "pool", "lora", "last_blocks", "all"]

TRAINABILITY_MODES: tuple[TrainabilityMode, ...] = T.get_args(TrainabilityMode)

# Marks a `FrameEncoder` whose constructor has not declared its normalization yet.
_PIXEL_STATS_UNDECLARED = object()


class PixelStats(T.NamedTuple):
    """Per-channel normalization statistics, in ``[0, 1]`` pixel units."""

    mean: tuple[float, ...]
    std: tuple[float, ...]


def to_pixel_stats(value) -> PixelStats:
    """Coerces a mean/std pair to :class:`PixelStats`.

    Accepts anything with ``mean`` and ``std`` attributes (such as a dataset's
    ``info.pixel_stats``), a mapping with those keys, or a ``(mean, std)`` pair.
    """
    if isinstance(value, PixelStats):
        return value
    if isinstance(value, T.Mapping):
        mean, std = value["mean"], value["std"]
    elif hasattr(value, "mean") and hasattr(value, "std"):
        mean, std = value.mean, value.std
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        mean, std = value
    else:
        raise TypeError(f"Cannot read pixel statistics from {value!r}.")
    return PixelStats(mean=tuple(float(x) for x in mean), std=tuple(float(x) for x in std))


class FrameEncoder(nn.Module, abc.ABC):
    """Base class for per-frame backbones. See the module docstring for the contract.

    Subclasses implement :meth:`encode` and call :meth:`_set_pixel_stats` exactly once
    during construction.
    """

    def __init__(self):
        super().__init__()
        self._trainability: TrainabilityMode = "all"
        self._pixel_stats: PixelStats | None | object = _PIXEL_STATS_UNDECLARED

    # `forward` is a template method: normalization is applied here so that no subclass
    # can forget it, and `encode` sees inputs in the backbone's own units.
    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(self.normalize_input(frames))

    @abc.abstractmethod
    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Maps normalized ``(B, 3, H, W)`` frames to ``(feature_map, pooled)``."""

    # Normalization ################################################################################

    def _set_pixel_stats(self, stats: PixelStats | None) -> None:
        """Declares the normalization this backbone's checkpoint was pretrained with.

        Args:
            stats: Per-channel mean and standard deviation in ``[0, 1]`` units, or
                ``None`` if the backbone normalizes internally (e.g. it owns a Hugging
                Face image processor), in which case frames are passed through as-is.

        The statistics are stored as non-persistent buffers so they follow the module
        across devices and dtypes without entering checkpoints, and additionally kept as
        exact Python floats so that reporting them does not show float32 rounding.
        """
        self._pixel_stats = stats
        if stats is None:
            self.register_buffer("_pixel_mean", None, persistent=False)
            self.register_buffer("_pixel_std", None, persistent=False)
            return
        mean, std = (torch.tensor(x, dtype=torch.float32).view(1, -1, 1, 1)
                     for x in (stats.mean, stats.std))
        if torch.any(std == 0):
            raise ValueError(f"A pixel-statistics standard deviation is zero: {stats.std}.")
        self.register_buffer("_pixel_mean", mean, persistent=False)
        self.register_buffer("_pixel_std", std, persistent=False)

    @property
    def pixel_stats(self) -> PixelStats | None:
        """The normalization applied to inputs, or ``None`` if the backbone normalizes itself."""
        if self._pixel_stats is _PIXEL_STATS_UNDECLARED:
            raise RuntimeError(
                f"{type(self).__name__} did not call `_set_pixel_stats` during construction, so"
                f" the normalization it expects is unknown. Pass `None` if it normalizes"
                f" internally.")
        return self._pixel_stats

    def normalize_input(self, frames: torch.Tensor) -> torch.Tensor:
        if self.pixel_stats is None:
            return frames
        return (frames - self._pixel_mean.to(frames.dtype)) / self._pixel_std.to(frames.dtype)

    # Trainability #################################################################################

    @property
    def trainability(self) -> TrainabilityMode:
        return self._trainability

    def set_trainable(self, mode: TrainabilityMode) -> None:
        """Configures parameter trainability (`requires_grad`) according to the specified mode.

        Args:
            mode: One of `TrainabilityMode`, which documents the modes.

        Raises:
            ValueError: If `mode` is unknown or unsupported by this encoder.
        """
        if mode not in TRAINABILITY_MODES:
            raise ValueError(f"{mode=} is not one of {TRAINABILITY_MODES}.")
        self.requires_grad_(mode == "all")
        if mode in ("pool", "lora", "last_blocks"):
            params = {"pool": self.pool_parameters, "lora": self.lora_parameters,
                      "last_blocks": self.last_block_parameters}[mode]()
            # 'pool' is legitimately empty (CLS-token / average pooling). The other two
            # would silently train nothing, which is not what the mode asks for.
            if not params and mode != "pool":
                raise ValueError(
                    f"{type(self).__name__} exposes no parameters for {mode=}, so it would"
                    f" train nothing. Build it accordingly (a LoRA rank for 'lora', a"
                    f" transformer backbone for 'last_blocks'), or use another mode.")
            for p in params:
                p.requires_grad_(True)
        self._trainability = mode

    def pool_parameters(self) -> list[nn.Parameter]:
        """Parameters of the pooling head, trained in the ``'pool'`` (linear-probe) mode.

        Empty for backbones that pool without parameters (CLS token, average pooling).
        """
        return []

    def lora_parameters(self) -> list[nn.Parameter]:
        """Parameters of injected low-rank adapters, trained in the ``'lora'`` mode."""
        return []

    def last_block_parameters(self) -> list[nn.Parameter]:
        """Parameters of the last few blocks, trained in the ``'last_blocks'`` mode."""
        return []

    # Optimization #################################################################################

    def param_groups(self, *, lr: float, weight_decay: float,
                     layer_decay: float | None = None) -> list[dict]:
        """Builds optimizer parameter groups for the currently trainable parameters.

        Args:
            lr: The learning rate of the topmost layer.
            weight_decay: Weight decay for the parameters that take it.
            layer_decay: Per-block multiplicative decay of the learning rate towards the
                input end, for the backbones that implement it.

        Returns:
            Parameter group dictionaries accepted by PyTorch optimizers.
        """
        if layer_decay is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support layer-wise LR decay"
                f" ({layer_decay=}); it is implemented for transformer backbones only.")
        return [dict(params=[p for p in self.parameters() if p.requires_grad],
                     lr=lr, weight_decay=weight_decay)]

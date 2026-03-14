"""
VLM dataset wrapper for BihSequence.

The dataset returns plain Record objects with standard types. Tokenization
happens in the training step where the processor is available.
"""
from pathlib import Path

from vidlu.data import Dataset, Record

from vidlu_irap_gaim.attrs import get_attrs_to_include
from vidlu_irap_gaim.attribute_frequencies import (
    compute_attribute_frequency_stats,
    frequency_stats_to_attr_to_default_class_idx,
)
from vidlu_irap_gaim.datasets import make_bih_data
from vidlu_irap_gaim.vlm.prompts import DEFAULT_DETAIL_LEVEL
from vidlu_irap_gaim.vlm.response_scheme import (
    ResponseScheme,
    registry,
    SPARSE_SCHEME_NAMES,
    SparseDefaultInstruction,
    get_response_scheme_name,
    make_response_scheme,
)

DEFAULT_RESPONSE_SCHEME_NAME = "standard"


class VLMBihDataset(Dataset):
    """Wraps BihSequence to produce VLM-ready samples for Vidlu.

    Returns plain Record objects with standard types:
    - image: (C, H, W) tensor
    - target: (A,) tensor of attribute indices
    - prompt: str (pre-built prompt, same for all samples)
    - response: str (formatted ground-truth response for training)
    - segment_id: str

    Standard collation will:
    - Stack image tensors to (B, C, H, W)
    - Stack target tensors to (B, A)
    - Keep strings as list[str]

    Tokenization happens in VLMTrainStep/VLMEvalStep where the processor
    is available.

    The selected response scheme is stored in ``self.info.vlm_response_scheme``
    so that eval steps can derive the correct parser without configuration.

    Args:
        base_dataset: BihSequence dataset to wrap.
        response_scheme: ResponseScheme instance that owns prompt building,
            ground-truth formatting, and response parsing.
        attrs_to_include: List of attribute names to include.
        detail_level: Prompt detail level ("attr_desc_vals", "attr_vals", "attr", "none").
    """

    def __init__(
        self,
        base_dataset,
        response_scheme: ResponseScheme,
        attrs_to_include: list[str],
        detail_level: str = DEFAULT_DETAIL_LEVEL,
    ):
        from vidlu.data import Record

        base_info = getattr(base_dataset, "info", None)
        info = Record(base_info if base_info is not None else {},
                      vlm_response_scheme=response_scheme)

        super().__init__(name="vlm_bih", data=base_dataset, info=info)
        self.base_dataset = base_dataset
        self.response_scheme = response_scheme
        self.response_scheme_name = get_response_scheme_name(response_scheme)
        self.attrs_to_include = attrs_to_include
        self.detail_level = detail_level

        # Pre-build prompt (same for all samples - saves computation)
        self._prompt = response_scheme.build_prompt(attrs_to_include, detail_level=detail_level)

    def get_example(self, idx: int) -> Record:
        sample = self.base_dataset[idx]

        # Extract center frame from sequence
        # rgb shape is (S, C, H, W) where S is sequence length
        rgb = sample["rgb"]
        if rgb.ndim == 4:
            image = rgb[0]  # Center frame (index 0), shape (C, H, W)
        else:
            image = rgb  # Already single frame

        # Get target tensor
        target = sample["target"]

        # Build response from ground-truth target using the format's convention
        response = self.response_scheme.format_ground_truth(target, self.attrs_to_include)

        # Get segment ID for tracking
        segment_id = sample["segment_id"]

        return Record(
            image=image,  # (C, H, W) tensor
            target=target,  # (A,) tensor
            prompt=self._prompt,  # str
            response=response,  # str
            segment_id=segment_id,  # str
        )

    @property
    def identifier(self) -> str:
        """Dataset identifier for logging."""
        base_id = getattr(self.base_dataset, "identifier", str(type(self.base_dataset).__name__))
        return f"VLM[{self.response_scheme_name}]({base_id})"


def make_vlm_bih_data(
    response_scheme: str | ResponseScheme = DEFAULT_RESPONSE_SCHEME_NAME,
    detail_level: str = DEFAULT_DETAIL_LEVEL,
    val_quick_size: int | None = None,
    attr_to_default_class_idx: dict[str, int] | None = None,
    sparse_default_instruction: SparseDefaultInstruction = "per_attribute",
) -> dict[str, VLMBihDataset]:
    """Factory for VLM-formatted BIH datasets.

    Creates train/val/test datasets that produce Record items for VLM
    fine-tuning. Tokenization happens in the training step.

    The chosen response scheme governs both the training response format and the
    eval-step response parser. The eval step derives the correct scheme
    automatically from ``dataset.info.vlm_response_scheme``.

    Args:
        detail_level: Prompt detail level ("attr_desc_vals", "attr_vals", "attr", "none").
        val_quick_size: If set, add "val_quick" as first N samples of val for
            in-epoch quick validation. Use None to disable.
        response_scheme_name: Short name of the response scheme to use.
            One of: {schemes}. Defaults to "standard".
        attr_to_default_class_idx: For sparse schemes, mapping of attribute name
            -> default class index. If None and a sparse scheme is selected,
            defaults are computed from the training split (most common class).
        sparse_default_instruction: For sparse schemes, how to describe defaults
            in the prompt: "per_attribute" (Default: X per attr) or "no_none_not"
            (generic instruction for values starting with No/None/Not).

    Returns:
        Dictionary with "train", "val", "test" VLMBihDataset instances.
        Optionally includes "val_quick" when val_quick_size is set.

    Usage in run.py::
        "irap_gaim.make_vlm_bih_data()"
        "irap_gaim.make_vlm_bih_data(response_scheme_name='sparse_indexed')"
        "irap_gaim.make_vlm_bih_data(detail_level='attr_vals', val_quick_size=100)"
    """.format(schemes=", ".join(f'"{k}"' for k in sorted(registry)))
    base_data = make_bih_data()
    attr_to_value_to_class_idx = base_data["train"].info.attr_to_value_to_class_idx
    prompt_config = Path(__file__).parent.parent / "attribute_prompts.yaml"

    if isinstance(response_scheme, str):
        if response_scheme in SPARSE_SCHEME_NAMES and attr_to_default_class_idx is None:
            stats = compute_attribute_frequency_stats(base_data["train"])
            attr_to_default_class_idx = frequency_stats_to_attr_to_default_class_idx(
                stats, attr_to_value_to_class_idx
            )

        response_scheme = make_response_scheme(
            response_scheme,
            attr_to_value_to_class_idx,
            prompt_config,
            attr_to_default_class_idx=attr_to_default_class_idx,
            sparse_default_instruction=sparse_default_instruction,
        )
    attrs_to_include = list(get_attrs_to_include())

    def _make_split(split_ds):
        return VLMBihDataset(split_ds, response_scheme, attrs_to_include, detail_level)

    result = {
        "train": _make_split(base_data["train"]),
        "val": _make_split(base_data["val"]),
        "test": _make_split(base_data["test"]),
    }
    if val_quick_size is not None and val_quick_size > 0:
        result["val_quick"] = result["val"][:val_quick_size]
    return result

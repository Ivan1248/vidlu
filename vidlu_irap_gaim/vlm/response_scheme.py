"""
Prompt-response format definitions for VLM attribute classification.

A ResponseScheme ties together three always-coupled responsibilities:
  1. build_prompt   – generate the text prompt that instructs the VLM
  2. format_ground_truth – encode a target tensor as a training response string
  3. parse_response – parse a VLM-generated response back to predictions

All three must agree on the same textual convention, so they live in one class.
Adding a new response scheme = adding one new ResponseScheme subclass.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal, Sequence

import torch

from irap_data import IGNORE_LABEL_INDEX

from vidlu.utils.collections import Registry

from .response_parser import (
    AttributePrediction,
    build_idx_to_value,
    parse_indexed_response,
    parse_sparse_response,
    parse_vlm_response,
)
from .prompts import DEFAULT_DETAIL_LEVEL, PromptBuilder, DetailLevel


def _build_prompt_builder(
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    prompt_config_path: Path | str | None = None,
) -> PromptBuilder:
    """Build a PromptBuilder, loading YAML config if available.

    Args:
        attr_to_value_to_class_idx: Mapping of attribute name -> {value -> class_idx}.
        prompt_config_path: Path to attribute_prompts.yaml. If None, uses
            attribute_prompts.yaml in the vlm package directory.

    Returns:
        PromptBuilder instance.
    """
    path = Path(prompt_config_path) if prompt_config_path else Path(__file__).parent / "attribute_prompts.yaml"
    if path.exists():
        return PromptBuilder.from_yaml(path, attr_to_value_to_class_idx)
    elif prompt_config_path is None:
        return PromptBuilder(attr_to_value_to_class_idx)
    else:
        raise ValueError(f"Prompt config path does not exist: {prompt_config_path}")


class ResponseScheme(ABC):
    """Abstract base for VLM prompt/response conventions.

    A single instance is shared across dataset, eval step, and predictor.
    It is stateless with respect to the attribute selection: ``attrs_to_include``
    is passed per-call so that one instance can serve both full-attribute and
    chunked (subset) requests.

    Args:
        attr_to_value_to_class_idx: Full attribute metadata.
            Maps attribute name → {value string → class index}.
        prompt_builder: Optional pre-configured PromptBuilder. If None, a
            default PromptBuilder is constructed from ``attr_to_value_to_class_idx``.
    """

    def __init__(
        self,
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
        prompt_builder: PromptBuilder | None = None,
    ):
        self.attr_to_value_to_class_idx = attr_to_value_to_class_idx
        self.prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder(attr_to_value_to_class_idx)

        # Build fast reverse mapping for ground-truth formatting once.
        self._idx_to_value = build_idx_to_value(attr_to_value_to_class_idx)

    def build_prompt(
        self,
        attrs_to_include: Sequence[str],
        *,
        detail_level: "DetailLevel" = DEFAULT_DETAIL_LEVEL,
    ) -> str:
        """Build a prompt string for the given attributes.

        Args:
            attrs_to_include: Ordered attribute names to classify.
            detail_level: Verbosity of per-attribute descriptions.

        Returns:
            Complete prompt string ready to send to the VLM.
        """
        instructions = self._get_response_instructions(attrs_to_include)
        value_formatter = self._get_value_formatter()
        preamble, attr_sections = self.prompt_builder.build_attribute_sections(
            attrs_to_include, detail_level, value_formatter=value_formatter
        )
        parts = [preamble]
        if attr_sections:
            parts.extend(["", "ATTRIBUTES TO CLASSIFY:", "", attr_sections])
        parts.extend(["", instructions])
        return "\n".join(parts)

    def _get_response_instructions(self, attrs_to_include: Sequence[str]) -> str:
        """Return the response format instructions for this scheme."""
        if hasattr(self, "RESPONSE_INSTRUCTIONS"):
            return self.RESPONSE_INSTRUCTIONS
        raise NotImplementedError("Subclasses must implement _get_response_instructions")

    def _get_value_formatter(self) -> Callable[[list[str]], str] | None:
        """Return optional formatter for valid values (e.g. indexed 0=Val0 | 1=Val1)."""
        return None

    @abstractmethod
    def format_ground_truth(
        self,
        target: torch.Tensor,
        attrs_to_include: Sequence[str],
    ) -> str:
        """Encode a ground-truth target tensor as a response string.

        The returned string must parse correctly with ``parse_response``.

        Args:
            target: (A,) tensor of class indices, one per attribute in the
                *full* attribute order (same order as attr_to_value_to_class_idx).
            attrs_to_include: Ordered subset of attributes to include in the
                response (must be a subset of attr_to_value_to_class_idx keys).

        Returns:
            Response string in the format expected by the VLM.
        """

    @abstractmethod
    def parse_response(
        self,
        response_text: str,
        attrs_to_include: Sequence[str],
    ) -> dict[str, AttributePrediction]:
        """Parse a VLM response into structured attribute predictions.

        Args:
            response_text: Raw text output from the VLM.
            attrs_to_include: Ordered attribute names that were requested.

        Returns:
            Dict mapping attribute name → AttributePrediction.
        """

def target_to_class_indices(
    target: torch.Tensor,
    attrs_to_include: Sequence[str],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
) -> dict[str, int]:
    """Return {attr_name: class_idx} for the requested attrs.

    Attributes whose target is ``IGNORE_LABEL_INDEX`` (unlabeled, e.g. IRAP-Vietnam's
    empty flow attributes) are omitted so they are not formatted or supervised.
    """
    attr_order = list(attr_to_value_to_class_idx.keys())
    attr_global_idx = {attr: i for i, attr in enumerate(attr_order)}
    result = {}
    for attr in attrs_to_include:
        if attr not in attr_global_idx:
            continue
        class_idx = int(target[attr_global_idx[attr]].item())
        if class_idx == IGNORE_LABEL_INDEX:
            continue
        result[attr] = class_idx
    return result


def target_to_value_map(
    target: torch.Tensor,
    attrs_to_include: Sequence[str],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    idx_to_value: dict[str, dict[int, str]],
) -> dict[str, str]:
    """Return {attr_name: value_string} for the requested attrs.

    Attributes whose target is ``IGNORE_LABEL_INDEX`` (unlabeled, e.g. IRAP-Vietnam's
    empty flow attributes) are omitted so they are not formatted or supervised.
    """
    attr_order = list(attr_to_value_to_class_idx.keys())
    attr_global_idx = {attr: i for i, attr in enumerate(attr_order)}

    result = {}
    for attr_name in attrs_to_include:
        glob_idx = attr_global_idx.get(attr_name)
        if glob_idx is None:
            raise ValueError(f"Attribute {attr_name} not found in global attribute order")
        class_idx = int(target[glob_idx].item())
        if class_idx == IGNORE_LABEL_INDEX:
            continue
        iv = idx_to_value.get(attr_name)
        if iv is None:
            raise ValueError(f"Attribute {attr_name} not found in attr_to_value_to_class_idx")
        if class_idx not in iv:
            raise ValueError(f"Class index {class_idx} not found for attribute {attr_name}")
        result[attr_name] = iv[class_idx]
    return result


def format_sparse_ground_truth(
    target: torch.Tensor,
    attrs_to_include: Sequence[str],
    format_line: Callable[[str, int, int], str],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    attr_to_default_class_idx: dict[str, int],
) -> str:
    """Format sparse ground truth. format_line(attr, display_idx, class_idx) -> line string."""
    attr_to_idx = target_to_class_indices(
        target, attrs_to_include, attr_to_value_to_class_idx
    )
    lines = [
        format_line(attr, attrs_to_include.index(attr) + 1, idx)
        for attr in attrs_to_include
        if (idx := attr_to_idx.get(attr)) is not None
        and idx != attr_to_default_class_idx[attr]
    ]
    return "All default" if not lines else "\n".join(lines)


def _indexed_value_formatter(vals: list[str]) -> str:
    """Format valid values as 0=Val0 | 1=Val1 for indexed response schemes."""
    return " | ".join(f"{i}={v}" for i, v in enumerate(vals))


registry = Registry()


@registry.register("json")
class JsonResponseScheme(ResponseScheme):
    """JSON object response scheme.

    Prompt instructs the VLM to respond with a JSON object::

        {
          "Area type": "Urban",
          "Land use - driver-side": "Residential"
        }
    """

    RESPONSE_INSTRUCTIONS = """\
RESPONSE FORMAT (JSON):
- Return ONLY a JSON object.
- Use the exact attribute names from the list above as keys.
- Use EXACT values from the valid options for each key.
- No extra text, no JSON, no markdown, no LaTeX.
- Incomplete example:
{
  "Area type": "Urban",
  "Land use - driver-side": "Residential"
}
"""

    def format_ground_truth(
        self,
        target: torch.Tensor,
        attrs_to_include: Sequence[str],
    ) -> str:
        attr_to_value = target_to_value_map(target, attrs_to_include, self.attr_to_value_to_class_idx, self._idx_to_value)
        ordered = {attr: attr_to_value[attr] for attr in attrs_to_include if attr in attr_to_value}
        return json.dumps(ordered, ensure_ascii=False)

    def parse_response(
        self,
        response_text: str,
        attrs_to_include: Sequence[str],
    ) -> dict[str, AttributePrediction]:
        return parse_vlm_response(
            response_text,
            self.attr_to_value_to_class_idx,
            attrs_to_include,
            output_format="json",
        )


@registry.register("standard")
class StandardResponseScheme(ResponseScheme):
    """Standard numbered-line response scheme.

    Prompt instructs the VLM to respond with one line per attribute::

        1: Urban
        2: Residential
        3: None

    Ground-truth and parser use the same 1-based indexed convention.
    """

    RESPONSE_INSTRUCTIONS = """\
RESPONSE FORMAT:
- Return ONLY numbered lines, one per attribute: NUMBER: VALUE
- Use the exact attribute number from the list above.
- Use EXACT values from the valid options.
- No extra text, no JSON, no markdown, no LaTeX.
- Incomplete example:
1: None
2: Urban
3: Present"""

    def format_ground_truth(
        self,
        target: torch.Tensor,
        attrs_to_include: Sequence[str],
    ) -> str:
        attr_to_value = target_to_value_map(
            target,
            attrs_to_include,
            self.attr_to_value_to_class_idx,
            self._idx_to_value,
        )
        lines = [f"{i}: {attr_to_value[attr]}" for i, attr in enumerate(attrs_to_include, 1) if attr in attr_to_value]
        return "\n".join(lines)

    def parse_response(
        self,
        response_text: str,
        attrs_to_include: Sequence[str],
    ) -> dict[str, AttributePrediction]:
        return parse_vlm_response(
            response_text,
            self.attr_to_value_to_class_idx,
            attrs_to_include,
            output_format="standard",
        )

@registry.register("indexed")
class IndexedResponseScheme(ResponseScheme):
    """Indexed format: values shown as 0=Val0 | 1=Val1, response uses integer indices.

    Prompt shows valid values as indexed pairs (e.g. 0=None | 1=Urban | 2=Rural).
    Response format: NUMBER: INDEX (attribute number : value index).
    Reduces token usage and eliminates fuzzy matching.
    """

    RESPONSE_INSTRUCTIONS = """\
RESPONSE FORMAT (INDEXED):
- Return ONLY numbered lines, one per attribute: NUMBER: INDEX
- Use the exact attribute number from the list above.
- INDEX is the integer shown before "=" in the valid values list.
- No extra text, no JSON, no markdown, no LaTeX.
- Incomplete example:
1: 0
2: 3
3: 1"""

    def _get_value_formatter(self) -> Callable[[list[str]], str] | None:
        return _indexed_value_formatter

    def format_ground_truth(
        self,
        target: torch.Tensor,
        attrs_to_include: Sequence[str],
    ) -> str:
        attr_to_idx = target_to_class_indices(
            target, attrs_to_include, self.attr_to_value_to_class_idx
        )
        lines = [
            f"{i}: {attr_to_idx[attr]}"
            for i, attr in enumerate(attrs_to_include, 1)
            if attr in attr_to_idx
        ]
        return "\n".join(lines)

    def parse_response(
        self,
        response_text: str,
        attrs_to_include: Sequence[str],
    ) -> dict[str, AttributePrediction]:
        attr_to_default_class_idx = {a: 0 for a in attrs_to_include}
        return parse_indexed_response(
            response_text,
            attrs_to_include,
            self.attr_to_value_to_class_idx,
            attr_to_default_class_idx,
        )


SparseDefaultInstruction = Literal["per_attribute", "no_none_not"]


class SparseResponseSchemeBase(ResponseScheme):
    """Base for sparse response schemes (omit default attributes from response).

    Subclasses define USE_VALUE_INDICES to choose value vs index format.
    """

    def __init__(
        self,
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
        attr_to_default_class_idx: dict[str, int],
        prompt_builder: PromptBuilder | None = None,
        sparse_default_instruction: SparseDefaultInstruction = "per_attribute",
    ):
        super().__init__(attr_to_value_to_class_idx, prompt_builder)
        self.attr_to_default_class_idx = attr_to_default_class_idx
        self.sparse_default_instruction = sparse_default_instruction

    def _format_default_for_prompt(self, attr: str) -> str:
        """Return the default value string for prompt (value or idx=value)."""
        idx = self.attr_to_default_class_idx[attr]
        val = self._idx_to_value[attr][idx]
        return f"{idx}={val}" if self.USE_VALUE_INDICES else val

    def _get_format_line(self) -> Callable[[str, int, int], str]:
        """Return format_line(attr, display_idx, class_idx) -> line string."""
        if self.USE_VALUE_INDICES:
            return lambda attr, di, ci: f"{di}: {ci}"
        idx_to_value = self._idx_to_value
        return lambda attr, di, ci: f"{di}: {idx_to_value[attr][ci]}"

    def build_prompt(
        self,
        attrs_to_include: Sequence[str],
        *,
        detail_level: "DetailLevel" = DEFAULT_DETAIL_LEVEL,
    ) -> str:
        attr_to_default_value: dict[str, str] | None = None
        if detail_level not in ("attr", "none") and self.sparse_default_instruction == "per_attribute":
            attr_to_default_value = {
                attr: self._format_default_for_prompt(attr)
                for attr in attrs_to_include
            }
        instructions = self._get_response_instructions(attrs_to_include)
        value_formatter = _indexed_value_formatter if self.USE_VALUE_INDICES and detail_level not in ("attr", "none") else None
        preamble, attr_sections = self.prompt_builder.build_attribute_sections(
            attrs_to_include,
            detail_level,
            value_formatter=value_formatter,
            attr_to_default_value=attr_to_default_value,
        )
        parts = [preamble]
        if attr_sections:
            parts.extend(["", "ATTRIBUTES TO CLASSIFY:", "", attr_sections])
        parts.extend(["", instructions])
        return "\n".join(parts)

    def _get_default_phrase(self) -> str:
        """Return the phrase describing which attributes to omit."""
        if self.sparse_default_instruction == "no_none_not":
            return "Do NOT include attributes whose value starts with 'No', 'None', or 'Not'."
        return "If an attribute has its DEFAULT value (marked per attribute above), do NOT include it."

    def _get_response_instructions(self, attrs_to_include: Sequence[str]) -> str:
        format_name = "SPARSE INDEXED" if self.USE_VALUE_INDICES else "SPARSE"
        value_or_index = "INDEX" if self.USE_VALUE_INDICES else "VALUE"
        default_phrase = self._get_default_phrase()
        value_instruction = (
            'INDEX is the integer shown before "=" in the valid values list.'
            if self.USE_VALUE_INDICES
            else "Use EXACT values from the valid options."
        )
        example = "2: 3\n5: 1" if self.USE_VALUE_INDICES else "2: Urban\n5: Present"
        return f"""\
RESPONSE FORMAT ({format_name}):
- Return ONLY numbered lines, one per non-default attribute: NUMBER: {value_or_index}
- Use the exact attribute number from the list above.
- {value_instruction}
- No extra text, no JSON, no markdown, no LaTeX.
- {default_phrase}
- If ALL attributes are default, respond with exactly: All default
- Incomplete example:
{example}"""

    def format_ground_truth(
        self,
        target: torch.Tensor,
        attrs_to_include: Sequence[str],
    ) -> str:
        return format_sparse_ground_truth(
            target,
            attrs_to_include,
            self._get_format_line(),
            attr_to_value_to_class_idx=self.attr_to_value_to_class_idx,
            attr_to_default_class_idx=self.attr_to_default_class_idx,
        )

    def parse_response(
        self,
        response_text: str,
        attrs_to_include: Sequence[str],
    ) -> dict[str, AttributePrediction]:
        return parse_sparse_response(
            response_text,
            attrs_to_include,
            self.attr_to_value_to_class_idx,
            self.attr_to_default_class_idx,
            use_indices=self.USE_VALUE_INDICES,
        )


@registry.register("sparse_standard")
class SparseStandardResponseScheme(SparseResponseSchemeBase):
    """Sparse standard format: only non-default attributes in response.

    Prompt instructs the VLM to omit attributes with the default value.
    Absent attributes are parsed using per-attribute defaults.
    """

    USE_VALUE_INDICES = False


@registry.register("sparse_indexed")
class SparseIndexedResponseScheme(SparseResponseSchemeBase):
    """Sparse indexed format: only non-default attributes, values as integer indices.

    Combines IndexedResponseScheme (0=Val0 | 1=Val1) with SparseStandardResponseScheme
    (omit defaults). Reduces token usage and eliminates fuzzy matching.
    """

    USE_VALUE_INDICES = True


SPARSE_SCHEME_NAMES = frozenset(
    n for n, c in registry.items()
    if issubclass(c, SparseResponseSchemeBase)
)

EXPECTED_SCHEME_NAMES = frozenset(registry.keys())


def get_response_scheme_name(scheme: ResponseScheme) -> str:
    """Return the registry name for a ResponseScheme instance.

    Raises:
        KeyError: If the scheme's class is not in RESPONSE_SCHEME_REGISTRY.
    """
    try:
        return registry.get_name(type(scheme))
    except KeyError as e:
        raise ValueError(str(e)) from e


def make_response_scheme(
    name: str,
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    prompt_config_path: Path | str | None = None,
    attr_to_default_class_idx: dict[str, int] | None = None,
    sparse_default_instruction: SparseDefaultInstruction = "per_attribute",
) -> ResponseScheme:
    """Build a ResponseScheme by short name.

    All schemes are constructed with a PromptBuilder derived from the YAML
    config (when available), so attribute descriptions are consistent regardless
    of which scheme is chosen.

    Args:
        name: One of the registered scheme names (see RESPONSE_SCHEME_REGISTRY).
        attr_to_value_to_class_idx: Mapping of attribute name -> {value -> class_idx}.
        prompt_config_path: Path to attribute_prompts.yaml. If None, uses
            attribute_prompts.yaml in the vlm package directory.
        attr_to_default_class_idx: Required for sparse schemes. Mapping of
            attribute name -> default class index. Ignored for non-sparse schemes.
        sparse_default_instruction: For sparse schemes, how to describe defaults
            in the prompt: "per_attribute" (Default: X per attr) or "no_none_not"
            (generic instruction for values starting with No/None/Not).

    Returns:
        ResponseScheme instance for the given name.

    Raises:
        ValueError: If name is not a registered scheme name, or if a sparse
            scheme is requested without attr_to_default_class_idx.
    """
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown response scheme name: {name!r}. Available: {available}."
        )
    scheme_cls = registry[name]
    prompt_builder = _build_prompt_builder(attr_to_value_to_class_idx, prompt_config_path)
    if issubclass(scheme_cls, SparseResponseSchemeBase):
        if attr_to_default_class_idx is None:
            raise ValueError(
                f"Sparse scheme {name!r} requires attr_to_default_class_idx. "
                "Pass it explicitly or use make_vlm_bih_data() which auto-computes from training data."
            )
        return scheme_cls(
            attr_to_value_to_class_idx,
            attr_to_default_class_idx,
            prompt_builder=prompt_builder,
            sparse_default_instruction=sparse_default_instruction,
        )
    return scheme_cls(attr_to_value_to_class_idx, prompt_builder=prompt_builder)

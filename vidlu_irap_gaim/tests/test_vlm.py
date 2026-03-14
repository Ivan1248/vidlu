"""
Tests for VLM integration components.

These tests verify prompt generation and response parsing without
requiring the actual VLM model to be loaded.
"""

import json
import pytest
import torch

from vidlu_irap_gaim.vlm.prompts import DEFAULT_DETAIL_LEVEL, PromptBuilder
from vidlu_irap_gaim.vlm.response_scheme import (
    EXPECTED_SCHEME_NAMES,
    StandardResponseScheme,
    registry,
    SPARSE_SCHEME_NAMES,
    make_response_scheme,
)
from vidlu_irap_gaim.vlm.response_parser import (
    parse_vlm_response,
    predictions_to_output_tuple,
    _extract_json_from_response,
    _fuzzy_match_value,
)


# Sample attribute metadata for testing
SAMPLE_ATTR_METADATA = {
    "Area type": {"Urban": 0, "Rural": 1},
    "Land use - driver-side": {
        "Residential": 0,
        "Commercial": 1,
        "Industrial": 2,
        "Undeveloped": 3,
    },
    "Roadside severity - driver-side object": {
        "Safety barrier": 0,
        "Tree ≥10cm": 1,
        "Pole": 2,
        "None": 3,
    },
}


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    def test_attr_vals_prompt_contains_attributes(self):
        """attr_vals prompt should list all attributes with valid values."""
        builder = PromptBuilder(SAMPLE_ATTR_METADATA)
        preamble, attr_sections = builder.build_attribute_sections(
            list(SAMPLE_ATTR_METADATA.keys()), detail_level="attr_vals"
        )

        assert "Area type" in attr_sections
        assert "Land use - driver-side" in attr_sections
        assert "Urban" in attr_sections
        assert "Rural" in attr_sections
        assert "Residential" in attr_sections

    def test_attr_desc_vals_longer_than_attr_vals(self):
        """attr_desc_vals prompt should include descriptions, making it longer."""
        builder = PromptBuilder(
            SAMPLE_ATTR_METADATA,
            attribute_descriptions={"Area type": "Urban vs rural classification"},
        )
        attrs = list(SAMPLE_ATTR_METADATA.keys())

        _, attr_vals = builder.build_attribute_sections(attrs, detail_level="attr_vals")
        _, attr_desc_vals = builder.build_attribute_sections(attrs, detail_level=DEFAULT_DETAIL_LEVEL)

        assert len(attr_desc_vals) > len(attr_vals)
        assert "Description:" in attr_desc_vals

    def test_attr_level_omits_values(self):
        """attr level should include attribute names but not valid values."""
        builder = PromptBuilder(SAMPLE_ATTR_METADATA)
        _, attr_sections = builder.build_attribute_sections(
            list(SAMPLE_ATTR_METADATA.keys()), detail_level="attr"
        )

        assert "Area type" in attr_sections
        assert "Land use - driver-side" in attr_sections
        assert "Valid values" not in attr_sections
        assert "Urban" not in attr_sections

    def test_none_level_empty_sections(self):
        """none level should return empty preamble and empty attr sections."""
        builder = PromptBuilder(SAMPLE_ATTR_METADATA)
        preamble, attr_sections = builder.build_attribute_sections(
            list(SAMPLE_ATTR_METADATA.keys()), detail_level="none"
        )

        assert preamble == ""
        assert attr_sections == ""

    def test_attrs_to_include_filters_attributes(self):
        """Only specified attributes should appear in prompt."""
        builder = PromptBuilder(SAMPLE_ATTR_METADATA)
        _, attr_sections = builder.build_attribute_sections(
            ["Area type"], detail_level="attr_vals"
        )

        assert "Area type" in attr_sections
        assert "Land use - driver-side" not in attr_sections

    def test_custom_preamble_overrides_default(self):
        """Custom preamble should replace default."""
        builder = PromptBuilder(SAMPLE_ATTR_METADATA, preamble="CUSTOM PREAMBLE TEXT")
        preamble, _ = builder.build_attribute_sections(
            list(SAMPLE_ATTR_METADATA.keys()), detail_level="attr_vals"
        )

        assert "CUSTOM PREAMBLE TEXT" in preamble
        assert "iRAP" not in preamble  # Default preamble mention


class TestResponseParser:
    """TPleests for response parsing functions."""

    def test_parse_valid_json_response(self):
        """Should correctly parse a valid JSON response."""
        response = '{"Area type": "Urban", "Land use - driver-side": "Commercial"}'

        predictions = parse_vlm_response(
            response,
            SAMPLE_ATTR_METADATA,
            attrs_to_include=["Area type", "Land use - driver-side"],
        )

        assert predictions["Area type"].pred_value == "Urban"
        assert predictions["Area type"].pred_idx == 0
        assert predictions["Land use - driver-side"].pred_value == "Commercial"
        assert predictions["Land use - driver-side"].pred_idx == 1

    def test_parse_json_in_markdown_block(self):
        """Should extract JSON from markdown code block."""
        response = """Here is my analysis:

```json
{"Area type": "Rural", "Land use - driver-side": "Residential"}
```

This road appears to be in a rural area."""

        predictions = parse_vlm_response(
            response,
            SAMPLE_ATTR_METADATA,
            attrs_to_include=["Area type", "Land use - driver-side"],
        )

        assert predictions["Area type"].pred_value == "Rural"
        assert predictions["Area type"].pred_idx == 1

    def test_line_parser_handles_quoted_json_like_lines_and_commas(self):
        """Fallback parsing should handle quoted keys and trailing commas."""
        response = '"Area type": "Urban",\n"Land use - driver-side": "Commercial",\n'
        predictions = parse_vlm_response(
            response,
            SAMPLE_ATTR_METADATA,
            attrs_to_include=["Area type", "Land use - driver-side"],
        )
        assert predictions["Area type"].pred_idx == 0
        assert predictions["Land use - driver-side"].pred_idx == 1

    def test_line_parser_populates_missing_json(self):
        """If JSON extraction fails, line parser must still populate values."""
        # Broken JSON (missing comma) but lines are still parseable.
        response = '```json\n{\n  "Area type": "Urban"\n  "Land use - driver-side": "Commercial"\n}\n```\n'
        predictions = parse_vlm_response(
            response,
            SAMPLE_ATTR_METADATA,
            attrs_to_include=["Area type", "Land use - driver-side"],
        )
        assert predictions["Area type"].pred_idx == 0
        assert predictions["Land use - driver-side"].pred_idx == 1

    def test_key_mapping_and_truncated_value_are_recovered(self):
        """Should recover from near-correct keys and truncated string values."""
        metadata = {
            "Bicycle observed flow": {"None": 0, "Low": 1, "Medium": 2, "High": 3, "Very high": 4, "Unknown": 5},
        }
        # Wrong/truncated key + malformed/truncated value (missing closing quote).
        response = '  "Bicycle observed": "None,\n'
        predictions = parse_vlm_response(
            response,
            metadata,
            attrs_to_include=["Bicycle observed flow"],
        )
        assert predictions["Bicycle observed flow"].pred_value == "None"
        assert predictions["Bicycle observed flow"].pred_idx == 0

    def test_fuzzy_matching_close_values(self):
        """Should fuzzy match similar but not exact values."""
        valid_values = {"Tree ≥10cm": 1, "Pole": 2}

        # Missing the special character
        matched, idx = _fuzzy_match_value("Tree 10cm", valid_values)
        assert idx == 1
        assert matched == "Tree ≥10cm"

    def test_fuzzy_matching_case_insensitive(self):
        """Should match values case-insensitively."""
        valid_values = {"Urban": 0, "Rural": 1}

        matched, idx = _fuzzy_match_value("urban", valid_values)
        assert idx == 0
        assert matched == "Urban"

    def test_fuzzy_matching_no_match(self):
        """Should return -1 for values that don't match."""
        valid_values = {"Urban": 0, "Rural": 1}

        matched, idx = _fuzzy_match_value("Completely different", valid_values)
        assert idx == -1

    def test_invalid_prediction_gets_negative_index(self):
        """Unknown values should get pred_idx = -1."""
        response = '{"Area type": "InvalidValue"}'

        predictions = parse_vlm_response(
            response,
            SAMPLE_ATTR_METADATA,
            attrs_to_include=["Area type"],
        )

        assert predictions["Area type"].pred_idx == -1

    def test_missing_attribute_in_response(self):
        """Missing attributes should get pred_idx = -1."""
        response = '{"Area type": "Urban"}'  # Missing "Land use"

        predictions = parse_vlm_response(
            response,
            SAMPLE_ATTR_METADATA,
            attrs_to_include=["Area type", "Land use - driver-side"],
        )

        assert predictions["Area type"].pred_idx == 0
        assert predictions["Land use - driver-side"].pred_idx == -1

    def test_extract_json_from_text_with_extra_content(self):
        """Should extract JSON even with surrounding text."""
        text = "Based on my analysis, the result is {'Area type': 'Urban'} as shown."

        # Note: This uses single quotes which isn't valid JSON, so it may fail
        # Let's test with proper JSON
        text = 'Based on my analysis, the result is {"Area type": "Urban"} as shown.'
        result = _extract_json_from_response(text)

        assert result == {"Area type": "Urban"}


class TestPredictionsToOutputTuple:
    """Tests for converting predictions to metric-compatible format."""

    def test_valid_predictions_create_correct_tensors(self):
        """Valid predictions should create one-hot tensors."""
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

        predictions = {
            "Area type": AttributePrediction("Area type", "Urban", 0),
            "Land use - driver-side": AttributePrediction("Land use - driver-side", "Commercial", 1),
        }

        out = predictions_to_output_tuple(
            predictions,
            SAMPLE_ATTR_METADATA,
            attrs_order=["Area type", "Land use - driver-side"],
        )

        assert len(out) == 2
        assert out[0].shape == (1, 2)  # Area type has 2 classes
        assert out[1].shape == (1, 4)  # Land use has 4 classes
        assert out[0].argmax(1).item() == 0  # Predicted Urban
        assert out[1].argmax(1).item() == 1  # Predicted Commercial

    def test_invalid_prediction_raises_by_default(self):
        """Invalid predictions should raise by default (no silent fallback)."""
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

        predictions = {
            "Area type": AttributePrediction("Area type", "Unknown", -1),
        }

        with pytest.raises(ValueError):
            predictions_to_output_tuple(
                predictions,
                SAMPLE_ATTR_METADATA,
                attrs_order=["Area type"],
            )

    def test_invalid_prediction_can_be_allowed_explicitly(self):
        """If explicitly allowed, invalid predictions map to class 0 (biased)."""
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

        predictions = {
            "Area type": AttributePrediction("Area type", "Unknown", -1),
        }

        out = predictions_to_output_tuple(
            predictions,
            SAMPLE_ATTR_METADATA,
            attrs_order=["Area type"],
            allow_invalid=True,
        )

        assert out[0].argmax(1).item() == 0

    def test_missing_non_required_attr_is_allowed(self):
        """Missing predictions for non-required attrs should not raise."""
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

        predictions = {
            "Area type": AttributePrediction("Area type", "Urban", 0),
        }

        out = predictions_to_output_tuple(
            predictions,
            SAMPLE_ATTR_METADATA,
            attrs_order=["Area type", "Land use - driver-side"],
            required_attrs={"Area type"},
        )

        assert out[0].argmax(1).item() == 0
        assert out[1].shape == (1, 4)
        assert out[1].argmax(1).item() == 0  # filler for non-required attr

    def test_missing_required_attr_raises(self):
        """Missing prediction for required attr should raise (fail-fast)."""
        predictions = {}

        with pytest.raises(ValueError):
            predictions_to_output_tuple(
                predictions,
                SAMPLE_ATTR_METADATA,
                attrs_order=["Area type"],
                required_attrs={"Area type"},
            )


class TestYAMLConfig:
    """Tests for YAML configuration loading."""

    def test_load_from_yaml(self, tmp_path):
        """Should load prompt config from YAML file."""
        yaml_content = """
preamble: Custom preamble text

attributes:
  "Area type":
    description: Whether urban or rural
    values:
      Urban: Dense area
      Rural: Open area
    tips:
      - Look for buildings
"""
        yaml_file = tmp_path / "test_prompts.yaml"
        yaml_file.write_text(yaml_content)

        builder = PromptBuilder.from_yaml(yaml_file, SAMPLE_ATTR_METADATA)
        scheme = StandardResponseScheme(SAMPLE_ATTR_METADATA, prompt_builder=builder)
        prompt = scheme.build_prompt(
            list(SAMPLE_ATTR_METADATA.keys()), detail_level="attr_desc_vals"
        )

        assert "Custom preamble text" in prompt
        assert "Whether urban or rural" in prompt


class TestResponseSchemes:
    """Tests for ResponseScheme implementations."""

    # Reuse the shared metadata from the top of this module
    ATTRS = ["Area type", "Land use - driver-side", "Roadside severity - driver-side object"]

    # Target tensor: class index per attribute in full attribute order
    # Area type=0 (Urban), Land use=1 (Commercial), Roadside=3 (None)
    TARGET = torch.tensor([0, 1, 3])

    def _make_standard(self):
        from vidlu_irap_gaim.vlm.response_scheme import StandardResponseScheme

        return StandardResponseScheme(SAMPLE_ATTR_METADATA)

    def _make_json(self):
        from vidlu_irap_gaim.vlm.response_scheme import JsonResponseScheme

        return JsonResponseScheme(SAMPLE_ATTR_METADATA)

    def _make_sparse(self):
        from vidlu_irap_gaim.vlm.response_scheme import SparseStandardResponseScheme

        attr_to_default_class_idx = {attr: 0 for attr in SAMPLE_ATTR_METADATA}
        return SparseStandardResponseScheme(
            SAMPLE_ATTR_METADATA, attr_to_default_class_idx
        )

    # ------------------------------------------------------------------
    # StandardResponseScheme
    # ------------------------------------------------------------------

    def test_standard_build_prompt_contains_attrs(self):
        fmt = self._make_standard()
        prompt = fmt.build_prompt(self.ATTRS, detail_level="attr_vals")
        assert "Area type" in prompt
        assert "Land use - driver-side" in prompt
        # Standard format instructs numbered lines
        assert any(kw in prompt.lower() for kw in ["number", "numbered", "value", "line"])

    def test_standard_format_ground_truth(self):
        fmt = self._make_standard()
        gt = fmt.format_ground_truth(self.TARGET, self.ATTRS)
        # Numbered lines: "1: Urban", "2: Commercial", "3: None"
        assert "1: Urban" in gt
        assert "2: Commercial" in gt
        assert "3: None" in gt

    def test_standard_parse_response(self):
        fmt = self._make_standard()
        response = "1: Urban\n2: Commercial\n3: None"
        preds = fmt.parse_response(response, self.ATTRS)
        assert "Area type" in preds
        assert preds["Area type"].pred_idx == 0  # Urban
        assert "Land use - driver-side" in preds
        assert preds["Land use - driver-side"].pred_idx == 1  # Commercial

    def test_standard_round_trip(self):
        """format_ground_truth → parse_response must recover original class indices."""
        fmt = self._make_standard()
        gt = fmt.format_ground_truth(self.TARGET, self.ATTRS)
        preds = fmt.parse_response(gt, self.ATTRS)

        attr_order = list(SAMPLE_ATTR_METADATA.keys())
        for i, attr in enumerate(self.ATTRS):
            global_idx = attr_order.index(attr)
            expected_class_idx = int(self.TARGET[global_idx].item())
            assert preds[attr].pred_idx == expected_class_idx, (
                f"{attr}: expected class {expected_class_idx}, got {preds[attr].pred_idx}"
            )

    # ------------------------------------------------------------------
    # JsonResponseScheme
    # ------------------------------------------------------------------

    def test_json_build_prompt_contains_attrs(self):
        fmt = self._make_json()
        prompt = fmt.build_prompt(self.ATTRS, detail_level="attr_vals")
        assert "Area type" in prompt
        assert any(kw in prompt.lower() for kw in ["json", "object"])

    def test_json_format_ground_truth(self):
        fmt = self._make_json()
        gt = fmt.format_ground_truth(self.TARGET, self.ATTRS)
        data = json.loads(gt)
        assert data["Area type"] == "Urban"
        assert data["Land use - driver-side"] == "Commercial"
        assert data["Roadside severity - driver-side object"] == "None"

    def test_json_round_trip(self):
        """format_ground_truth → parse_response must recover original class indices."""
        fmt = self._make_json()
        gt = fmt.format_ground_truth(self.TARGET, self.ATTRS)
        preds = fmt.parse_response(gt, self.ATTRS)

        attr_order = list(SAMPLE_ATTR_METADATA.keys())
        for i, attr in enumerate(self.ATTRS):
            global_idx = attr_order.index(attr)
            expected_class_idx = int(self.TARGET[global_idx].item())
            assert preds[attr].pred_idx == expected_class_idx, (
                f"{attr}: expected class {expected_class_idx}, got {preds[attr].pred_idx}"
            )

    # ------------------------------------------------------------------
    # SparseStandardResponseScheme
    # ------------------------------------------------------------------

    def test_sparse_standard_build_prompt(self):
        fmt = self._make_sparse()
        prompt = fmt.build_prompt(self.ATTRS)
        assert "Area type" in prompt
        assert "do NOT include" in prompt
        assert "All default" in prompt

    def test_sparse_standard_format_ground_truth(self):
        fmt = self._make_sparse()
        # TARGET is [0, 1, 3]
        # Attr 1 (Area type): 0 (Default) -> OMIT
        # Attr 2 (Land use): 1 (Commercial) -> INCLUDE "2: Commercial"
        # Attr 3 (Roadside): 3 (None) -> INCLUDE "3: None"
        gt = fmt.format_ground_truth(self.TARGET, self.ATTRS)
        assert "1: Urban" not in gt
        assert "2: Commercial" in gt
        assert "3: None" in gt

    def test_sparse_standard_format_ground_truth_all_default(self):
        fmt = self._make_sparse()
        all_default = torch.zeros(len(SAMPLE_ATTR_METADATA), dtype=torch.long)
        gt = fmt.format_ground_truth(all_default, self.ATTRS)
        assert gt == "All default"

    def test_sparse_standard_parse_response_partial(self):
        fmt = self._make_sparse()
        # Only attribute 2 (Commercial) is present in response.
        # Attributes 1 and 3 should be assigned default (index 0).
        response = "2: Commercial"
        preds = fmt.parse_response(response, self.ATTRS)

        assert preds["Area type"].pred_idx == 0  # Default (Urban)
        assert preds["Land use - driver-side"].pred_idx == 1  # Parsed (Commercial)
        assert preds["Roadside severity - driver-side object"].pred_idx == 0  # Default (Safety barrier)

    def test_sparse_standard_parse_response_all_default(self):
        fmt = self._make_sparse()
        preds = fmt.parse_response("All default", self.ATTRS)
        for attr in self.ATTRS:
            assert preds[attr].pred_idx == 0

    def test_sparse_standard_round_trip(self):
        """format_ground_truth → parse_response must recover original class indices."""
        fmt = self._make_sparse()
        # Full round trip with various values including defaults
        gt = fmt.format_ground_truth(self.TARGET, self.ATTRS)
        preds = fmt.parse_response(gt, self.ATTRS)

        attr_order = list(SAMPLE_ATTR_METADATA.keys())
        for attr in self.ATTRS:
            global_idx = attr_order.index(attr)
            expected_class_idx = int(self.TARGET[global_idx].item())
            assert preds[attr].pred_idx == expected_class_idx, (
                f"{attr}: expected class {expected_class_idx}, got {preds[attr].pred_idx}"
            )


class TestResponseSchemeRegistry:
    """Tests for the centralized make_response_scheme factory."""

    def test_all_registered_names_produce_correct_types(self):
        """make_response_scheme returns the right class for each registered name."""
        from vidlu_irap_gaim.vlm.response_scheme import (
            StandardResponseScheme,
            JsonResponseScheme,
            SparseStandardResponseScheme,
            IndexedResponseScheme,
            SparseIndexedResponseScheme,
        )

        expected = {
            "standard": StandardResponseScheme,
            "json": JsonResponseScheme,
            "sparse_standard": SparseStandardResponseScheme,
            "indexed": IndexedResponseScheme,
            "sparse_indexed": SparseIndexedResponseScheme,
        }
        attr_to_default_class_idx = {attr: 0 for attr in SAMPLE_ATTR_METADATA}
        for name, cls in expected.items():
            scheme = make_response_scheme(
                name,
                SAMPLE_ATTR_METADATA,
                attr_to_default_class_idx=(
                    attr_to_default_class_idx if name in SPARSE_SCHEME_NAMES else None
                ),
            )
            assert isinstance(scheme, cls), f"Expected {cls.__name__} for name {name!r}"

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown response scheme name"):
            make_response_scheme("nonexistent", SAMPLE_ATTR_METADATA)

    def test_sparse_scheme_requires_attr_to_default_class_idx(self):
        """Sparse schemes raise if attr_to_default_class_idx is not provided."""
        with pytest.raises(ValueError, match="requires attr_to_default_class_idx"):
            make_response_scheme("sparse_indexed", SAMPLE_ATTR_METADATA)

    def test_error_message_lists_available_names(self):
        with pytest.raises(ValueError) as exc_info:
            make_response_scheme("bad_name", SAMPLE_ATTR_METADATA)
        msg = str(exc_info.value)
        for name in registry:
            assert name in msg

    def test_registry_contains_all_scheme_classes(self):
        """RESPONSE_SCHEME_REGISTRY must include all expected schemes."""
        assert set(registry.keys()) == EXPECTED_SCHEME_NAMES


class TestIndexedResponseScheme:
    """Tests for IndexedResponseScheme round-trip behaviour."""

    ATTRS = ["Area type", "Land use - driver-side", "Roadside severity - driver-side object"]
    # Area type=0 (Urban), Land use=1 (Commercial), Roadside=3 (None)
    TARGET = torch.tensor([0, 1, 3])

    def _make(self):
        from vidlu_irap_gaim.vlm.response_scheme import IndexedResponseScheme

        return IndexedResponseScheme(SAMPLE_ATTR_METADATA)

    def test_prompt_contains_indexed_values(self):
        fmt = self._make()
        prompt = fmt.build_prompt(self.ATTRS)
        # Value formatter shows 0=Val | 1=Val style
        assert "0=" in prompt
        assert "1=" in prompt

    def test_format_ground_truth_uses_integer_indices(self):
        fmt = self._make()
        gt = fmt.format_ground_truth(self.TARGET, self.ATTRS)
        # Expect "1: 0" (attr1 = class 0), "2: 1" (attr2 = class 1), "3: 3" (attr3 = class 3)
        assert "1: 0" in gt
        assert "2: 1" in gt
        assert "3: 3" in gt

    def test_round_trip(self):
        fmt = self._make()
        gt = fmt.format_ground_truth(self.TARGET, self.ATTRS)
        preds = fmt.parse_response(gt, self.ATTRS)

        attr_order = list(SAMPLE_ATTR_METADATA.keys())
        for attr in self.ATTRS:
            global_idx = attr_order.index(attr)
            expected = int(self.TARGET[global_idx].item())
            assert preds[attr].pred_idx == expected, (
                f"{attr}: expected class {expected}, got {preds[attr].pred_idx}"
            )


class TestSparseIndexedResponseScheme:
    """Tests for SparseIndexedResponseScheme round-trip behaviour."""

    ATTRS = ["Area type", "Land use - driver-side", "Roadside severity - driver-side object"]
    # Area type=0 (default=Urban), Land use=1 (Commercial), Roadside=3 (None)
    TARGET = torch.tensor([0, 1, 3])

    def _make(self):
        from vidlu_irap_gaim.vlm.response_scheme import SparseIndexedResponseScheme

        attr_to_default_class_idx = {attr: 0 for attr in SAMPLE_ATTR_METADATA}
        return SparseIndexedResponseScheme(
            SAMPLE_ATTR_METADATA, attr_to_default_class_idx
        )

    def test_prompt_contains_indexed_values_and_sparse_instructions(self):
        fmt = self._make()
        prompt = fmt.build_prompt(self.ATTRS)
        assert "0=" in prompt
        assert "All default" in prompt
        assert "do NOT include" in prompt

    def test_format_ground_truth_omits_default_and_uses_indices(self):
        fmt = self._make()
        gt = fmt.format_ground_truth(self.TARGET, self.ATTRS)
        # Area type is default (class 0) -> should NOT appear
        assert "1:" not in gt
        # Land use class 1 -> "2: 1"
        assert "2: 1" in gt
        # Roadside class 3 -> "3: 3"
        assert "3: 3" in gt

    def test_format_ground_truth_all_default_returns_marker(self):
        fmt = self._make()
        all_default = torch.zeros(len(SAMPLE_ATTR_METADATA), dtype=torch.long)
        gt = fmt.format_ground_truth(all_default, self.ATTRS)
        assert gt == "All default"

    def test_parse_response_all_default_assigns_zero(self):
        fmt = self._make()
        preds = fmt.parse_response("All default", self.ATTRS)
        for attr in self.ATTRS:
            assert preds[attr].pred_idx == 0

    def test_round_trip(self):
        fmt = self._make()
        gt = fmt.format_ground_truth(self.TARGET, self.ATTRS)
        preds = fmt.parse_response(gt, self.ATTRS)

        attr_order = list(SAMPLE_ATTR_METADATA.keys())
        for attr in self.ATTRS:
            global_idx = attr_order.index(attr)
            expected = int(self.TARGET[global_idx].item())
            assert preds[attr].pred_idx == expected, (
                f"{attr}: expected class {expected}, got {preds[attr].pred_idx}"
            )


class TestVLMDatasetSchemeInfo:
    """Tests that VLMBihDataset stores scheme metadata accessible to eval steps."""

    def test_dataset_info_contains_response_scheme(self):
        """VLMBihDataset must expose vlm_response_scheme on info."""
        from unittest.mock import MagicMock
        from vidlu_irap_gaim.vlm.finetuning.dataset import VLMBihDataset
        from vidlu_irap_gaim.vlm.response_scheme import StandardResponseScheme

        scheme = StandardResponseScheme(SAMPLE_ATTR_METADATA)

        # Minimal mock for base_dataset
        mock_ds = MagicMock()
        mock_ds.info = None
        mock_ds.__len__ = MagicMock(return_value=5)

        attrs = list(SAMPLE_ATTR_METADATA.keys())
        dataset = VLMBihDataset(mock_ds, scheme, attrs)

        assert hasattr(dataset.info, "vlm_response_scheme")
        assert dataset.info.vlm_response_scheme is scheme
        assert dataset.response_scheme_name == "standard"

    def test_dataset_info_propagates_base_info_fields(self):
        """Augmented info must still contain original base info fields."""
        from unittest.mock import MagicMock
        from vidlu.data import Record
        from vidlu_irap_gaim.vlm.finetuning.dataset import VLMBihDataset
        from vidlu_irap_gaim.vlm.response_scheme import StandardResponseScheme

        scheme = StandardResponseScheme(SAMPLE_ATTR_METADATA)
        base_info = Record(problem="multi_attribute_classification", some_field="hello")

        mock_ds = MagicMock()
        mock_ds.info = base_info
        mock_ds.__len__ = MagicMock(return_value=5)

        attrs = list(SAMPLE_ATTR_METADATA.keys())
        dataset = VLMBihDataset(mock_ds, scheme, attrs)

        assert dataset.info.problem == "multi_attribute_classification"
        assert dataset.info.some_field == "hello"
        assert dataset.info.vlm_response_scheme is scheme

    def test_extract_response_scheme_from_data_succeeds(self):
        """_extract_response_scheme_from_data retrieves scheme from dataset info."""
        from unittest.mock import MagicMock
        from vidlu.data import Record
        from vidlu_irap_gaim.vlm.finetuning.steps import _extract_response_scheme_from_data
        from vidlu_irap_gaim.vlm.response_scheme import SparseIndexedResponseScheme

        attr_to_default_class_idx = {attr: 0 for attr in SAMPLE_ATTR_METADATA}
        scheme = SparseIndexedResponseScheme(
            SAMPLE_ATTR_METADATA, attr_to_default_class_idx
        )
        mock_ds = MagicMock()
        mock_ds.info = Record(vlm_response_scheme=scheme)

        extracted = _extract_response_scheme_from_data({"train": mock_ds, "val": mock_ds})
        assert extracted is scheme

    def test_extract_response_scheme_from_data_raises_without_info(self):
        """_extract_response_scheme_from_data raises RuntimeError when info is missing."""
        from unittest.mock import MagicMock
        from vidlu_irap_gaim.vlm.finetuning.steps import _extract_response_scheme_from_data

        mock_ds = MagicMock()
        mock_ds.info = None

        with pytest.raises(RuntimeError, match="vlm_response_scheme"):
            _extract_response_scheme_from_data({"train": mock_ds})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

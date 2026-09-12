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
    _extract_json_from_response,
    _fuzzy_match_value,
)
from vidlu_irap_gaim.vlm.predictions import (
    attribute_predictions_to_one_hot_outputs)


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


class TestPredictionConversion:
    """Tests for converting predictions to metric-compatible format."""

    ATTRS = ["Area type", "Land use - driver-side"]

    def _convert(self, predictions, *, attrs_order=None, **kwargs):
        return attribute_predictions_to_one_hot_outputs(
            [predictions], SAMPLE_ATTR_METADATA, attrs_order=attrs_order or self.ATTRS, **kwargs)

    def test_valid_predictions_create_correct_tensors(self):
        """Valid predictions should create one-hot tensors."""
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

        out, _ = self._convert({
            "Area type": AttributePrediction("Area type", "Urban", 0),
            "Land use - driver-side":
                AttributePrediction("Land use - driver-side", "Commercial", 1),
        })

        assert len(out) == 2
        assert out[0].shape == (1, 2)  # Area type has 2 classes
        assert out[1].shape == (1, 4)  # Land use has 4 classes
        assert out[0].argmax(1).item() == 0  # Predicted Urban
        assert out[1].argmax(1).item() == 1  # Predicted Commercial

    def test_invalid_prediction_is_one_hot_at_class_0_and_flagged(self):
        """The output scores an unusable response as class 0 – generously, since class 0
        is often the majority value – and the mask lets the caller exclude it instead
        (see `vlm.scoring.update_both_scorings`)."""
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

        out, is_invalid = self._convert(
            {"Area type": AttributePrediction("Area type", "Unknown", -1),
             "Land use - driver-side":
                 AttributePrediction("Land use - driver-side", "Commercial", 1)})

        assert out[0].argmax(1).item() == 0
        assert out[1].argmax(1).item() == 1
        assert is_invalid.tolist() == [[True, False]]

    def test_an_omitted_attribute_is_flagged_too(self):
        _, is_invalid = self._convert({})
        assert is_invalid.tolist() == [[True, True]]

    def test_attributes_outside_the_prompt_get_a_dummy(self):
        """Metrics only index the attributes that were asked about; the rest need
        a placeholder to keep the tuple aligned with the global attribute order."""
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

        out, is_invalid = self._convert(
            {"Area type": AttributePrediction("Area type", "Urban", 0)},
            attrs_to_include=["Area type"])

        assert out[0].shape == (1, 2)
        assert out[1].shape == (1, 1)  # dummy, never read
        assert is_invalid.tolist() == [[False, False]]  # not asked about, so not invalid


class _RecordingMetric:
    """A metric that only remembers what it was scored against."""

    def __init__(self):
        self.updates = []

    def reset(self):
        self.updates.clear()

    def update(self, iter_result):
        self.updates.append(iter_result)


class TestMetricArms:
    """Both scorings of an unusable response are applied, to their own metric set.

    The bug these guard against is subtle and was live in two evaluators: under
    "ignore" the exclusion lives in the *targets* the conversion returns, not in
    its outputs, so scoring against the caller's original targets silently
    reverts to "class0" while still being labelled "ignore".
    """

    ATTRS = ["Area type", "Land use - driver-side"]

    def _run(self, predictions, targets):
        from vidlu_irap_gaim.vlm.scoring import update_both_scorings

        class0_metric, ignore_metric = _RecordingMetric(), _RecordingMetric()
        update_both_scorings([class0_metric], [ignore_metric], [predictions], targets,
                           SAMPLE_ATTR_METADATA, self.ATTRS,
                           attrs_to_include=self.ATTRS)
        return class0_metric.updates[0], ignore_metric.updates[0]

    def test_the_ignore_arm_is_scored_against_masked_targets(self):
        """The regression test: the ignore scoring must not see the raw targets."""
        from irap_data import IGNORE_LABEL_INDEX
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction
        import torch

        targets = torch.tensor([[1, 2]])
        class0_update, ignore_update = self._run(
            {"Area type": AttributePrediction("Area type", "Unknown", -1),
             "Land use - driver-side":
                 AttributePrediction("Land use - driver-side", "Industrial", 2)},
            targets)

        assert ignore_update.target[0, 0].item() == IGNORE_LABEL_INDEX
        assert ignore_update.target[0, 1].item() == 2  # the usable one is untouched
        assert class0_update.target[0, 0].item() == 1  # the class0 scoring sees the truth
        assert targets[0, 0].item() == 1  # neither scoring mutates the caller's tensor

    def test_the_scorings_disagree_on_an_unusable_response(self):
        """Wiring both scorings to one policy would make these identical."""
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction
        import torch

        class0_update, ignore_update = self._run(
            {"Area type": AttributePrediction("Area type", "Unknown", -1),
             "Land use - driver-side":
                 AttributePrediction("Land use - driver-side", "Industrial", 2)},
            torch.tensor([[1, 2]]))

        assert not torch.equal(class0_update.target, ignore_update.target)

    def test_a_sample_with_no_responses_is_scored_by_each_scoring_in_turn(self):
        """A segment the model never responded to: class 0 everywhere in one scoring,
        excluded everywhere in the other. This is how a missing segment stays in
        the comparable denominator without polluting the conditional one."""
        from irap_data import IGNORE_LABEL_INDEX
        import torch

        class0_update, ignore_update = self._run({}, torch.tensor([[1, 2]]))

        assert (ignore_update.target == IGNORE_LABEL_INDEX).all()
        assert torch.equal(class0_update.target, torch.tensor([[1, 2]]))
        assert [o.argmax(1).item() for o in class0_update.out] == [0, 0]

    def test_valid_responses_score_identically_in_both_scorings(self):
        """With nothing unusable there is nothing to exclude, so the two scorings
        agree – which is what makes a gap between them read as the invalid rate."""
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction
        import torch

        targets = torch.tensor([[1, 2]])
        class0_update, ignore_update = self._run(
            {"Area type": AttributePrediction("Area type", "Rural", 1),
             "Land use - driver-side":
                 AttributePrediction("Land use - driver-side", "Industrial", 2)},
            targets)

        assert torch.equal(class0_update.target, ignore_update.target)
        assert torch.equal(ignore_update.target, targets)


class TestResponseCounting:
    """`count_scored_and_invalid_responses` defines the invalid rate the ignore scoring is
    read against."""

    ATTRS = ["Area type", "Land use - driver-side"]

    def test_an_unmatched_value_counts_as_invalid(self):
        from vidlu_irap_gaim.vlm.scoring import count_scored_and_invalid_responses
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

        num_scored, num_invalid = count_scored_and_invalid_responses(
            {"Area type": AttributePrediction("Area type", "Unknown", -1),
             "Land use - driver-side":
                 AttributePrediction("Land use - driver-side", "Industrial", 2)},
            self.ATTRS, SAMPLE_ATTR_METADATA)

        assert (num_scored, num_invalid) == (2, 1)

    def test_an_omitted_attribute_counts_as_invalid(self):
        from vidlu_irap_gaim.vlm.scoring import count_scored_and_invalid_responses
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

        num_scored, num_invalid = count_scored_and_invalid_responses(
            {"Area type": AttributePrediction("Area type", "Rural", 1)}, self.ATTRS,
            SAMPLE_ATTR_METADATA)

        assert (num_scored, num_invalid) == (2, 1)

    def test_an_index_outside_the_value_set_counts_as_invalid(self):
        """The same rule as the two scorings (`is_usable_prediction`): "Area type" has two
        classes, so index 2 is not a class of it."""
        from vidlu_irap_gaim.vlm.scoring import count_scored_and_invalid_responses
        from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

        num_scored, num_invalid = count_scored_and_invalid_responses(
            {"Area type": AttributePrediction("Area type", "Rural", 2),
             "Land use - driver-side":
                 AttributePrediction("Land use - driver-side", "Industrial", 2)},
            self.ATTRS, SAMPLE_ATTR_METADATA)

        assert (num_scored, num_invalid) == (2, 1)

    def test_a_sample_with_no_responses_is_all_invalid(self):
        """This is what makes the invalid rate account for missing segments."""
        from vidlu_irap_gaim.vlm.scoring import count_scored_and_invalid_responses

        assert count_scored_and_invalid_responses({}, self.ATTRS, SAMPLE_ATTR_METADATA) == (2, 2)


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


def _all_schemes():
    """One instance of every registered scheme, over the shared metadata."""
    attr_to_default_class_idx = {attr: 0 for attr in SAMPLE_ATTR_METADATA}
    return {
        name: make_response_scheme(
            name, SAMPLE_ATTR_METADATA,
            attr_to_default_class_idx=(attr_to_default_class_idx
                                       if name in SPARSE_SCHEME_NAMES else None))
        for name in registry
    }


class TestSingleAttributeSessions:
    """Prompting for exactly one attribute – the per-attribute evaluation arm.

    Every scheme must stay usable at a session size of one: the parser is
    unchanged, so what has to hold is that the *instructions* stop describing a
    list of attributes the prompt does not contain.
    """

    ATTR = "Land use - driver-side"
    # Class index per attribute in full attribute order; Land use = 1 (Commercial).
    TARGET = torch.tensor([0, 1, 3])

    @pytest.mark.parametrize("name", sorted(registry))
    def test_round_trip_for_one_attribute(self, name):
        """format_ground_truth → parse_response recovers the class index."""
        scheme = _all_schemes()[name]
        ground_truth = scheme.format_ground_truth(self.TARGET, [self.ATTR])
        predictions = scheme.parse_response(ground_truth, [self.ATTR])
        assert predictions[self.ATTR].pred_idx == 1

    @pytest.mark.parametrize("name", sorted(registry))
    def test_prompt_covers_only_the_requested_attribute(self, name):
        scheme = _all_schemes()[name]
        prompt = scheme.build_prompt([self.ATTR], detail_level="attr_vals")
        assert self.ATTR in prompt
        assert "Area type" not in prompt

    @pytest.mark.parametrize("name", sorted(registry))
    def test_instructions_do_not_cite_absent_attribute_numbers(self, name):
        """The multi-attribute examples refer to attributes 2, 3, 5 – numbers a
        one-attribute prompt has no entry for. Showing them provokes format errors
        that say nothing about the attribute, biasing the per-attribute arm."""
        scheme = _all_schemes()[name]
        instructions = scheme._get_response_instructions([self.ATTR])
        for absent in ("2:", "3:", "5:"):
            assert absent not in instructions

    def test_instructions_still_describe_a_list_for_many_attributes(self):
        """The single-attribute wording must not leak into the normal arm, which
        is what the comparison is against."""
        scheme = _all_schemes()["standard"]
        attrs = list(SAMPLE_ATTR_METADATA)
        assert "one per attribute" in scheme._get_response_instructions(attrs)
        assert "exactly ONE line" in scheme._get_response_instructions([self.ATTR])

    def test_response_budget_shrinks_with_the_session(self):
        """The bound is derived from the format and the closed value set, so a
        one-attribute session must not be given a 41-attribute budget."""
        class _WordTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

        scheme = _all_schemes()["standard"]
        tokenizer = _WordTokenizer()
        one = scheme.compute_max_response_tokens(tokenizer, [self.ATTR], margin_tokens=0)
        many = scheme.compute_max_response_tokens(
            tokenizer, list(SAMPLE_ATTR_METADATA), margin_tokens=0)
        assert one < many

    def test_margin_tokens_are_added_on_top_of_the_derived_bound(self):
        """The bound catches gross non-compliance; it must not be a tight leash,
        since a budget that binds in one arm manufactures a difference."""
        class _WordTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

        scheme = _all_schemes()["standard"]
        tokenizer = _WordTokenizer()
        tight = scheme.compute_max_response_tokens(tokenizer, [self.ATTR], margin_tokens=0)
        loose = scheme.compute_max_response_tokens(tokenizer, [self.ATTR], margin_tokens=32)
        assert loose == tight + 32


class TestAttributeSessions:
    """Splitting the attributes into sessions."""

    def test_none_puts_every_attribute_in_one_session(self):
        from vidlu_irap_gaim.vlm.models.base import attribute_sessions

        attrs = list(SAMPLE_ATTR_METADATA)
        assert attribute_sessions(attrs, None) == [attrs]

    def test_one_gives_every_attribute_its_own_session(self):
        from vidlu_irap_gaim.vlm.models.base import attribute_sessions

        attrs = list(SAMPLE_ATTR_METADATA)
        assert attribute_sessions(attrs, 1) == [[a] for a in attrs]

    def test_groups_cover_the_attributes_in_order(self):
        from vidlu_irap_gaim.vlm.models.base import attribute_sessions

        attrs = list(SAMPLE_ATTR_METADATA)
        sessions = attribute_sessions(attrs, 2)
        assert [a for s in sessions for a in s] == attrs
        assert all(len(s) <= 2 for s in sessions)

    def test_zero_is_rejected(self):
        """Silently treating it as "all" would make an obvious typo invisible."""
        from vidlu_irap_gaim.vlm.models.base import attribute_sessions

        with pytest.raises(ValueError, match="at least 1"):
            attribute_sessions(list(SAMPLE_ATTR_METADATA), 0)


class TestPredictorResponseScheme:
    """A predictor must use the dataset's convention, not a default of its own.

    Prompt building, ground-truth formatting and parsing are one convention. A
    predictor left to construct its own `standard` default would prompt in one
    format while a `sparse_indexed` dataset scored in another – silently, since
    the parser would simply fail to match and every attribute would fall back.
    """

    def _predictor(self, **kwargs):
        from vidlu_irap_gaim.vlm.models.qwen3_vl import Qwen3VLPredictor

        return Qwen3VLPredictor(model_id="stub/model", **kwargs)

    def _scheme(self, name):
        return make_response_scheme(
            name, SAMPLE_ATTR_METADATA,
            attr_to_default_class_idx=({attr: 0 for attr in SAMPLE_ATTR_METADATA}
                                       if name in SPARSE_SCHEME_NAMES else None))

    def test_a_predictor_without_one_adopts_the_dataset_scheme(self):
        predictor = self._predictor()
        assert predictor.response_scheme is None

        scheme = self._scheme("sparse_indexed")
        predictor.response_scheme = scheme

        assert predictor.response_scheme is scheme

    def test_reassigning_the_same_scheme_is_accepted(self):
        """Reusing one predictor across splits must keep working; the splits share
        a single scheme instance."""
        scheme = self._scheme("standard")
        predictor = self._predictor(response_scheme=scheme)

        predictor.response_scheme = scheme

        assert predictor.response_scheme is scheme

    def test_a_conflicting_scheme_raises(self):
        predictor = self._predictor(response_scheme=self._scheme("standard"))

        with pytest.raises(ValueError, match="share one convention"):
            predictor.response_scheme = self._scheme("sparse_indexed")


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
    """Tests that VLMIrapDataset stores scheme metadata accessible to eval steps."""

    def test_dataset_info_contains_response_scheme(self):
        """VLMIrapDataset must expose vlm_response_scheme on info."""
        from unittest.mock import MagicMock
        from vidlu_irap_gaim.vlm.finetuning.dataset import VLMIrapDataset
        from vidlu_irap_gaim.vlm.response_scheme import StandardResponseScheme

        scheme = StandardResponseScheme(SAMPLE_ATTR_METADATA)

        # Minimal mock for base_dataset
        mock_ds = MagicMock()
        mock_ds.info = None
        mock_ds.__len__ = MagicMock(return_value=5)

        attrs = list(SAMPLE_ATTR_METADATA.keys())
        dataset = VLMIrapDataset(mock_ds, scheme, attrs)

        assert hasattr(dataset.info, "vlm_response_scheme")
        assert dataset.info.vlm_response_scheme is scheme
        assert dataset.response_scheme_name == "standard"

    def test_dataset_info_propagates_base_info_fields(self):
        """Augmented info must still contain original base info fields."""
        from unittest.mock import MagicMock
        from vidlu_irap_gaim.vlm.finetuning.dataset import VLMIrapDataset
        from vidlu_irap_gaim.vlm.response_scheme import StandardResponseScheme

        scheme = StandardResponseScheme(SAMPLE_ATTR_METADATA)
        base_info = dict(problem="multi_attribute_classification", some_field="hello")

        mock_ds = MagicMock()
        mock_ds.info = base_info
        mock_ds.__len__ = MagicMock(return_value=5)

        attrs = list(SAMPLE_ATTR_METADATA.keys())
        dataset = VLMIrapDataset(mock_ds, scheme, attrs)

        assert dataset.info.problem == "multi_attribute_classification"
        assert dataset.info.some_field == "hello"
        assert dataset.info.vlm_response_scheme is scheme

    def test_dataset_info_records_the_detail_level(self):
        """Published so evaluation cannot silently prompt at a different verbosity
        than the model was fine-tuned against."""
        from unittest.mock import MagicMock
        from vidlu_irap_gaim.vlm.finetuning.dataset import VLMIrapDataset
        from vidlu_irap_gaim.vlm.response_scheme import StandardResponseScheme

        mock_ds = MagicMock()
        mock_ds.info = None
        mock_ds.__len__ = MagicMock(return_value=5)

        dataset = VLMIrapDataset(mock_ds, StandardResponseScheme(SAMPLE_ATTR_METADATA),
                                 list(SAMPLE_ATTR_METADATA), detail_level="attr_vals")

        assert dataset.info.vlm_detail_level == "attr_vals"

    def test_wrapping_does_not_evaluate_the_base_info_lazy_entries(self):
        """Copying `info` with `dict(...)` reads every key through `__getitem__`, which
        evaluates each `Lazy` – so merely wrapping a dataset would compute `pixel_stats`
        and every other deferred entry."""
        from unittest.mock import MagicMock
        from irap_data import Lazy, LazyDict
        from vidlu_irap_gaim.vlm.finetuning.dataset import VLMIrapDataset
        from vidlu_irap_gaim.vlm.response_scheme import StandardResponseScheme

        num_evaluations = 0

        def expensive():
            nonlocal num_evaluations
            num_evaluations += 1
            return "computed"

        mock_ds = MagicMock()
        mock_ds.info = LazyDict(problem="multi_attribute_classification",
                                pixel_stats=Lazy(expensive))
        mock_ds.__len__ = MagicMock(return_value=5)

        dataset = VLMIrapDataset(mock_ds, StandardResponseScheme(SAMPLE_ATTR_METADATA),
                                 list(SAMPLE_ATTR_METADATA))

        assert num_evaluations == 0
        assert "<unevaluated>" in repr(dataset.info)
        # Still readable, and only then computed.
        assert dataset.info.pixel_stats == "computed"
        assert num_evaluations == 1


class TestVLMDatasetConfig:
    """Reading the prompt convention back off a dataset.

    One source of truth for prompt building, parsing and metrics: the eval step,
    the standalone evaluation tool and the metrics all read this rather than
    each deriving its own attribute list.
    """

    def _mock_dataset(self, **info):
        from unittest.mock import MagicMock
        from vidlu.utils.collections import NameDict

        mock_ds = MagicMock()
        # Datasets expose info attribute-style (vidlu wraps the dict), which is
        # how the config is read – a plain dict here would not reflect that.
        mock_ds.info = NameDict(**info) if info else None
        return mock_ds

    def _scheme(self):
        from vidlu_irap_gaim.vlm.response_scheme import SparseIndexedResponseScheme

        return SparseIndexedResponseScheme(
            SAMPLE_ATTR_METADATA, {attr: 0 for attr in SAMPLE_ATTR_METADATA})

    def test_config_is_read_from_dataset_info(self):
        from vidlu_irap_gaim.vlm.finetuning.dataset import vlm_config_from_data

        scheme = self._scheme()
        attrs = list(SAMPLE_ATTR_METADATA)
        mock_ds = self._mock_dataset(
            vlm_response_scheme=scheme, vlm_attrs_to_include=attrs,
            vlm_detail_level="attr_vals",
            attr_to_value_to_class_idx=SAMPLE_ATTR_METADATA)

        config = vlm_config_from_data({"train": mock_ds, "val": mock_ds})

        assert config.response_scheme is scheme
        assert config.attrs_to_include == attrs
        assert config.detail_level == "attr_vals"

    def test_the_first_split_that_carries_it_wins(self):
        """Only the VLM-wrapped splits carry it; a bare split must be skipped
        rather than aborting the lookup."""
        from vidlu_irap_gaim.vlm.finetuning.dataset import vlm_config_from_data

        scheme = self._scheme()
        configured = self._mock_dataset(
            vlm_response_scheme=scheme,
            vlm_attrs_to_include=list(SAMPLE_ATTR_METADATA),
            attr_to_value_to_class_idx=SAMPLE_ATTR_METADATA)

        config = vlm_config_from_data({"bare": self._mock_dataset(), "val": configured})

        assert config.response_scheme is scheme

    def test_missing_info_raises(self):
        """Naming the factory, since the fix is always to build the data with it."""
        from vidlu_irap_gaim.vlm.finetuning.dataset import (
            vlm_config_from_data, vlm_config_from_dataset)

        with pytest.raises(RuntimeError, match="make_vlm_bih_data"):
            vlm_config_from_data({"train": self._mock_dataset()})

        with pytest.raises(RuntimeError, match="vlm_response_scheme"):
            vlm_config_from_dataset(self._mock_dataset())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

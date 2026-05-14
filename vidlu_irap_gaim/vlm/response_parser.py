"""
Response parsing for VLM outputs.

Converts VLM text responses to structured predictions compatible with
the existing metrics system (tuple of (B, K) tensors).
"""

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal, Sequence

OutputFormat = Literal["json", "standard"]


def build_idx_to_value(
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
) -> dict[str, dict[int, str]]:
    """Build reverse mapping: attribute -> {class_idx -> value_string}."""
    return {
        attr: {idx: val for val, idx in val_to_idx.items()}
        for attr, val_to_idx in attr_to_value_to_class_idx.items()
    }


@dataclass
class AttributePrediction:
    """A single attribute prediction parsed from VLM output."""

    attr_name: str
    pred_value: str  # The string value predicted by VLM
    pred_idx: int  # Mapped class index (-1 if invalid/unknown)
    confidence: float = 1.0  # VLM doesn't provide probabilities, so always 1.0


def _fuzzy_match_value(
    pred_value: str,
    valid_values: dict[str, int],
    threshold: float = 0.6,
) -> tuple[str | None, int]:
    """Find the closest matching valid value using string similarity.

    Args:
        pred_value: The predicted value string from VLM.
        valid_values: Mapping of valid value strings to class indices.
        threshold: Minimum similarity ratio to accept a match.

    Returns:
        Tuple of (matched_value, class_idx) or (None, -1) if no match found.
    """
    if not pred_value or not valid_values:
        return None, -1

    pred_lower = pred_value.lower().strip()
    best_match = None
    best_ratio = 0.0
    best_idx = -1

    for valid_val, idx in valid_values.items():
        valid_lower = valid_val.lower().strip()

        # Exact match (case-insensitive)
        if pred_lower == valid_lower:
            return valid_val, idx

        # Substring match
        if pred_lower in valid_lower or valid_lower in pred_lower:
            ratio = 0.9  # High but not perfect
        else:
            ratio = SequenceMatcher(None, pred_lower, valid_lower).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_match = valid_val
            best_idx = idx

    if best_ratio >= threshold:
        return best_match, best_idx

    return None, -1


def _extract_json_from_response(response_text: str) -> dict:
    """Extract JSON object from VLM response text.

    Handles various formats:
    - Raw JSON: {"key": "value", ...}
    - Markdown code block: ```json ... ```
    - Mixed text with embedded JSON
    """
    # Try to find ```json ... ``` block first
    json_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text, re.IGNORECASE)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object (greedy match for nested braces)
    # Use a more robust pattern that handles nested objects
    brace_count = 0
    start_idx = None
    for i, char in enumerate(response_text):
        if char == "{":
            if start_idx is None:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    return json.loads(response_text[start_idx : i + 1])
                except json.JSONDecodeError:
                    start_idx = None  # Reset and try to find another

    return {}


def _normalize_for_match(s: str) -> str:
    """Normalize free-form text for robust matching.

    Used for matching (possibly truncated) attribute names and values.
    """
    s = str(s)
    s = s.strip().strip("\"'")
    s = s.lower()
    # Keep alphanumerics; collapse everything else to spaces.
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_pred_value(v: object) -> str:
    """Normalize a raw predicted value (possibly malformed/truncated) to text."""
    if v is None:
        return "None"
    if isinstance(v, (int, float, bool)):
        return str(v)
    s = str(v).strip()
    # Many broken outputs look like:  "None,   (missing closing quote)
    s = s.rstrip(",")
    s = s.strip().strip("\"'")
    s = s.strip()
    return s


def _match_attr_name(pred_key: str, attrs_to_include: Sequence[str]) -> str | None:
    """Map a predicted key to the closest requested attribute name.

    Returns:
        Canonical attribute name from attrs_to_include, or None if no safe match.
    """
    pred_norm = _normalize_for_match(pred_key)
    if not pred_norm:
        return None

    # 1) Exact match after normalization
    norm_to_attr = {_normalize_for_match(a): a for a in attrs_to_include}
    exact = norm_to_attr.get(pred_norm)
    if exact is not None:
        return exact

    # 2) Unique prefix match (common for truncated keys like "Bicycle observed")
    prefix_matches = [a for a in attrs_to_include if _normalize_for_match(a).startswith(pred_norm)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    # 3) Similarity match (guarded by threshold and uniqueness)
    best_attr = None
    best_ratio = 0.0
    second_best_ratio = 0.0
    for a in attrs_to_include:
        a_norm = _normalize_for_match(a)
        ratio = SequenceMatcher(None, pred_norm, a_norm).ratio()
        if ratio > best_ratio:
            second_best_ratio = best_ratio
            best_ratio = ratio
            best_attr = a
        elif ratio > second_best_ratio:
            second_best_ratio = ratio

    # Require a strong match and separation from the runner-up.
    if best_attr is not None and best_ratio >= 0.82 and (best_ratio - second_best_ratio) >= 0.05:
        return best_attr

    return None


def _parse_any_kv_lines(response_text: str) -> dict[str, str]:
    """Parse any key/value pairs from line-based / JSON-like output.

    This is intentionally permissive and does NOT require keys to match known
    attributes. It allows recovering from slightly-wrong keys and malformed
    JSON (e.g. truncated quotes).
    """
    result: dict[str, str] = {}
    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line or line in ("{", "}", "```", "```json"):
            continue

        # Prefer JSON-like quoted keys first (avoid splitting inside the key).
        # Examples:
        #   "Land use - driver-side": "Commercial",
        #   'Land use - driver-side': 'Commercial',
        m = re.match(r'^(?:\d+\.\s*)?"(?P<k>[^"]+)"\s*:\s*(?P<v>.+?)\s*$', line)
        if not m:
            m = re.match(r"^(?:\d+\.\s*)?'(?P<k>[^']+)'\s*:\s*(?P<v>.+?)\s*$", line)
        if not m:
            # Unquoted key/value lines:
            #   Area type: Urban
            #   1. Area type - Urban
            m = re.match(
                r"^(?:\d+\.\s*)?(?P<k>[^:\-]+?)\s*[:\-]\s*(?P<v>.+?)\s*$",
                line,
            )
        if not m:
            continue

        k = m.group("k").strip().strip("\"'")
        v = m.group("v").strip()
        # Strip a trailing comma, then surrounding quotes (even if only one side exists).
        v = v.rstrip(",").strip().strip("\"'")

        if k and k not in result:
            result[k] = v

    return result


def _parse_line_format(
    response_text: str,
    attrs_to_include: Sequence[str],
) -> dict[str, str]:
    """Fallback parser for line-based format like 'Attribute: Value'.

    Handles formats like:
    - "Area type: Urban"
    - "1. Area type: Urban"
    - "Area type - Urban"
    """
    result = {}
    lines = response_text.split("\n")

    for attr_name in attrs_to_include:
        # Create pattern to match this attribute
        # Escape special regex characters in attribute name
        escaped_name = re.escape(attr_name)
        # Support both plain text lines and JSON-like lines:
        #   Area type: Urban
        #   "Area type": "Urban",
        #   1. Area type - Urban
        pattern = (
            rf"(?:^\d+\.\s*)?"
            rf"(?:['\"])?{escaped_name}(?:['\"])?"
            rf"\s*[:\-]\s*(.+?)$"
        )

        for line in lines:
            match = re.search(pattern, line.strip(), re.IGNORECASE)
            if match:
                v = match.group(1).strip()
                # Remove trailing commas common in JSON-like outputs and surrounding quotes.
                v = v.rstrip(",").strip().strip("\"'")
                result[attr_name] = v
                break

    return result


def _parse_standard_format(
    response_text: str,
    attrs_to_include: Sequence[str],
) -> dict[str, str]:
    """Parse standard numbered line format: '1: Value'.

    Expected format (one line per attribute):
        1: None
        2: Urban
        3: Present

    Args:
        response_text: Raw VLM response text.
        attrs_to_include: Ordered list of attribute names (index maps to number).

    Returns:
        Dictionary mapping attribute names to predicted values.
    """
    result: dict[str, str] = {}

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Match patterns like "1: None", "1 : None", "1:None"
        match = re.match(r"^(\d+)\s*:\s*(.+?)\s*$", line)
        if match:
            attr_idx = int(match.group(1)) - 1  # Convert 1-based to 0-based
            value = match.group(2).strip()

            if 0 <= attr_idx < len(attrs_to_include):
                attr_name = attrs_to_include[attr_idx]
                result[attr_name] = value

    return result


def parse_indexed_response(
    response_text: str,
    attrs_to_include: Sequence[str],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    attr_to_default_class_idx: dict[str, int],
) -> dict[str, AttributePrediction]:
    """Parse NUMBER: INDEX format, fill missing attributes with per-attr default.

    Expected format (one line per attribute):
        1: 0
        2: 3
        3: 1

    Args:
        response_text: Raw VLM response text.
        attrs_to_include: Ordered list of attribute names (index maps to number).
        attr_to_value_to_class_idx: Mapping of attribute name -> {value -> class_idx}.
        attr_to_default_class_idx: Mapping of attribute name -> default class index.

    Returns:
        Dictionary mapping attribute names to AttributePrediction objects.
    """
    results: dict[str, AttributePrediction] = {}
    attrs_list = list(attrs_to_include)

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^(\d+)\s*:\s*(\d+)\s*$", line)
        if match:
            attr_idx = int(match.group(1)) - 1
            value_idx = int(match.group(2))

            if 0 <= attr_idx < len(attrs_list):
                attr_name = attrs_list[attr_idx]
                value_to_idx = attr_to_value_to_class_idx.get(attr_name, {})
                valid_values = list(value_to_idx.keys())
                if not valid_values:
                    raise ValueError(f"Attribute {attr_name!r} has no valid values in metadata")

                if 0 <= value_idx < len(valid_values):
                    pred_value = valid_values[value_idx]
                    pred_idx = value_to_idx[pred_value]
                else:
                    pred_value = valid_values[0]  # placeholder for display; pred_idx=-1 marks invalid
                    pred_idx = -1

                results[attr_name] = AttributePrediction(
                    attr_name=attr_name,
                    pred_value=pred_value,
                    pred_idx=pred_idx,
                    confidence=1.0 if pred_idx >= 0 else 0.0,
                )

    # Fill missing attributes with per-attr default.
    idx_to_value = build_idx_to_value(attr_to_value_to_class_idx)
    for attr in attrs_to_include:
        if attr not in results:
            if attr not in attr_to_value_to_class_idx:
                raise ValueError(f"Attribute {attr!r} not found in attr_to_value_to_class_idx")
            default_idx = attr_to_default_class_idx[attr]
            default_val = idx_to_value[attr][default_idx]
            results[attr] = AttributePrediction(
                attr_name=attr,
                pred_value=default_val,
                pred_idx=default_idx,
            )

    return results


def all_default_predictions(
    attrs_to_include: Sequence[str],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    attr_to_default_class_idx: dict[str, int],
) -> dict[str, AttributePrediction]:
    """Return AttributePrediction with per-attr default class index for each attr."""
    idx_to_value = build_idx_to_value(attr_to_value_to_class_idx)
    results: dict[str, AttributePrediction] = {}
    for attr in attrs_to_include:
        if attr not in attr_to_value_to_class_idx:
            raise ValueError(f"Attribute {attr!r} not found in attr_to_value_to_class_idx")
        default_idx = attr_to_default_class_idx[attr]
        default_val = idx_to_value[attr][default_idx]
        results[attr] = AttributePrediction(
            attr_name=attr,
            pred_value=default_val,
            pred_idx=default_idx,
        )
    return results


def parse_sparse_response(
    response_text: str,
    attrs_to_include: Sequence[str],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    attr_to_default_class_idx: dict[str, int],
    *,
    use_indices: bool = False,
) -> dict[str, AttributePrediction]:
    """Parse sparse response format.

    Sparse format: only non-default attributes in response.
    Missing attributes are filled with per-attr default class index.

    Args:
        response_text: Raw VLM response text.
        attrs_to_include: Ordered attribute names that were requested.
        attr_to_value_to_class_idx: Mapping of attribute name -> {value -> class_idx}.
        attr_to_default_class_idx: Mapping of attribute name -> default class index.
        use_indices: If True, values are integer indices (NUMBER: INDEX).
            If False, values are strings (NUMBER: VALUE) with fuzzy matching.

    Returns:
        Dict mapping attribute name -> AttributePrediction.
    """
    if "all default" in response_text.lower():
        return all_default_predictions(
            attrs_to_include, attr_to_value_to_class_idx, attr_to_default_class_idx
        )

    if use_indices:
        return parse_indexed_response(
            response_text,
            attrs_to_include,
            attr_to_value_to_class_idx,
            attr_to_default_class_idx,
        )

    # Sparse standard: values as strings, fuzzy matching
    raw_parsed = _parse_standard_format(response_text, attrs_to_include)
    if not raw_parsed or len(raw_parsed) < 1:
        raw_parsed = _parse_any_kv_lines(response_text)

    idx_to_value = build_idx_to_value(attr_to_value_to_class_idx)
    results: dict[str, AttributePrediction] = {}
    for attr in attrs_to_include:
        if attr in raw_parsed:
            val_str = _clean_pred_value(raw_parsed[attr])
            matched_val, idx = _fuzzy_match_value(
                val_str, attr_to_value_to_class_idx[attr]
            )
            pred_value = matched_val if matched_val else val_str
            results[attr] = AttributePrediction(
                attr_name=attr, pred_value=pred_value, pred_idx=idx
            )
        else:
            default_idx = attr_to_default_class_idx[attr]
            default_val = idx_to_value[attr][default_idx]
            results[attr] = AttributePrediction(
                attr_name=attr, pred_value=default_val, pred_idx=default_idx
            )
    return results


def parse_vlm_response(
    response_text: str,
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    attrs_to_include: Sequence[str] | None = None,
    fuzzy_threshold: float = 0.6,
    output_format: OutputFormat = "standard",
) -> dict[str, AttributePrediction]:
    """Parse VLM response text into structured predictions.

    Args:
        response_text: Raw text output from the VLM.
        attr_to_value_to_class_idx: Mapping of attribute name -> {value -> class_idx}.
        attrs_to_include: Subset of attributes to parse. If None, uses all attributes.
        fuzzy_threshold: Minimum similarity for fuzzy value matching.
        output_format: Expected format - "json" (verbose) or "standard" (numbered lines).

    Returns:
        Dictionary mapping attribute names to AttributePrediction objects.
    """
    if attrs_to_include is None:
        attrs_to_include = list(attr_to_value_to_class_idx.keys())

    # Parse based on expected output format
    if output_format == "standard":
        # Standard format: parse numbered lines first
        parsed_dict = _parse_standard_format(response_text, list(attrs_to_include))

        # Fallback to other parsers if primary parsing was incomplete
        if len(parsed_dict) < len(attrs_to_include):
            # Try JSON as fallback (model may have ignored format instructions)
            json_parsed = _extract_json_from_response(response_text)
            for k, v in json_parsed.items():
                if k not in parsed_dict:
                    parsed_dict[k] = v

            # Try line-based parsing as last resort
            any_kv = _parse_any_kv_lines(response_text)
            for k, v in any_kv.items():
                if k not in parsed_dict:
                    parsed_dict[k] = v
    else:
        # JSON format: try JSON extraction first
        parsed_dict = _extract_json_from_response(response_text)

        # Fallback to line-based parsing if JSON failed or is incomplete
        if not parsed_dict or len(parsed_dict) < len(attrs_to_include):
            # First, parse any key/value-like lines (doesn't require exact attribute names).
            any_kv = _parse_any_kv_lines(response_text)
            for k, v in any_kv.items():
                if k not in parsed_dict:
                    parsed_dict[k] = v

            # Then, as a last resort, try strict per-attribute extraction.
            line_parsed = _parse_line_format(response_text, attrs_to_include)
            # Merge, preferring JSON results
            for k, v in line_parsed.items():
                if k not in parsed_dict:
                    parsed_dict[k] = v

    # Remap slightly-wrong/truncated keys onto canonical requested attribute names.
    # (Important for JSON or fallback line-based parsing where keys are strings)
    if parsed_dict:
        candidates: dict[str, list[tuple[str, object]]] = {}
        for k, v in list(parsed_dict.items()):
            if k in attrs_to_include:
                target = k
            else:
                target = _match_attr_name(k, attrs_to_include)
                if target is None:
                    continue  # ignore unknown/unmappable keys
            candidates.setdefault(target, []).append((k, v))

        remapped: dict[str, object] = {}
        for target, items in candidates.items():
            if len(items) == 1:
                remapped[target] = items[0][1]
                continue

            # Prefer an exact canonical key if present.
            exact = [(k, v) for (k, v) in items if k == target]
            if exact:
                # If multiple exact keys disagree, fail fast.
                unique_vals = {json.dumps(v, sort_keys=True) for (_, v) in exact}
                if len(unique_vals) > 1:
                    raise ValueError(
                        f"Ambiguous VLM output: multiple identical keys for {target!r} disagree: "
                        f"{[v for (_, v) in exact]!r}."
                    )
                remapped[target] = exact[0][1]
                continue

            # Otherwise select the closest key by similarity.
            target_norm = _normalize_for_match(target)
            best_ratio = -1.0
            for k, v in items:
                ratio = SequenceMatcher(None, _normalize_for_match(k), target_norm).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_v = v

            # If there are multiple candidates with different values, keep the best match and drop others.
            remapped[target] = best_v

        parsed_dict = remapped

    # Map parsed values to predictions
    predictions = {}
    for attr_name in attrs_to_include:
        value_to_idx = attr_to_value_to_class_idx.get(attr_name, {})
        pred_value = _clean_pred_value(parsed_dict.get(attr_name, ""))

        # Try exact match first
        if pred_value in value_to_idx:
            pred_idx = value_to_idx[pred_value]
            matched_value = pred_value
        else:
            # Try fuzzy matching
            matched_value, pred_idx = _fuzzy_match_value(pred_value, value_to_idx, fuzzy_threshold)
            if matched_value is None:
                matched_value = pred_value  # Keep original for debugging

        predictions[attr_name] = AttributePrediction(
            attr_name=attr_name,
            pred_value=matched_value,
            pred_idx=pred_idx,
            confidence=1.0 if pred_idx >= 0 else 0.0,
        )

    return predictions

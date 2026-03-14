"""
Prompt generation for VLM-based road attribute classification.

Supports multiple detail levels and external YAML configuration
for easy prompt iteration without code changes.
"""

from pathlib import Path
from typing import Callable, Literal, Sequence

import yaml


DetailLevel = Literal["attr_desc_vals", "attr_vals", "attr", "none"]
DEFAULT_DETAIL_LEVEL: DetailLevel = "attr_desc_vals"


class PromptBuilder:
    """Flexible prompt builder supporting multiple verbosity levels.

    Detail levels:
        - attr_desc_vals: Attribute description + valid values + default
        - attr_vals: Attribute name + valid values + default (no descriptions)
        - attr: Attribute names only (no values; model learns from data)
        - none: Empty preamble and no attribute list

    Example usage:
        builder = PromptBuilder(attr_to_value_to_class_idx)
        preamble, attr_sections = builder.build_attribute_sections(attrs, detail_level=DEFAULT_DETAIL_LEVEL)

        # Or load from YAML config
        builder = PromptBuilder.from_yaml("attribute_prompts.yaml", attr_to_value_to_class_idx)
        preamble, attr_sections = builder.build_attribute_sections(attrs, detail_level="attr_vals")
    """

    DEFAULT_PREAMBLE = """\
You are an expert road safety analyst trained on iRAP (International Road Assessment Programme) coding standards.

Analyze the road image and classify each attribute by selecting exactly ONE value from the valid options listed.
Be precise and only use the exact values provided."""

    def __init__(
        self,
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
        attribute_descriptions: dict[str, str] | None = None,
        value_descriptions: dict[str, dict[str, str]] | None = None,
        attribute_tips: dict[str, list[str]] | None = None,
        preamble: str | None = None,
    ):
        """Initialize the prompt builder.

        Args:
            attr_to_value_to_class_idx: Mapping of attribute name -> {value -> class_idx}.
            attribute_descriptions: Optional descriptions for each attribute.
            value_descriptions: Optional per-value descriptions for each attribute.
            attribute_tips: Optional tips for classifying each attribute.
            preamble: Custom preamble text (overrides default).
        """
        self.attr_to_values = {
            attr: list(val_to_idx.keys())
            for attr, val_to_idx in attr_to_value_to_class_idx.items()
        }
        self.attr_descriptions = attribute_descriptions or {}
        self.value_descriptions = value_descriptions or {}
        self.attr_tips = attribute_tips or {}
        self.preamble = preamble or self.DEFAULT_PREAMBLE

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
    ) -> "PromptBuilder":
        """Load prompt configuration from a YAML file.

        Expected YAML structure:
            preamble: |
              Custom preamble text...

            attributes:
              "Attribute Name":
                description: "What this attribute means"
                values:
                  "Value1": "Description of value1"
                  "Value2": "Description of value2"
                tips:
                  - "Tip for classifying this attribute"
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompt config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        attr_descriptions = {}
        value_descriptions = {}
        attr_tips = {}

        if "attributes" in config:
            for attr_name, attr_config in config["attributes"].items():
                if isinstance(attr_config, dict):
                    if "description" in attr_config:
                        attr_descriptions[attr_name] = attr_config["description"]
                    if "values" in attr_config:
                        value_descriptions[attr_name] = attr_config["values"]
                    if "tips" in attr_config:
                        attr_tips[attr_name] = attr_config["tips"]

        return cls(
            attr_to_value_to_class_idx=attr_to_value_to_class_idx,
            attribute_descriptions=attr_descriptions,
            value_descriptions=value_descriptions,
            attribute_tips=attr_tips,
            preamble=config.get("preamble"),
        )

    def build_attribute_sections(
        self,
        attrs_to_include: Sequence[str],
        detail_level: DetailLevel = DEFAULT_DETAIL_LEVEL,
        value_formatter: Callable[[list[str]], str] | None = None,
        attr_to_default_value: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Return format-agnostic prompt parts for composition by ResponseScheme subclasses.

        Args:
            attrs_to_include: Ordered attribute names to include.
            detail_level: One of "attr_desc_vals", "attr_vals", "attr", "none".
            value_formatter: Optional callable to format valid values per attribute.
                Default: comma-separated string. Ignored for "attr" and "none".
            attr_to_default_value: Optional mapping of attribute name -> default value
                string. When provided, adds a "Default: ..." line per attribute.
                Ignored for "attr" and "none".

        Returns:
            Tuple of (preamble, joined_attribute_sections).
        """
        if detail_level == "none":
            return ("", "")

        preamble = self.preamble
        attr_sections = []
        for i, attr_name in enumerate(attrs_to_include, 1):
            default_value = (
                attr_to_default_value.get(attr_name)
                if attr_to_default_value and detail_level not in ("attr", "none") else None
            )
            section = self._format_attribute(
                attr_name,
                i,
                detail_level,
                value_formatter=value_formatter,
                default_value=default_value,
            )
            if section:
                attr_sections.append(section)
        return preamble, "\n".join(attr_sections)

    def _format_attribute(
        self,
        attr_name: str,
        index: int,
        detail_level: DetailLevel,
        value_formatter: Callable[[list[str]], str] | None = None,
        default_value: str | None = None,
    ) -> str:
        """Format a single attribute section based on detail level."""
        if detail_level == "attr":
            return f"{index}. {attr_name}"

        valid_values = self.attr_to_values.get(attr_name, [])
        if not valid_values:
            return ""

        lines = [f"{index}. {attr_name}"]

        # Add attribute description for attr_desc_vals
        if detail_level == "attr_desc_vals":
            desc = self.attr_descriptions.get(attr_name)
            if desc:
                desc = " ".join(desc.split())
                lines.append(f"   Description: {desc}")

        # Add valid values
        if value_formatter is not None:
            values_str = value_formatter(valid_values)
        else:
            values_str = ", ".join(valid_values)
        lines.append(f"   Valid values: {values_str}")

        if default_value is not None:
            lines.append(f"   Default: {default_value}")

        lines.append("")  # Blank line separator
        return "\n".join(lines)

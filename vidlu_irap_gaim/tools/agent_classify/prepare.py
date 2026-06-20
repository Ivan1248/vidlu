"""
Generate a task JSON and rendered agent prompt markdown for agent-based iRAP attribute
classification (e.g. via the Google Antigravity workspace agent).

Driven by the ``agent_classify prepare`` frontend (see ``__main__.py``); the public
entry point here is :func:`prepare_agent_tasks`. The generated files are consumed by
an AI agent that has read/write access to the local filesystem:
  - ``agent_tasks_<ds>_<split>.json``  – ordered list of segments to classify, with image paths.
  - ``agent_prompt_<ds>_<split>.md``   – self-contained instructions the agent receives as its prompt.

The agent then writes ``agent_predictions_<ds>_<split>.json``, which
``agent_classify evaluate`` scores. File names come from
``_common.agent_artifact_paths``.
"""

import json
from pathlib import Path


def _build_segment_entries(dataset, limit: int | None) -> list[dict]:
    """Build the segment list with absolute image paths and context image paths."""
    context_offsets = dataset.context_offsets  # e.g. [0, -1, -4]
    primary_offset_idx = context_offsets.index(0)

    entries = []
    segment_ids = dataset.segment_ids if limit is None else dataset.segment_ids[:limit]
    for sid in segment_ids:
        context_ids = dataset.segment_id_to_context_ids[sid]  # tuple, one per context offset
        primary_sid = context_ids[primary_offset_idx]
        primary_path = dataset.seg_to_paths.get(primary_sid, {}).get("rgb")

        context_paths = []
        for i, cid in enumerate(context_ids):
            if i == primary_offset_idx:
                continue
            p = dataset.seg_to_paths.get(cid, {}).get("rgb")
            if p is not None:
                context_paths.append(str(p))

        entries.append(
            {
                "segment_id": sid,
                "image_path": str(primary_path) if primary_path is not None else None,
                "context_image_paths": context_paths,
            }
        )
    return entries


def _render_prompt(
    dataset_name: str,
    split: str,
    attr_to_value_to_class_idx: dict,
    attrs_to_include: list[str],
    tasks_file_abs: str,
    predictions_file_abs: str,
    prompt_config_path: Path | None,
) -> str:
    """Render the self-contained agent prompt markdown."""
    from vidlu_irap_gaim.vlm.prompts import PromptBuilder

    yaml_path = prompt_config_path or (
        Path(__file__).parent.parent / "vlm" / "attribute_prompts.yaml"
    )
    if yaml_path.exists():
        builder = PromptBuilder.from_yaml(yaml_path, attr_to_value_to_class_idx)
    else:
        builder = PromptBuilder(attr_to_value_to_class_idx)

    preamble, attr_sections = builder.build_attribute_sections(
        attrs_to_include, detail_level="attr_desc_vals"
    )

    num_attrs = len(attrs_to_include)
    lines = [
        "# iRAP Road Attribute Classification Task",
        "",
        f"**Dataset:** {dataset_name.upper()}  **Split:** {split}  **Attributes:** {num_attrs}",
        "",
        "---",
        "",
        "## Role",
        "",
        preamble,
        "",
        "---",
        "",
        "## Task",
        "",
        f"Classify road attributes for every segment listed in `{tasks_file_abs}`.",
        "",
        "**Steps:**",
        "",
        f"1. Read `{tasks_file_abs}` to get the list of segments.",
        f"2. If `{predictions_file_abs}` already exists, load it so you can skip",
        f"   already-processed segments (resumability).",
        "3. For each segment (in order):",
        "   - Use `view_file` (or equivalent) to open `image_path`.",
        "   - Optionally view `context_image_paths` for temporal context",
        "     (earlier frames from the same road).",
        "   - Classify **every attribute** listed in the *Attributes* section below.",
        "   - Record your predictions.",
        f"4. After processing all segments (or after a reasonable batch), write results to",
        f"   `{predictions_file_abs}` using the output format below.",
        "5. If an image file is missing, record `\"image_not_found\": true` instead of",
        "   attribute predictions for that segment.",
        "",
        "---",
        "",
        "## Output Format",
        "",
        f"Write a single JSON file at `{predictions_file_abs}` with this structure:",
        "",
        "```json",
        "{",
        '  "segment_id_1": {',
        '    "Attribute Name 1": "Value A",',
        '    "Attribute Name 2": "Value B"',
        "  },",
        '  "segment_id_2": {',
        '    "Attribute Name 1": "Value C",',
        '    "image_not_found": true',
        "  }",
        "}",
        "```",
        "",
        "Rules:",
        "- Use **exactly** the attribute names and valid values shown below.",
        "- Write `null` for an attribute you cannot determine.",
        "- Do not add extra keys or nesting.",
        "- Preserve existing entries when resuming (only append new segment IDs).",
        "",
        "---",
        "",
        "## Attributes",
        "",
        attr_sections,
    ]
    return "\n".join(lines)


def prepare_agent_tasks(
    dataset_name: str = "bih",
    split: str = "test",
    output_dir: str | Path = "agent_tasks",
    limit: int | None = None,
    prompt_config: str | None = None,
) -> dict:
    """Generate ``agent_tasks.json`` and ``agent_prompt_rendered.md``.

    Args:
        dataset_name: ``"bih"`` or ``"vietnam"``.
        split: Dataset split (``"train"``, ``"val"``, ``"test"``).
        output_dir: Directory to write the generated files.
        limit: Maximum number of segments to include (default: all).
        prompt_config: Path to a custom ``attribute_prompts.yaml``.

    Returns:
        Dict with keys ``tasks_file`` and ``prompt_file`` (Path objects).
    """
    from ._common import agent_artifact_paths, load_split_and_attrs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = agent_artifact_paths(output_dir, dataset_name, split)

    print(f"Loading {dataset_name.upper()} dataset (split={split})...")
    dataset, attr_to_value_to_class_idx, attrs_to_include = load_split_and_attrs(
        dataset_name, split
    )
    print(f"  {len(dataset)} segments available.")

    print("Building segment list with image paths...")
    segments = _build_segment_entries(dataset, limit)
    effective_count = len(segments)
    print(f"  {effective_count} segments included.")

    predictions_abs = str(paths["predictions"].resolve())

    task_data = {
        "dataset": dataset_name,
        "split": split,
        "output_file": predictions_abs,
        "num_segments": effective_count,
        "attributes": attrs_to_include,
        "attribute_values": {
            attr: list(val_to_idx.keys())
            for attr, val_to_idx in attr_to_value_to_class_idx.items()
            if attr in set(attrs_to_include)
        },
        "segments": segments,
    }

    tasks_file = paths["tasks"]
    with open(tasks_file, "w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)
    print(f"Saved task list → {tasks_file}")

    print("Rendering agent prompt...")
    prompt_md = _render_prompt(
        dataset_name=dataset_name,
        split=split,
        attr_to_value_to_class_idx=attr_to_value_to_class_idx,
        attrs_to_include=attrs_to_include,
        tasks_file_abs=str(tasks_file.resolve()),
        predictions_file_abs=predictions_abs,
        prompt_config_path=Path(prompt_config) if prompt_config else None,
    )

    prompt_file = paths["prompt"]
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt_md)
    print(f"Saved agent prompt → {prompt_file}")

    return {"tasks_file": tasks_file, "prompt_file": prompt_file}

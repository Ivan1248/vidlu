"""Streamlit-based IRAP dataset viewer.

Requires the package to be installed (``pip install -e .[viewer]`` from the
``irap-data`` directory).

Run with (Streamlit >= 1.41)::

    streamlit run -m irap_data.dataset_viewer

or, on any Streamlit version (the file uses absolute imports, so it works as a
plain script as long as ``irap_data`` is importable)::

    streamlit run <path-to>/irap_data/dataset_viewer.py
"""

import dataclasses as dc
import os
import random
from pathlib import Path
import typing as T

import streamlit as st

from irap_data import make_bih_data, make_vietnam_data
from irap_data.irap_dataset import IGNORE_LABEL_INDEX
from irap_data.jitter import make_sequence_color_jitter, JITTER_STANDARD, JITTER_STRONG
from irap_data.vis_utils import (
    AttributeMetadataDecoder,
    create_composite_view_strip,
    get_index_color,
    tensor_image_to_uint8_np,
)


# Constants ####################################################################

DATASET_NAMES = ("IRAP_BIH", "IRAP_Vietnam")

_DATASETS_DIR_HELP = (
    "Parent directory containing one or more IRAP datasets.\n\n"
    "Expected layout:\n"
    "- IRAP-BiH:   <datasets_dir>/IRAP_BIH/ (data)\n"
    "              <datasets_dir>/IRAP_BIH_METADATA/ (metadata)\n"
    "- IRAP-Vietnam: <datasets_dir>/IRAP_Vietnam/ (data + metadata together)\n\n"
    "Default is read from $IRAP_HOME, then $DATASETS_DIR."
)

_JITTER_OPTIONS = ["None", "Standard", "Strong (semi-supervised)"]

_VIEWER_CSS = """
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    div[data-testid="stImage"] {
        border-radius: 0px !important;
        display: flex;
        justify-content: center;
    }
    div[data-testid="stImage"] img {
        border-radius: 0px !important;
        max-height: 90vh !important;
        width: 100% !important;
        max-width: 100% !important;
        object-fit: contain;
    }
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    body, [data-testid="stAppViewContainer"] * {
        opacity: 1 !important;
        filter: none !important;
        animation: none !important;
        transition: none !important;
    }
    .attributes-text {
        line-height: 1.0;
        font-size: 70%;
        display: flex;
        align-items: center;
        margin-bottom: 2px;
    }
    .seq-square {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin-right: 1px;
        border-radius: 1px;
        opacity: 0.8;
    }
    .seq-container {
        display: inline-flex;
        align-items: center;
        margin: 0 4px;
    }
    .attr-key {
        width: 200px;
        min-width: 140px;
        display: inline-block;
        text-align: right;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
"""


# Configuration ################################################################


@dc.dataclass
class ViewerConfig:
    """User-facing sidebar settings (pending or active)."""

    dataset_name: str
    ds_dir: Path
    md_dir: Path
    split: str
    context_offsets: tuple[int, ...]

    @property
    def load_key(self) -> tuple:
        """Hashable key for dataset loading (excludes display-only settings like jitter)."""
        return (self.dataset_name, str(self.ds_dir), str(self.md_dir), self.split, self.context_offsets)


# Dataset detection and loading ################################################


def _detect_datasets(datasets_dir: Path) -> list[str]:
    return [name for name in DATASET_NAMES if (datasets_dir / name).is_dir()]


def _resolve_paths(datasets_dir: Path, dataset_name: str) -> tuple[Path, Path]:
    """Return (dataset_dir, metadata_dir) for the chosen dataset."""
    ds_dir = datasets_dir / dataset_name
    if dataset_name == "IRAP_BIH":
        # BiH ships metadata as a sibling directory.
        md_dir = datasets_dir / "IRAP_BIH_METADATA"
    else:
        # Vietnam stores metadata inside the dataset root.
        md_dir = ds_dir
    return ds_dir, md_dir


@st.cache_data(max_entries=256)
def load_example(_ds, idx: int, cache_key: tuple) -> dict:
    """Cache `ds[idx]` so Prev/Next revisits don't re-decode PNGs.

    `cache_key` is the active load key; it invalidates the cache when the
    user switches dataset/split/context. The `_ds` underscore prefix tells
    Streamlit to skip hashing the IRAPDataset (already identified by cache_key).
    """
    return dict(_ds[idx])  # Convert to dict in case of lazy fields


@st.cache_resource(hash_funcs={Path: str})
def load_dataset(dataset_name: str, dataset_dir: Path, metadata_dir: Path, split: str, context_offsets: tuple):
    if dataset_name == "IRAP_Vietnam":
        splits = make_vietnam_data(
            dataset_dir=dataset_dir,
            metadata_dir=metadata_dir,
            context_offsets=context_offsets,
        )
    else:
        splits = make_bih_data(
            dataset_dir=dataset_dir,
            metadata_dir=metadata_dir,
            context_offsets=context_offsets,
        )
    return splits[split]


# Filtering ####################################################################


def compute_filtered_indices(
    ds,
    decoder: AttributeMetadataDecoder,
    filters: dict[str, set[str]],
) -> list[int]:
    """Indices in `ds` whose label satisfies the per-attribute allow-sets.

    Empty allow-set for an attribute = no constraint on that attribute.
    """
    if not any(filters.values()):
        return list(range(len(ds)))

    val_to_idx = decoder.attr_to_value_to_class_idx
    allowed_idx_by_attr = {a: {val_to_idx[a][v] for v in vs} for a, vs in filters.items() if vs}
    attr_pos = {a: i for i, a in enumerate(val_to_idx.keys())}

    out = []
    for i, sid in enumerate(ds.segment_ids):
        lbls = ds.segment_id_to_labels[sid]
        if all(lbls[attr_pos[a]] in allowed for a, allowed in allowed_idx_by_attr.items()):
            out.append(i)
    return out


def _filters_key(filters: dict[str, set[str]]) -> frozenset:
    return frozenset((a, frozenset(vs)) for a, vs in filters.items() if vs)


# Neighbor sequence display ####################################################


def get_context_rgb_relpaths(ds, center_sid: str) -> list[tuple[int, str | None]]:
    """List of (offset, relpath-or-None) in context_offsets order."""
    ctx_ids = ds.segment_id_to_context_ids.get(center_sid)
    if ctx_ids is None:
        return []
    out = []
    for offset, sid in zip(ds.context_offsets, ctx_ids):
        p = ds.seg_to_paths.get(sid, {}).get("rgb")
        out.append((offset, p.relative_to(ds.root).as_posix() if p is not None else None))
    return out


def get_neighbor_labels(ds, ordered_attrs: list[str], center_sid: str, radius: int = 5) -> dict:
    """Map offset -> {attr: class_idx} for neighbors within +/-radius along the road.

    Uses the dataset's road-sequence index for correct neighbor resolution
    across multi-road datasets (e.g. Vietnam), instead of integer arithmetic
    on segment IDs.
    """
    if center_sid not in ds.seq_index:
        return {}

    road_id, pos = ds.seq_index[center_sid]
    seq = ds.road_to_seq[road_id]
    sid_to_labels = ds.segment_id_to_labels

    neighbors = {}
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        neighbor_pos = pos + offset
        if 0 <= neighbor_pos < len(seq):
            neighbor_sid = seq[neighbor_pos]
            if neighbor_sid in sid_to_labels:
                raw_labels = sid_to_labels[neighbor_sid]
                if len(raw_labels) == len(ordered_attrs):
                    neighbors[offset] = {a: raw_labels[i] for i, a in enumerate(ordered_attrs)}
    return neighbors


def generate_sequence_html(
    attr_values: dict[str, list[str]],
    neighbors: dict,
    attr_key: str,
    offsets: T.Iterable[int],
) -> str:
    squares = []
    for offset in offsets:
        if offset in neighbors and attr_key in neighbors[offset]:
            n_idx = neighbors[offset][attr_key]
            n_color = get_index_color(n_idx)
            sign = "+" if offset > 0 else ""
            tooltip = f"Offset {sign}{offset}: {n_idx}"
            if attr_key in attr_values and 0 <= n_idx < len(attr_values[attr_key]):
                tooltip = f"{attr_values[attr_key][n_idx]} ({n_idx})"
            squares.append(f'<div class="seq-square" style="background-color:{n_color};" title="{tooltip}"></div>')
        else:
            squares.append('<div class="seq-square" style="background-color:transparent;"></div>')
    return "".join(squares)


# Sidebar sections #############################################################


def _render_sidebar_config() -> ViewerConfig | None:
    """Render sidebar input widgets and return the pending configuration.

    Returns None if the inputs are invalid or incomplete (error already shown).
    """
    default_datasets_dir = os.environ.get("IRAP_HOME") or os.environ.get("DATASETS_DIR") or ""
    datasets_dir_input = st.sidebar.text_input(
        "Datasets directory",
        value=default_datasets_dir,
        help=_DATASETS_DIR_HELP,
    )
    if not datasets_dir_input:
        st.error("Set $IRAP_HOME or $DATASETS_DIR, or enter a path in the sidebar.")
        return None
    datasets_dir = Path(datasets_dir_input).expanduser()

    # Dataset selector — only datasets that exist under datasets_dir are offered.
    detected = _detect_datasets(datasets_dir)
    if not detected:
        st.error(f"No datasets found under {datasets_dir}. Looked for: {', '.join(DATASET_NAMES)}.")
        return None
    dataset_name = st.sidebar.selectbox("Dataset", detected, index=0)

    ds_dir, md_dir = _resolve_paths(datasets_dir, dataset_name)

    split = st.sidebar.selectbox("Split", ["train", "val", "test"])

    seq_input = st.sidebar.text_input("Context offsets (e.g., 0, -1, -4)", value="0, -1, -4")
    try:
        context_offsets = tuple(int(x.strip()) for x in seq_input.split(","))
    except ValueError:
        st.error("Invalid context offsets format. Use comma-separated integers.")
        return None

    return ViewerConfig(
        dataset_name=dataset_name,
        ds_dir=ds_dir,
        md_dir=md_dir,
        split=split,
        context_offsets=context_offsets,
    )


def _ensure_dataset_loaded(config: ViewerConfig):
    """Handle the deferred-load button and return (ds, active_config) or None.

    Uses session state to avoid reloading the dataset on every widget change.
    The user must click the button to trigger loading/reloading.
    """
    pending_load = config.load_key
    active_load = st.session_state.get("active_load")
    stale = active_load != pending_load

    if active_load is None:
        load_label = "Load dataset"
    elif stale:
        load_label = "Reload (settings changed)"
    else:
        load_label = None

    if load_label is not None:
        if st.sidebar.button(load_label, width="stretch", type="primary"):
            st.session_state.active_load = pending_load
            active_load = pending_load
            stale = False

    if active_load is None:
        st.info("Configure the sidebar settings and click **Load dataset** to begin.")
        return None

    # Reconstruct active config from the stored load key.
    a_name, a_ds_dir, a_md_dir, a_split, a_ctx_seq = active_load
    active_config = ViewerConfig(
        dataset_name=a_name,
        ds_dir=Path(a_ds_dir),
        md_dir=Path(a_md_dir),
        split=a_split,
        context_offsets=a_ctx_seq,
    )

    if stale:
        st.sidebar.warning("Settings changed — showing previously loaded data. Click Reload to apply.")

    with st.spinner(f"Loading {active_config.dataset_name} / {active_config.split}..."):
        ds = load_dataset(
            active_config.dataset_name,
            active_config.ds_dir,
            active_config.md_dir,
            active_config.split,
            active_config.context_offsets,
        )

    if not ds or len(ds) == 0:
        st.warning("Dataset is empty.")
        return None

    return ds, active_config


def _render_filters_and_navigation(
    ds,
    decoder: AttributeMetadataDecoder,
    load_key: tuple,
    ordered_attrs: list[str],
    attr_values: dict[str, list[str]],
) -> tuple[int, int, list[int]] | None:
    """Render filter widgets and navigation controls in the sidebar.

    Returns (cursor, ds_idx, filtered) or None if no examples match.
    """
    # --- Filtering UI ---
    if "filters" not in st.session_state:
        st.session_state.filters = {}

    # When the dataset (name + split) changes, reset filters and cursor.
    ds_key = load_key
    if st.session_state.get("ds_key") != ds_key:
        st.session_state.ds_key = ds_key
        st.session_state.filters = {a: [] for a in ordered_attrs}
        st.session_state.cursor = 0

    # Ensure filter dict matches current attributes.
    for a in ordered_attrs:
        st.session_state.filters.setdefault(a, [])

    with st.sidebar.expander("Filter by attributes", expanded=False):
        if st.button("Clear filters", width="stretch"):
            for a in ordered_attrs:
                st.session_state.filters[a] = []
        for a in ordered_attrs:
            st.session_state.filters[a] = st.multiselect(
                a,
                attr_values[a],
                default=st.session_state.filters.get(a, []),
                key=f"filter_{hash(load_key)}_{a}",
            )

    # Active filters dict[attr, set[str]]
    active_filters = {a: set(v) for a, v in st.session_state.filters.items() if v}

    # Cache filtered indices by filter signature.
    fkey = (ds_key, _filters_key(active_filters))
    cache = st.session_state.setdefault("_filtered_cache", {})
    if fkey not in cache:
        cache.clear()  # keep cache small — only the latest filter result
        cache[fkey] = compute_filtered_indices(ds, decoder, active_filters)
    filtered = cache[fkey]

    st.sidebar.write(f"Size: {len(ds)} segments, {len(filtered)} matching")

    if len(filtered) == 0:
        st.warning("No examples match the current filters.")
        return None

    num_filtered = len(filtered)

    # Normalize cursor: wrap at boundaries, convert negative indices to positive.
    if "cursor" not in st.session_state:
        st.session_state.cursor = 0
    else:
        st.session_state.cursor = st.session_state.cursor % num_filtered

    # Navigation
    if st.sidebar.button("Random", width="stretch"):
        st.session_state.cursor = random.randint(0, num_filtered - 1)

    cursor = st.sidebar.number_input(
        "Index (within filtered)",
        min_value=-num_filtered,
        max_value=num_filtered,
        step=1,
        key="cursor",
    )
    cursor = cursor % num_filtered
    ds_idx = filtered[cursor]
    st.sidebar.caption(f"Dataset index: {ds_idx}")

    return cursor, ds_idx, filtered


# Main content columns #########################################################


def _render_image_column(data: dict, ds, context_offsets: tuple[int, ...], jitter_selection: str) -> None:
    """Render the image column: composite view strip with optional jitter preview."""
    rgb = data.get("rgb")
    if rgb is None:
        return

    jitter_caption_suffix = ""
    if jitter_selection != "None":
        preset = JITTER_STANDARD if jitter_selection == "Standard" else JITTER_STRONG
        rgb = make_sequence_color_jitter(preset=preset)(dict(rgb=rgb))["rgb"]
        jitter_caption_suffix = f" [{jitter_selection.lower().split()[0]} jitter]"

    imgs_np = tensor_image_to_uint8_np(rgb)
    final_img = create_composite_view_strip(imgs_np)

    caption = f"Segment: {data.get('segment_id', 'Unknown')}{jitter_caption_suffix}"
    if len(imgs_np) > 1:
        caption += f" | Context offsets: {', '.join(map(str, context_offsets))}"

    st.image(final_img, caption=caption, width="stretch")

    paths = get_context_rgb_relpaths(ds, data.get("segment_id"))
    if paths:
        lines = [f"{('+' if off > 0 else '')}{off}: {rel if rel is not None else '(missing)'}" for off, rel in paths]
        st.markdown(
            "<div style='font-family:monospace;font-size:75%;line-height:1.2;"
            "white-space:nowrap;overflow-x:auto;'>" + "<br>".join(lines) + "</div>",
            unsafe_allow_html=True,
        )


def _render_attribute_column(
    data: dict, ds, decoder: AttributeMetadataDecoder, ordered_attrs: list[str], attr_values: dict[str, list[str]]
) -> None:
    """Render the attribute column: decoded labels with neighbor sequence squares."""
    target = data.get("target")
    if target is None:
        st.info("No attributes")
        return

    decoded = decoder.decode_label_tensor(target)
    radius = 8
    neighbors = get_neighbor_labels(ds, ordered_attrs, data.get("segment_id"), radius=radius)

    html_lines = []
    for k, (v_str, v_idx) in decoded.items():
        color = get_index_color(v_idx)
        pre_html = generate_sequence_html(attr_values, neighbors, k, range(-radius, 0))
        post_html = generate_sequence_html(attr_values, neighbors, k, range(1, radius + 1))
        html_lines.append(
            f"""
            <div class="attributes-text">
                <span class="attr-key" title="{k}"><b>{k}</b>:</span>
                <span class="seq-container">{pre_html}</span>
                <span style="color:{color}; font-weight:semibold; margin: 0 4px;">{v_str}</span>
                <span class="seq-container">{post_html}</span>
            </div>
            """
        )
    st.markdown("".join(html_lines), unsafe_allow_html=True)


# Main #########################################################################


def main():
    st.set_page_config(layout="wide", page_title="IRAP Dataset Viewer")

    config = _render_sidebar_config()
    if config is None:
        return

    result = _ensure_dataset_loaded(config)
    if result is None:
        return
    ds, active_config = result

    # Decoder built from the dataset's authoritative metadata
    decoder = AttributeMetadataDecoder(ds.info.attr_to_value_to_class_idx, ignore_class_idx=IGNORE_LABEL_INDEX)
    ordered_attrs = list(decoder.attr_to_value_to_class_idx.keys())
    attr_values = {a: list(decoder.attr_to_value_to_class_idx[a].keys()) for a in ordered_attrs}

    nav = _render_filters_and_navigation(ds, decoder, active_config.load_key, ordered_attrs, attr_values)
    if nav is None:
        return
    cursor, ds_idx, filtered = nav

    # Load example (cached — see load_example).
    try:
        data = load_example(ds, ds_idx, cache_key=active_config.load_key)
    except Exception as e:
        st.error(f"Error reading item {ds_idx}: {e}")
        return

    # Jitter selection at the bottom of the sidebar
    jitter_selection = st.sidebar.selectbox(
        "ColorJitter",
        _JITTER_OPTIONS,
        index=0,
        help="Preview ColorJitter augmentation as used during training.",
    )

    st.markdown(_VIEWER_CSS, unsafe_allow_html=True)

    col_imgs, col_attrs = st.columns([1, 1])

    with col_imgs:
        _render_image_column(data, ds, active_config.context_offsets, jitter_selection)

    with col_attrs:
        _render_attribute_column(data, ds, decoder, ordered_attrs, attr_values)

    with st.expander("Raw Dictionary"):
        st.write({k: (str(v.shape) if hasattr(v, "shape") else v) for k, v in data.items()})


if __name__ == "__main__":
    main()

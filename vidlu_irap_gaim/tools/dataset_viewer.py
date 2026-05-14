import os
import sys
import random
from pathlib import Path
import typing as T

import streamlit as st

# Add project root to path so we can import vidlu_irap_gaim
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vidlu.data import Record
from vidlu_irap_gaim.data import make_bih_data, make_vietnam_data
from vidlu_irap_gaim.training import make_sequence_color_jitter, JITTER_STANDARD, JITTER_STRONG
from vidlu_irap_gaim.tools.vis_utils import (
    AttributeMetadataDecoder,
    create_composite_view_strip,
    get_index_color,
    tensor_image_to_uint8_np,
)


# --- Dataset detection / loading ---

DATASET_NAMES = ("IRAP_BIH", "IRAP_Vietnam")


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


_DATASETS_DIR_HELP = (
    "Parent directory containing one or more IRAP datasets.\n\n"
    "Expected layout:\n"
    "- IRAP-BiH:   <datasets_dir>/IRAP_BIH/ (data)\n"
    "              <datasets_dir>/IRAP_BIH_METADATA/ (metadata)\n"
    "- IRAP-Vietnam: <datasets_dir>/IRAP_Vietnam/ (data + metadata together)\n\n"
    "Default is read from $IRAP_HOME, then $DATASETS_DIR."
)


@st.cache_data(max_entries=256)
def load_example(_ds, idx: int, cache_key: tuple) -> dict:
    """Cache `ds.get_example(idx)` so Prev/Next revisits don't re-decode PNGs.

    `cache_key` is the active_load tuple; it invalidates the cache when the
    user switches dataset/split/context. The `_ds` underscore prefix tells
    Streamlit to skip hashing the BihSequence (already identified by cache_key).
    """
    return dict(_ds.get_example(idx).items())


@st.cache_resource(hash_funcs={Path: str})
def load_dataset(dataset_name: str, dataset_dir: Path, metadata_dir: Path,
                 split: str, context_sequence: tuple):
    if dataset_name == "IRAP_Vietnam":
        splits = make_vietnam_data(
            dataset_dir=dataset_dir,
            metadata_dir=metadata_dir,
            context_sequence=context_sequence,
        )
    else:
        splits = make_bih_data(
            dataset_dir=dataset_dir,
            metadata_dir=metadata_dir,
            context_sequence=context_sequence,
        )
    return splits[split]


# --- Filtering ---


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
    allowed_idx_by_attr = {
        a: {val_to_idx[a][v] for v in vs}
        for a, vs in filters.items() if vs
    }
    attr_pos = {a: i for i, a in enumerate(val_to_idx.keys())}

    out = []
    for i, sid in enumerate(ds.segment_ids):
        lbls = ds.segment_id_to_labels[sid]
        if all(lbls[attr_pos[a]] in allowed for a, allowed in allowed_idx_by_attr.items()):
            out.append(i)
    return out


def _filters_key(filters: dict[str, set[str]]) -> frozenset:
    return frozenset((a, frozenset(vs)) for a, vs in filters.items() if vs)


# --- Sequence-of-neighbors HTML (unchanged from prior version) ---


def get_neighbor_labels(ds, ordered_attrs: list[str], center_sid: str, radius: int = 5) -> dict:
    try:
        center_int = int(center_sid)
    except (TypeError, ValueError):
        return {}

    neighbors = {}
    sid_to_labels = ds.segment_id_to_labels
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        neighbor_sid = str(center_int + offset)
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
            squares.append(
                f'<div class="seq-square" style="background-color:{n_color};" title="{tooltip}"></div>'
            )
        else:
            squares.append('<div class="seq-square" style="background-color:transparent;"></div>')
    return "".join(squares)


# --- Main UI ---


def main():
    st.set_page_config(layout="wide", page_title="IRAP Dataset Viewer")

    # Datasets directory
    default_datasets_dir = os.environ.get("IRAP_HOME") or os.environ.get("DATASETS_DIR") or ""
    datasets_dir_input = st.sidebar.text_input(
        "Datasets directory", value=default_datasets_dir, help=_DATASETS_DIR_HELP,
    )
    if not datasets_dir_input:
        st.error("Set $IRAP_HOME or $DATASETS_DIR, or enter a path in the sidebar.")
        return
    datasets_dir = Path(datasets_dir_input).expanduser()

    # Dataset selector — only datasets that exist under datasets_dir are offered.
    detected = _detect_datasets(datasets_dir)
    if not detected:
        st.error(
            f"No datasets found under {datasets_dir}. "
            f"Looked for: {', '.join(DATASET_NAMES)}."
        )
        return
    dataset_name = st.sidebar.selectbox("Dataset", detected, index=0)

    ds_dir, md_dir = _resolve_paths(datasets_dir, dataset_name)

    # Split
    split = st.sidebar.selectbox("Split", ["train", "val", "test"])

    # Context sequence
    default_seq_str = "0, -1, -4"
    seq_input = st.sidebar.text_input("Context sequence (e.g., 0, -1, -4)", value=default_seq_str)
    try:
        context_sequence = tuple(int(x.strip()) for x in seq_input.split(","))
    except ValueError:
        st.error("Invalid context sequence format. Use comma-separated integers.")
        return

    # Jitter selection
    jitter_options = ["None", "Standard", "Strong (semi-supervised)"]
    jitter_selection = st.sidebar.selectbox(
        "ColorJitter", jitter_options, index=0,
        help="Preview ColorJitter augmentation as used during training.",
    )

    # Deferred load: gate slow load_dataset() behind an explicit button so users can
    # freely change settings without triggering work. Streamlit cannot interrupt the
    # in-flight Python call, so this is the only reliable cancellation strategy.
    pending_load = (dataset_name, str(ds_dir), str(md_dir), split, context_sequence)
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
        return

    # Use the *active* (loaded) values from here on, not pending ones.
    a_dataset_name, a_ds_dir, a_md_dir, a_split, a_context_sequence = active_load
    a_ds_dir = Path(a_ds_dir)
    a_md_dir = Path(a_md_dir)
    if stale:
        st.sidebar.warning("Settings changed — showing previously loaded data. Click Reload to apply.")

    with st.spinner(f"Loading {a_dataset_name} / {a_split}..."):
        try:
            ds = load_dataset(a_dataset_name, a_ds_dir, a_md_dir, a_split, a_context_sequence)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return

    # Rebind downstream-used names to the active values so captions, cache keys,
    # widget keys, etc. reflect what is actually loaded.
    dataset_name, split, context_sequence = a_dataset_name, a_split, a_context_sequence

    if not ds or len(ds) == 0:
        st.warning("Dataset is empty.")
        return

    # Decoder built from the dataset's authoritative metadata
    decoder = AttributeMetadataDecoder(ds.info.attr_to_value_to_class_idx)
    ordered_attrs = list(decoder.attr_to_value_to_class_idx.keys())
    attr_values = {a: list(decoder.attr_to_value_to_class_idx[a].keys()) for a in ordered_attrs}

    st.sidebar.info(f"Size: {len(ds)} sequences")

    # --- Filtering UI ---
    if "filters" not in st.session_state:
        st.session_state.filters = {}

    # When the dataset (name + split) changes, reset filters and cursor.
    ds_key = (dataset_name, split)
    if st.session_state.get("ds_key") != ds_key:
        st.session_state.ds_key = ds_key
        st.session_state.filters = {a: [] for a in ordered_attrs}
        st.session_state.cursor = 0

    # Ensure filter dict matches current attributes (in case of attribute mismatch across datasets).
    for a in ordered_attrs:
        st.session_state.filters.setdefault(a, [])

    with st.sidebar.expander("Filter by attributes", expanded=False):
        if st.button("Clear filters", width="stretch"):
            for a in ordered_attrs:
                st.session_state.filters[a] = []
        for a in ordered_attrs:
            st.session_state.filters[a] = st.multiselect(
                a, attr_values[a],
                default=st.session_state.filters.get(a, []),
                key=f"filter_{dataset_name}_{split}_{a}",
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

    st.sidebar.write(f"**{len(filtered)}/{len(ds)}** matching")

    if len(filtered) == 0:
        st.warning("No examples match the current filters.")
        return

    # Clamp cursor to filtered range.
    if "cursor" not in st.session_state or st.session_state.cursor >= len(filtered):
        st.session_state.cursor = 0

    # Navigation
    col_nav1, col_nav2 = st.sidebar.columns(2)
    with col_nav1:
        if st.button("Previous", width="stretch"):
            st.session_state.cursor = max(0, st.session_state.cursor - 1)
    with col_nav2:
        if st.button("Next", width="stretch"):
            st.session_state.cursor = min(len(filtered) - 1, st.session_state.cursor + 1)
    if st.sidebar.button("Random", width="stretch"):
        st.session_state.cursor = random.randint(0, len(filtered) - 1)

    # Bind via `key=` so Prev/Next button mutations and the widget share state.
    # Without this, the widget keeps its own internal value and reassignment after
    # rendering overwrites button-driven increments — producing the "updates every
    # other change" symptom.
    cursor = st.sidebar.number_input(
        "Index (within filtered)", min_value=0, max_value=len(filtered) - 1, step=1, key="cursor"
    )
    ds_idx = filtered[cursor]
    st.sidebar.caption(f"Dataset index: {ds_idx}")

    # Load example (cached — see load_example).
    try:
        data = load_example(ds, ds_idx, cache_key=active_load)
    except Exception as e:
        st.error(f"Error reading item {ds_idx}: {e}")
        return

    # CSS (unchanged)
    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )

    col_imgs, col_attrs = st.columns([1, 1])

    with col_imgs:
        rgb = data.get("rgb")
        if rgb is not None:
            jitter_caption_suffix = ""
            if jitter_selection != "None":
                preset = JITTER_STANDARD if jitter_selection == "Standard" else JITTER_STRONG
                jitter_fn = make_sequence_color_jitter(preset=preset)
                # Jitter expects a Record; build a minimal one from the cached rgb.
                rgb = jitter_fn(Record(rgb=rgb))["rgb"]
                jitter_caption_suffix = f" [{jitter_selection.lower().split()[0]} jitter]"

            imgs_np = tensor_image_to_uint8_np(rgb)
            final_img = create_composite_view_strip(imgs_np)

            caption = f"Segment: {data.get('segment_id', 'Unknown')}{jitter_caption_suffix}"
            if len(imgs_np) > 1:
                caption += f" | Context offsets: {', '.join(map(str, context_sequence))}"

            st.image(final_img, caption=caption, width="stretch")

    with col_attrs:
        target = data.get("target")
        if target is not None:
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
        else:
            st.info("No attributes")

    with st.expander("Raw Dictionary"):
        st.write({k: (str(v.shape) if hasattr(v, "shape") else v) for k, v in data.items()})


if __name__ == "__main__":
    main()

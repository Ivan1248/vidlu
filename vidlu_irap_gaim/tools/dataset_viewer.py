import os
import sys
import random
from pathlib import Path
import typing as T

import streamlit as st
import torch
import numpy as np

# Add project root to path so we can import vidlu_irap_gaim
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vidlu_irap_gaim.datasets import make_bih_data, resolve_irap_paths
from vidlu_irap_gaim.training import make_sequence_color_jitter, JITTER_STANDARD, JITTER_STRONG
from vidlu_irap_gaim.vis_utils import (
    AttributeMetadataDecoder,
    create_composite_view_strip,
    get_index_color,
    tensor_image_to_uint8_np,
)


# --- Utilities ---


class DatasetAdapter:
    """
    Adapts the vidlu_irap_gaim dataset for easy visualization.
    Handles data loading and label decoding.
    """

    def __init__(
        self,
        dataset_dir: T.Optional[str] = None,
        metadata_dir: T.Optional[str] = None,
        context_sequence: T.Sequence[int] = (0, -1, -4),
    ):
        try:
            self.ds_dir, self.md_dir = resolve_irap_paths(dataset_dir=dataset_dir, metadata_dir=metadata_dir)
            self.valid_path = True
        except Exception as e:
            self.valid_path = False
            self.error = str(e)
            return

        self.context_sequence = context_sequence

        # Load attribute metadata for decoding (shared utility so inference can reuse it later)
        self.decoder = AttributeMetadataDecoder.load(metadata_dir=self.md_dir)
        self.ordered_attrs = list(self.decoder.attribute_names)
        self.attr_values = dict(self.decoder.attr_to_values)

        # Cache for loaded datasets
        self.datasets = {}

    def get_dataset(self, split: str):
        if split not in self.datasets:
            datasets = make_bih_data(
                dataset_dir=self.ds_dir,
                metadata_dir=self.md_dir,
                context_sequence=self.context_sequence,
                data_types=("rgb", "depth"),
            )
            self.datasets.update(datasets)

        return self.datasets.get(split)

    def decode_labels(self, labels: torch.Tensor) -> dict:
        """
        Convert label tensor to dict of {attribute: value_string}.
        """
        return self.decoder.decode_label_tensor(labels)

    def get_neighbor_labels(self, dataset, center_sid: str, radius: int = 5) -> dict:
        """
        Returns labels for neighboring segments within radius.
        Output: {offset: {attr: class_idx}}
        """
        try:
            center_int = int(center_sid)
        except ValueError:
            return {}

        neighbors = {}
        # Precompute label map access
        sid_to_labels = dataset.segment_id_to_labels

        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue

            neighbor_sid = str(center_int + offset)
            if neighbor_sid in sid_to_labels:
                # Get raw label indices (list of ints)
                raw_labels = sid_to_labels[neighbor_sid]

                # Map to {attr: class_idx}
                attr_labels = {}
                if len(raw_labels) == len(self.ordered_attrs):
                    for i, attr in enumerate(self.ordered_attrs):
                        attr_labels[attr] = raw_labels[i]

                neighbors[offset] = attr_labels

        return neighbors


def generate_sequence_html(
    adapter,
    neighbors: dict,
    attr_key: str,
    offsets: T.Iterable[int],
) -> str:
    """Generates HTML for a sequence of colored squares representing neighbor attributes."""
    squares = []
    for offset in offsets:
        if offset in neighbors and attr_key in neighbors[offset]:
            n_idx = neighbors[offset][attr_key]
            n_color = get_index_color(n_idx)

            # Format tooltip
            sign = "+" if offset > 0 else ""
            tooltip = f"Offset {sign}{offset}: {n_idx}"

            if attr_key in adapter.attr_values and 0 <= n_idx < len(adapter.attr_values[attr_key]):
                tooltip = f"{adapter.attr_values[attr_key][n_idx]} ({n_idx})"

            sq = f'<div class="seq-square" style="background-color:{n_color};" title="{tooltip}"></div>'
            squares.append(sq)
        else:
            squares.append('<div class="seq-square" style="background-color:transparent;"></div>')

    return "".join(squares)


# --- Design: Streamlit UI ---
def main():
    st.set_page_config(layout="wide", page_title="IRAP-BH Dataset Viewer")
    # st.title("IRAP-BH Dataset Viewer")

    # Sidebar: Configuration
    # st.sidebar.header("Configuration")

    # Path handling
    default_irap_home = os.environ.get("IRAP_HOME", "")
    irap_home_input = st.sidebar.text_input("IRAP_HOME path", value=default_irap_home)

    if not irap_home_input:
        st.error("Please set IRAP_HOME environment variable or enter path in sidebar.")
        return

    os.environ["IRAP_HOME"] = irap_home_input  # Set for resolve_irap_paths

    # Context Sequence Configuration
    default_seq_str = "0, -1, -4"
    seq_input = st.sidebar.text_input("Context sequence (e.g., 0, -1, -4)", value=default_seq_str)

    try:
        context_sequence = tuple(int(x.strip()) for x in seq_input.split(","))
    except ValueError:
        st.error("Invalid context sequence format. Use comma-separated integers.")
        return

    # Cache the adapter to prevent reloading metadata on every interaction
    @st.cache_resource(hash_funcs={Path: str})
    def get_adapter(home_path: str, context_seq: tuple):
        if home_path:
            os.environ["IRAP_HOME"] = home_path
        return DatasetAdapter(context_sequence=context_seq)

    adapter = get_adapter(irap_home_input, context_sequence)

    if not adapter.valid_path:
        st.error(f"Could not resolve dataset paths: {adapter.error}")
        return

    st.sidebar.success(f"Loaded metadata: {adapter.md_dir.name}")

    # Split Selection
    split = st.sidebar.selectbox("Split", ["train", "val", "test"])

    # Jitter selection (preview ColorJitter augmentation as used during training)
    jitter_options = ["None", "Standard", "Strong (semi-supervised)"]
    jitter_selection = st.sidebar.selectbox(
        "ColorJitter",
        jitter_options,
        index=0,
        help="Preview ColorJitter augmentation: Standard (data augmentation) or Strong (semi-supervised)",
    )

    # Load Dataset
    with st.spinner(f"Loading {split} split..."):
        try:
            ds = adapter.get_dataset(split)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return

    if not ds or len(ds) == 0:
        st.warning("Dataset is empty.")
        return

    st.sidebar.info(f"Size: {len(ds)} sequences")

    # Navigation
    # Initialize session state for index if not present
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    col_nav1, col_nav2 = st.sidebar.columns(2)

    with col_nav1:
        if st.button("Previous", width="stretch"):
            st.session_state.idx = max(0, st.session_state.idx - 1)
    with col_nav2:
        if st.button("Next", width="stretch"):
            st.session_state.idx = min(len(ds) - 1, st.session_state.idx + 1)

    if st.sidebar.button("Random", width="stretch"):
        st.session_state.idx = random.randint(0, len(ds) - 1)

    # Direct Index Input
    selected_idx = st.sidebar.number_input("Index", 0, len(ds) - 1, st.session_state.idx)
    st.session_state.idx = selected_idx

    # Main Content
    try:
        data_record = ds.get_example(st.session_state.idx)
        # Record object (vidlu.data.record) allows .items() but not .get()
        # Convert to dict for safe access
        data = dict(data_record.items())
    except Exception as e:
        st.error(f"Error reading item {selected_idx}: {e}")
        return

    # Custom CSS for compactness and viewport fit
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
            /* Prevent fade/dim and transitions during reruns */
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

    # Layout: Images on Left (3), Attributes on Right (1)
    col_imgs, col_attrs = st.columns([1, 1])

    with col_imgs:
        rgb = data.get("rgb")
        if rgb is not None:
            # Optionally apply ColorJitter based on selection
            jitter_caption_suffix = ""
            if jitter_selection == "Standard":
                jitter_fn = make_sequence_color_jitter(preset=JITTER_STANDARD)
                jittered_record = jitter_fn(data_record)
                rgb = dict(jittered_record.items()).get("rgb", rgb)
                jitter_caption_suffix = " [standard jitter]"
            elif jitter_selection == "Strong (semi-supervised)":
                jitter_fn = make_sequence_color_jitter(preset=JITTER_STRONG)
                jittered_record = jitter_fn(data_record)
                rgb = dict(jittered_record.items()).get("rgb", rgb)
                jitter_caption_suffix = " [strong jitter]"

            # Data is already in [0,1]; convert directly to uint8 numpy image
            imgs_np = tensor_image_to_uint8_np(rgb)

            # Create composite view
            final_img = create_composite_view_strip(imgs_np)

            # Construct caption
            caption = f"Segment: {data.get('segment_id', 'Unknown')}{jitter_caption_suffix}"
            if len(imgs_np) > 1:
                caption += f" | Context offsets: {', '.join(map(str, context_sequence))}"

            st.image(final_img, caption=caption, width="stretch")

    with col_attrs:
        # st.write("### Attributes")
        target = data.get("target")
        if target is not None:
            decoded = adapter.decode_labels(target)

            # Get neighbor labels for context visualization
            radius = 8
            neighbors = adapter.get_neighbor_labels(ds, data.get("segment_id"), radius=radius)

            # Pure text display with bold keys and compact line height
            html_lines = []
            for k, (v_str, v_idx) in decoded.items():
                color = get_index_color(v_idx)

                # Build Preceding Squares (-radius to -1)
                pre_html = generate_sequence_html(adapter, neighbors, k, range(-radius, 0))

                # Build Succeeding Squares (1 to radius)
                post_html = generate_sequence_html(adapter, neighbors, k, range(1, radius + 1))

                # Layout: [Squares P] [Key]: [Value] [Squares S]
                # Modifying layout to: [Key] [Squares P] [Value] [Squares S] for better alignment?
                # User asked: "before and after the current class text"
                # Let's try: div.attributes-text is flex row.
                # Key (fixed width) | Pre | Value (colored) | Post

                row_html = f"""
                <div class="attributes-text">
                    <span class="attr-key" title="{k}"><b>{k}</b>:</span>
                    <span class="seq-container">{pre_html}</span>
                    <span style="color:{color}; font-weight:semibold; margin: 0 4px;">{v_str}</span>
                    <span class="seq-container">{post_html}</span>
                </div>
                """
                html_lines.append(row_html)

            st.markdown("".join(html_lines), unsafe_allow_html=True)
        else:
            st.info("No attributes")

    # Raw Info (Full Width)
    with st.expander("Raw Dictionary"):
        st.write({k: (str(v.shape) if hasattr(v, "shape") else v) for k, v in data.items()})


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vidlu.data import DataLoader, Record
from vidlu.data.dataset import Dataset as VidluDataset

from vidlu_irap_gaim.metrics import (
    IRAP_ATTRIBUTE_METRIC_NAMES,
    IRAP_IGNORE_MISSING_CLASSES,
    IRAP_MAIN_METRIC,
    get_irap_attribute_metrics,
)
from vidlu_irap_gaim.seq.dataset import (
    FeatDataSource,
    LabelDataSource,
    LogitDataSource,
    PackedSegmentArray,
    SeqEnhDataset,
    make_seq_enh_data,
)
from vidlu_irap_gaim.seq.feats import extract_features, pack_features, read_attribute_slices
from vidlu_irap_gaim.seq.models import (
    GeneralLSTMModel,
    IdentityEncoder,
    LabelEmbeddingEncoder,
)
from vidlu_irap_gaim.seq.pipeline import (
    DEFAULT_LSTM_METRICS,
    get_evaluated_attribute_indices,
    get_restored_checkpoint_name,
    make_feat_cache_dir,
    make_lstm_experiment_args,
    summarize_multi_attribute,
    _train_lstm_experiment,
)

# A window small enough to fit inside each synthetic split, which the same-split
# requirement needs. The production default is symmetric with radius 10.
TEST_CONTEXT_OFFSETS = (-4, -1, 0)


def _feat_source(feat_dir) -> FeatDataSource:
    return FeatDataSource(PackedSegmentArray(feat_dir))


# Feature extraction ###############################################################################


class _RecordListDataset(VidluDataset):
    def __init__(self, records, segment_ids):
        super().__init__(name="stub", data=records, info=dict(segment_ids=segment_ids))


class _FeatModel(nn.Module):
    def __init__(self, input_dim=3, feat_dim=4):
        super().__init__()
        self.lin = nn.Linear(input_dim, feat_dim)
        self.num_forward_calls = 0

    def forward(self, x, return_features=False):
        assert return_features
        self.num_forward_calls += 1
        feats = self.lin(x)
        return (feats,), feats


class _StubTrainer:
    eval_batch_size = 4

    def __init__(self, model):
        self.model = model

    def get_data_loader(self, dataset, *, batch_size, drop_last, shuffle):
        return DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle,
                          drop_last=drop_last)

    def prepare_batch(self, batch):
        return batch


def _make_extraction_setup(num_examples=10):
    torch.manual_seed(0)
    segment_ids = [f"seg{i:03d}" for i in range(num_examples)]
    records = [Record(rgb=torch.randn(3), target=i % 2, segment_id=sid)
               for i, sid in enumerate(segment_ids)]
    dataset = _RecordListDataset(records, segment_ids)
    trainer = _StubTrainer(_FeatModel())
    return trainer, dataset, segment_ids


def test_extract_features_saves_all(tmp_path):
    trainer, dataset, segment_ids = _make_extraction_setup()
    stats = extract_features(trainer, dataset, tmp_path)
    assert stats == dict(saved=len(segment_ids), skipped=0)
    assert trainer.model.num_forward_calls == 3  # ceil(10 / 4)
    for sid in segment_ids:
        feats = np.load(tmp_path / f"{sid}.npy")
        assert feats.dtype == np.float16
        assert feats.shape == (4,)


def test_extract_features_skips_cached(tmp_path):
    trainer, dataset, segment_ids = _make_extraction_setup()
    extract_features(trainer, dataset, tmp_path)
    trainer.model.num_forward_calls = 0

    stats = extract_features(trainer, dataset, tmp_path)
    assert stats == dict(saved=0, skipped=len(segment_ids))
    assert trainer.model.num_forward_calls == 0


def test_extract_features_recomputes_only_missing(tmp_path):
    trainer, dataset, segment_ids = _make_extraction_setup()
    extract_features(trainer, dataset, tmp_path)
    trainer.model.num_forward_calls = 0

    for sid in segment_ids[:2]:
        (tmp_path / f"{sid}.npy").unlink()
    stats = extract_features(trainer, dataset, tmp_path)
    assert stats == dict(saved=2, skipped=len(segment_ids) - 2)
    assert trainer.model.num_forward_calls == 1  # 2 missing examples -> one batch
    for sid in segment_ids:
        assert (tmp_path / f"{sid}.npy").exists()


def test_extract_features_progress_counts_only_uncached_segments(tmp_path, capsys):
    trainer, dataset, segment_ids = _make_extraction_setup()
    extract_features(trainer, dataset, tmp_path, desc="Extracting")
    assert "10/10" in capsys.readouterr().err  # tqdm writes to stderr

    for sid in segment_ids[:3]:
        (tmp_path / f"{sid}.npy").unlink()
    extract_features(trainer, dataset, tmp_path, desc="Extracting")
    err = capsys.readouterr().err
    assert "Extracting" in err
    assert "3/3" in err  # the 7 cached segments are not counted as work


def test_extract_features_float32(tmp_path):
    trainer, dataset, segment_ids = _make_extraction_setup(num_examples=2)
    extract_features(trainer, dataset, tmp_path, dtype="float32")
    assert np.load(tmp_path / f"{segment_ids[0]}.npy").dtype == np.float32


def test_extract_features_rejects_unknown_dtype(tmp_path):
    trainer, dataset, _ = _make_extraction_setup(num_examples=2)
    with pytest.raises(ValueError):
        extract_features(trainer, dataset, tmp_path, dtype="int8")


# Model input forms ################################################################################

# The vidlu model factory and the supervised training step move the model input to the
# device and call `untag` on it, so the input field must be a tensor, not a mapping.

def test_lstm_accepts_a_tensor_for_a_single_encoder():
    torch.manual_seed(0)
    model = GeneralLSTMModel(n_classes=3, input_encoders=dict(feats=IdentityEncoder(input_dim=6)))
    x = torch.randn(2, 3, 6)
    assert torch.equal(model(x), model(dict(feats=x)))


def test_lstm_rejects_an_ambiguous_tensor_for_several_encoders():
    model = GeneralLSTMModel(n_classes=3, input_encoders=dict(
        feats=IdentityEncoder(input_dim=6),
        labels=LabelEmbeddingEncoder(num_embeddings=3, embedding_dim=4)))
    with pytest.raises(ValueError, match="ambiguous"):
        model(torch.randn(2, 3, 6))


def _make_seq_enh_data(inputs, *, attribute=0, context_offsets=TEST_CONTEXT_OFFSETS, **kwargs):
    return make_seq_enh_data(feat_dir=inputs.feat_dir, logit_dir=inputs.logit_dir,
                             attribute=attribute, metadata_dir=inputs.metadata_dir,
                             context_offsets=context_offsets, **kwargs)


def test_seq_enh_dataset_input_is_a_tensor_for_a_single_source(tmp_path):
    datasets = _make_seq_enh_data(_write_synthetic_seq_enh_inputs(tmp_path))
    example = datasets["train"][0]
    assert isinstance(example["data"], torch.Tensor)
    assert example["data"].shape == (3, 6)  # (sequence length, feature dim)
    assert datasets["train"].info["class_count"] == 2
    assert datasets["train"].info["feat_dim"] == 6


def test_an_attribute_missing_from_every_coding_table_keeps_the_other_attributes(tmp_path):
    inputs = _write_synthetic_seq_enh_inputs(tmp_path, with_unlabeled_attribute=True)
    assert len(_make_seq_enh_data(inputs)["train"]) > 0


def test_an_unlabeled_attribute_reports_why_its_subsets_are_empty(tmp_path):
    inputs = _write_synthetic_seq_enh_inputs(tmp_path, with_unlabeled_attribute=True)
    with pytest.raises(RuntimeError, match="target_attribute_unlabeled=40"):
        _make_seq_enh_data(inputs, attribute=2)


def test_a_context_reaching_past_the_road_is_an_error_not_a_wraparound(tmp_path):
    """`seq[pos + off]` with a negative position is a valid Python index, so it would
    silently take a segment from the *end* of the road."""
    dataset = _make_seq_enh_data(_write_synthetic_seq_enh_inputs(tmp_path))["train"]

    # Direct construction bypasses the factory's `min(indices) < 0` guard, which is the case
    # the check in `get_example` exists for.
    unguarded = SeqEnhDataset(
        subset="train", data_sources=dataset.data_sources,
        segment_id_list=[dataset.segment_ids[0]], seq_index=dataset.seq_index,
        road_to_seq=dataset.road_to_seq, context_offsets=(-400, -1, 0),
        target_label_map=dataset.target_label_map,
        target_attribute_index=dataset.target_attribute_index, class_count=2)

    with pytest.raises(IndexError, match="outside the road sequence"):
        unguarded.get_example(0)


# The smoothing window ############################################################################


def test_a_window_without_the_classified_segment_is_rejected(tmp_path):
    inputs = _write_synthetic_seq_enh_inputs(tmp_path)
    with pytest.raises(ValueError, match="does not contain 0"):
        _make_seq_enh_data(inputs, context_offsets=(-4, -1))


def test_a_window_running_backwards_in_time_is_rejected_not_sorted(tmp_path):
    """The LSTM reads the sequence in the given order, so reordering it silently would
    change what an existing command does."""
    inputs = _write_synthetic_seq_enh_inputs(tmp_path)
    with pytest.raises(ValueError, match="strictly increasing"):
        _make_seq_enh_data(inputs, context_offsets=(0, -1, -4))


def test_the_target_index_is_where_the_classified_segment_sits(tmp_path):
    inputs = _write_synthetic_seq_enh_inputs(tmp_path)
    assert _make_seq_enh_data(inputs)["train"].info["target_index"] == 2
    assert _make_seq_enh_data(
        inputs, context_offsets=(-1, 0, 1))["train"].info["target_index"] == 1


def test_a_window_crossing_a_split_boundary_is_dropped(tmp_path):
    """A val segment refined using the base model's predictions on its *training*
    neighbours would be scored against optimistically accurate context."""
    inputs = _write_synthetic_seq_enh_inputs(tmp_path)
    datasets = _make_seq_enh_data(inputs)

    val_ids = set(inputs.segment_ids[40:50])
    for sid in datasets["val"].segment_ids:
        road_id, pos = datasets["val"].seq_index[sid]
        window = [inputs.segment_ids[pos + off] for off in TEST_CONTEXT_OFFSETS]
        assert set(window) <= val_ids
    # The first four segments of the split cannot have a full window inside it.
    assert set(datasets["val"].segment_ids) == set(inputs.segment_ids[44:50])


# Sequence inputs are the base model's outputs ####################################################


def test_the_label_input_is_the_predicted_class_not_the_annotation(tmp_path):
    """The original implementation's `LabelDataAccess` reads `y_pred`. Reading the
    ground-truth label instead would feed the target's own answer in at offset 0."""
    inputs = _write_synthetic_seq_enh_inputs(tmp_path)
    store = PackedSegmentArray(inputs.logit_dir)
    columns = slice(*read_attribute_slices(inputs.logit_dir)[0])

    logits = LogitDataSource(store, columns).get_sequence(["s005", "s006"])
    labels = LabelDataSource(store, columns).get_sequence(["s005", "s006"])

    assert logits.shape == (2, 2)  # (sequence length, attr_a's class count)
    assert labels.shape == (2, 1)
    assert torch.equal(labels, logits.argmax(dim=-1, keepdim=True))
    # Always a valid embedding row, which is why no "unknown" row is needed.
    assert bool(((labels >= 0) & (labels < 2)).all())


def test_the_label_input_does_not_come_from_the_target_label_map(tmp_path):
    """The predicted class and the annotation disagree in general; if they were the same
    array, offset 0 of the label input would be the target itself."""
    inputs = _write_synthetic_seq_enh_inputs(tmp_path)
    datasets = _make_seq_enh_data(inputs, input_types=("labels",))
    dataset = datasets["train"]

    targets_matching_own_input = 0
    for i in range(len(dataset)):
        example = dataset[i]
        own_label = int(example["data"][dataset.info["target_index"], 0])
        targets_matching_own_input += own_label == int(example["target"])
    # A leaked target would match on every single example.
    assert targets_matching_own_input < len(dataset)


def test_logits_are_extracted_alongside_features(tmp_path):
    trainer, dataset, segment_ids = _make_extraction_setup()
    feat_dir, logit_dir = tmp_path / "feats", tmp_path / "logits"

    stats = extract_features(trainer, dataset, feat_dir, logit_dir=logit_dir)

    assert stats == dict(saved=len(segment_ids), skipped=0)
    assert trainer.model.num_forward_calls == 3  # one pass, not one per output kind
    assert read_attribute_slices(logit_dir) == [(0, 4)]  # `_FeatModel`'s single head
    for sid in segment_ids:
        assert np.load(logit_dir / f"{sid}.npy").shape == (4,)


def test_a_feature_only_cache_is_completed_rather_than_reused(tmp_path):
    """`skip_existing` keyed on the features alone would leave the logits missing."""
    trainer, dataset, segment_ids = _make_extraction_setup()
    feat_dir, logit_dir = tmp_path / "feats", tmp_path / "logits"
    extract_features(trainer, dataset, feat_dir)  # as an earlier version wrote it
    trainer.model.num_forward_calls = 0

    stats = extract_features(trainer, dataset, feat_dir, logit_dir=logit_dir)

    assert stats == dict(saved=len(segment_ids), skipped=0)
    for sid in segment_ids:
        assert (logit_dir / f"{sid}.npy").exists()


# Packed feature store #############################################################################


def test_packing_reproduces_the_per_segment_features(tmp_path):
    feat_dir = _write_synthetic_seq_enh_inputs(tmp_path).feat_dir
    source = _feat_source(feat_dir)

    assert source.feat_dim == 6
    for sid in ["s000", "s007", "s019"]:
        assert source.has_features(sid)
        expected = np.load(feat_dir / f"{sid}.npy").reshape(-1).astype(np.float32)
        assert torch.equal(source.get_sequence([sid])[0], torch.from_numpy(expected))
    assert not source.has_features("s999")


def test_packing_is_redone_when_a_segment_is_added(tmp_path):
    feat_dir = _write_synthetic_seq_enh_inputs(tmp_path).feat_dir
    assert _feat_source(feat_dir).has_features("s019")

    np.save(feat_dir / "s099.npy", np.zeros(6, dtype=np.float16))
    source = _feat_source(feat_dir)
    assert source.has_features("s099")
    assert torch.equal(source.get_sequence(["s099"])[0], torch.zeros(6))


def test_packing_is_redone_when_a_segment_is_replaced_by_another(tmp_path):
    """The count is unchanged, so only comparing segment IDs catches this."""
    feat_dir = _write_synthetic_seq_enh_inputs(tmp_path).feat_dir
    pack_features(feat_dir)

    (feat_dir / "s019.npy").unlink()
    np.save(feat_dir / "s099.npy", np.full(6, 3, dtype=np.float16))
    source = _feat_source(feat_dir)

    assert not source.has_features("s019")
    assert source.has_features("s099")
    assert torch.equal(source.get_sequence(["s099"])[0], torch.full((6,), 3.))


def test_a_truncated_pack_index_is_rebuilt(tmp_path):
    """An interrupted index write must not make every later run fail to parse it."""
    feat_dir = _write_synthetic_seq_enh_inputs(tmp_path).feat_dir
    array_path, index_path = pack_features(feat_dir)
    index_path.write_text(index_path.read_text()[:20])  # interrupted mid-dump

    assert pack_features(feat_dir) == (array_path, index_path)
    assert _feat_source(feat_dir).has_features("s019")


def test_packing_rejects_a_directory_mixing_dtypes(tmp_path):
    """Packing them would cast to whichever dtype the first file happened to have."""
    feat_dir = _write_synthetic_seq_enh_inputs(tmp_path).feat_dir
    np.save(feat_dir / "s099.npy", np.zeros(6, dtype=np.float32))

    with pytest.raises(RuntimeError, match="mixes dtypes"):
        pack_features(feat_dir)


def test_examples_are_read_from_the_pack_rather_than_per_segment_files(tmp_path, monkeypatch):
    inputs = _write_synthetic_seq_enh_inputs(tmp_path)
    pack_features(inputs.feat_dir)  # what the pipeline does after extraction
    datasets = _make_seq_enh_data(inputs)

    loaded_paths = []
    real_load = np.load
    monkeypatch.setattr(np, "load", lambda p, *a, **k: (loaded_paths.append(p), real_load(p, *a, **k))[1])
    for i in range(len(datasets["train"])):
        datasets["train"][i]
    assert loaded_paths == []


# Cache key / checkpoint resolution ################################################################


def test_make_feat_cache_dir_is_keyed_by_experiment_and_checkpoint(tmp_path):
    dir_a = make_feat_cache_dir(tmp_path, "exp1", "0_10_0.5", "float16")
    dir_b = make_feat_cache_dir(tmp_path, "exp1", "1_20_0.6", "float16")
    dir_c = make_feat_cache_dir(tmp_path, "exp2", "0_10_0.5", "float16")
    assert len({dir_a, dir_b, dir_c}) == 3
    assert dir_a == make_feat_cache_dir(tmp_path, "exp1", "0_10_0.5", "float16")

    with (dir_a / "provenance.json").open() as f:
        assert json.load(f) == dict(experiment="exp1", checkpoint="0_10_0.5",
                                    feat_dtype="float16")


def test_reusing_a_feature_cache_with_another_dtype_is_rejected(tmp_path):
    """A second dtype in one directory would leave the pack silently downcast."""
    make_feat_cache_dir(tmp_path, "exp1", "0_10_0.5", "float16")
    with pytest.raises(RuntimeError, match="feat_dtype='float16', not 'float32'"):
        make_feat_cache_dir(tmp_path, "exp1", "0_10_0.5", "float32")


def test_get_restored_checkpoint_name():
    cpman = SimpleNamespace(saved=["0_1_0.500", "1_2_0.700"], last_index=1,
                            resuming_required=False, experiment_dir="d")
    assert get_restored_checkpoint_name(cpman) == "1_2_0.700"

    cpman.last_index = -1
    with pytest.raises(RuntimeError):
        get_restored_checkpoint_name(cpman)

    cpman.last_index = 1
    cpman.resuming_required = True
    with pytest.raises(RuntimeError):
        get_restored_checkpoint_name(cpman)


def test_make_lstm_experiment_args():
    args = make_lstm_experiment_args(
        feat_dir=Path("C:/cache/seq_enh_feats/abc/feats"),
        logit_dir=Path("C:/cache/seq_enh_feats/abc/logits"), attribute=2,
        metadata_dir=Path("C:/data/IRAP_Vietnam"), context_offsets=(-4, -1, 0),
        input_types=("feats",),
        trainer_str="irap_gaim.seq_enh_lstm_trainer", metrics_str="", main_metrics_str="mF1",
        resume="?", tracker=None, device="cuda:0")
    assert args.data == ("irap_gaim.make_seq_enh_data(feat_dir='C:/cache/seq_enh_feats/abc/feats',"
                         "logit_dir='C:/cache/seq_enh_feats/abc/logits',"
                         "attribute=2,metadata_dir='C:/data/IRAP_Vietnam',"
                         "context_offsets=(-4,-1,0),input_types=('feats',))")
    assert "n_classes=data.train.info.class_count" in args.model
    assert "input_dim=data.train.info.feat_dim" in args.model
    # Read at the classified segment, which is not the sequence midpoint for every window.
    assert "middle_index=data.train.info.target_index" in args.model
    assert (args.input_adapter, args.trainer, args.metrics) == (
        "id", "irap_gaim.seq_enh_lstm_trainer", "")
    assert (args.resume, args.device, args.distributed) == ("?", "cuda:0", False)
    assert args.main_metrics == "mF1"


# Metric definition shared by the base run and sequential enhancement ##############################


def test_single_attribute_metrics_match_the_multi_attribute_ones():
    from vidlu_irap_gaim.metrics import MultiAttributeClassificationMetrics

    multi = MultiAttributeClassificationMetrics(
        {"a": (0, 3)}, metrics=tuple("a" + n for n in IRAP_ATTRIBUTE_METRIC_NAMES),
        ignore_missing_classes=IRAP_IGNORE_MISSING_CLASSES)
    single = get_irap_attribute_metrics(class_count=3)

    inner = multi.attr_to_cm_metrics["a"]
    assert set(inner.metrics) == set(IRAP_ATTRIBUTE_METRIC_NAMES)
    assert set(single.metrics) == set(IRAP_ATTRIBUTE_METRIC_NAMES) | {"n"}
    assert inner.ignore_missing_classes == single.ignore_missing_classes


def test_averaging_per_attribute_metrics_reproduces_the_multi_attribute_ones():
    """`amF1` is the mean of the per-attribute `mF1`, so the summary can just average."""
    from vidlu.utils.collections import NameDict
    from vidlu_irap_gaim.metrics import MultiAttributeClassificationMetrics

    torch.manual_seed(0)
    class_counts = {0: 3, 1: 2}
    multi = MultiAttributeClassificationMetrics(
        {a: (a, c) for a, c in class_counts.items()},
        metrics=tuple("a" + n for n in IRAP_ATTRIBUTE_METRIC_NAMES),
        ignore_missing_classes=IRAP_IGNORE_MISSING_CLASSES)
    singles = {a: get_irap_attribute_metrics(class_count=c) for a, c in class_counts.items()}

    target = torch.stack([torch.randint(c, (32,)) for c in class_counts.values()], dim=1)
    outs = [torch.randn(32, c) for c in class_counts.values()]
    multi.update(NameDict(out=outs, target=target))
    for a, m in singles.items():
        m.update(NameDict(out=outs[a], target=target[:, a]))

    per_attribute = {a: dict(split=m.compute()) for a, m in singles.items()}
    averaged, missing = summarize_multi_attribute(per_attribute, list(class_counts))
    assert missing == []
    assert averaged["split"]["num_attributes"] == 2
    expected = multi.compute()
    for name in IRAP_ATTRIBUTE_METRIC_NAMES:
        assert averaged["split"][f"a{name}"] == pytest.approx(expected[f"a{name}"], rel=1e-6)


def test_summary_leaves_out_and_reports_attributes_without_results():
    per_attribute = {0: dict(val=dict(mF1=0.5, mP=0.4, mR=0.6)),
                     1: dict(val=dict(mF1=0.7, mP=0.8, mR=0.6))}
    averaged, missing = summarize_multi_attribute(per_attribute, [0, 1, 4])
    assert missing == [4]
    assert averaged["val"]["amF1"] == pytest.approx(0.6)
    assert averaged["val"]["num_attributes"] == 2


def test_the_evaluated_attributes_are_the_canonical_subset_labeled_in_this_release():
    """What `train_seq_enh(attributes=None)` now trains: exactly what the summary averages.

    IRAP-Vietnam annotates 34 of the canonical 41 and adds 4 attributes of its own; the 4 are
    outside `get_attrs_to_include`, so neither the base run's `amF1` nor this covers them.
    """
    from irap_data.attrs import (IRAP_BH_ATTRS_TO_INCLUDE, IRAP_VIETNAM_ATTRS_ALL,
                                 IRAP_VIETNAM_ATTRS_SHARED)

    schema_order = list(dict.fromkeys([*IRAP_BH_ATTRS_TO_INCLUDE, *IRAP_VIETNAM_ATTRS_ALL]))
    vietnam_info = dict(
        attr_to_value_to_class_idx={a: {} for a in schema_order},
        attr_to_num_labeled={a: (7 if a in IRAP_VIETNAM_ATTRS_ALL else 0) for a in schema_order})

    indices = get_evaluated_attribute_indices(vietnam_info)
    assert [schema_order[i] for i in indices] == list(IRAP_VIETNAM_ATTRS_SHARED)
    assert len(indices) == 34


def test_summary_drops_attributes_whose_metric_is_undefined():
    """`MCC` is NaN where a split's ground truth is one class. Counting that attribute would
    make the whole `aMCC` NaN; counting it as 0 would understate the rest."""
    per_attribute = {0: dict(val=dict(mF1=0.5, MCC=0.2)),
                     1: dict(val=dict(mF1=0.7, MCC=float("nan")))}
    averaged, _ = summarize_multi_attribute(per_attribute, [0, 1])
    assert averaged["val"]["aMCC"] == pytest.approx(0.2)
    assert averaged["val"]["amF1"] == pytest.approx(0.6)
    assert averaged["val"]["num_attributes"] == 2


def test_summary_reports_an_all_undefined_metric_as_undefined_rather_than_zero():
    per_attribute = {0: dict(val=dict(MCC=float("nan"))), 1: dict(val=dict(MCC=float("nan")))}
    averaged, _ = summarize_multi_attribute(per_attribute, [0, 1])
    assert np.isnan(averaged["val"]["aMCC"])


def test_summary_rejects_attributes_evaluated_with_different_metrics():
    """Averaging them would give mP a denominator of 1 and mF1 one of 2, silently."""
    per_attribute = {0: dict(val=dict(mF1=0.5, mP=0.4, mR=0.6)),
                     1: dict(val=dict(mF1=0.7, mR=0.6))}
    with pytest.raises(RuntimeError, match="different denominator"):
        summarize_multi_attribute(per_attribute, [0, 1])


# End-to-end LSTM experiment on synthetic metadata + features ######################################


def _write_synthetic_seq_enh_inputs(root: Path, *, num_segments=60, feat_dim=6,
                                    with_unlabeled_attribute=False):
    """Creates IRAP-style metadata JSONs and per-segment base-model outputs.

    Writes both of what `extract_features` writes: features, and the per-attribute
    logits with the `attribute_slices.json` describing their columns.

    `with_unlabeled_attribute` adds an attribute that appears in the metadata but
    in no coding table, as IRAP-Vietnam's flow attributes do.
    """
    rng = np.random.RandomState(0)
    metadata_dir = root / "metadata"
    feat_dir = root / "feats"
    logit_dir = root / "logits"
    metadata_dir.mkdir()
    feat_dir.mkdir()
    logit_dir.mkdir()

    segment_ids = [f"s{i:03d}" for i in range(num_segments)]
    attrs = ["attr_a", "attr_b"]
    attr_values = {"attr_a": ["v0", "v1"], "attr_b": ["w0", "w1", "w2"]}
    coded_attrs = list(attrs)
    if with_unlabeled_attribute:
        attrs.append("attr_empty")
        attr_values["attr_empty"] = ["e0", "e1"]

    with (metadata_dir / "attribute_metadata.json").open("w") as f:
        json.dump(dict(
            attribute_to_idx={a: i for i, a in enumerate(attrs)},
            attribute_value_to_irap_number={
                a: {v: i + 1 for i, v in enumerate(vals)} for a, vals in attr_values.items()},
        ), f)
    with (metadata_dir / "road_id_to_segment_id_sequence.json").open("w") as f:
        json.dump(dict(road0=segment_ids), f)
    with (metadata_dir / "segment_id_to_road_data.json").open("w") as f:
        json.dump({sid: dict(required_attributes={
            a: rng.randint(len(attr_values[a])) + 1 for a in coded_attrs})
            for sid in segment_ids}, f)
    # Contiguous runs long enough that a window fits inside each split.
    with (metadata_dir / "splits.json").open("w") as f:
        json.dump(dict(train=segment_ids[:40], val=segment_ids[40:50],
                       test=segment_ids[50:]), f)

    # One head per attribute, as the base classifier has, concatenated per segment.
    slices, start = [], 0
    for a in attrs:
        slices.append([start, start + len(attr_values[a])])
        start += len(attr_values[a])
    (logit_dir / "attribute_slices.json").write_text(json.dumps(slices))

    for sid in segment_ids:
        np.save(feat_dir / f"{sid}.npy", rng.randn(feat_dim).astype(np.float16))
        np.save(logit_dir / f"{sid}.npy", rng.randn(start).astype(np.float16))
    return SimpleNamespace(metadata_dir=metadata_dir, feat_dir=feat_dir, logit_dir=logit_dir,
                           segment_ids=segment_ids, attribute_slices=slices)


def _make_stub_dirs(root: Path):
    dirs = SimpleNamespace(datasets=root / "datasets", cache=root / "cache",
                           saved_states=root / "saved_states", pretrained=root / "pretrained")
    for d in vars(dirs).values():
        d.mkdir()
    return dirs


class _RecordingTracker:
    """Stands in for the wandb tracker, recording only what `finish` was told."""

    def __init__(self):
        self.exit_code = None

    def finish(self, exit_code):
        self.exit_code = exit_code


def _lstm_args_for(inputs, *, attribute=0, context_offsets=TEST_CONTEXT_OFFSETS,
                   input_types=("feats",)):
    return make_lstm_experiment_args(
        feat_dir=inputs.feat_dir, logit_dir=inputs.logit_dir, attribute=attribute,
        metadata_dir=inputs.metadata_dir, context_offsets=context_offsets,
        input_types=input_types,
        trainer_str="irap_gaim.seq_enh_lstm_trainer,epoch_count=2,batch_size=4,"
                    "eval_batch_size=4,eval_count=2",
        metrics_str=DEFAULT_LSTM_METRICS, main_metrics_str=IRAP_MAIN_METRIC, resume=None,
        tracker=None, device="cpu")


def _raise_boom(*args, **kwargs):
    raise RuntimeError("boom")


def test_a_failed_lstm_training_is_not_reported_to_the_tracker_as_successful(tmp_path,
                                                                            monkeypatch):
    """`finish(exit_code=0)` in a `finally` would mark a crashed run clean in wandb."""
    inputs = _write_synthetic_seq_enh_inputs(tmp_path)
    dirs = _make_stub_dirs(tmp_path)
    args = _lstm_args_for(inputs)

    tracker = _RecordingTracker()
    stub_experiment = SimpleNamespace(
        tracker=tracker, data={"train": []},
        trainer=SimpleNamespace(train=_raise_boom))

    import vidlu.experiments as ve
    monkeypatch.setattr(ve.TrainingExperiment, "from_args",
                        staticmethod(lambda *a, **kw: stub_experiment))
    with pytest.raises(RuntimeError, match="boom"):
        _train_lstm_experiment(args, dirs)
    assert tracker.exit_code not in (0, None)


def test_a_successful_lstm_training_reports_a_zero_exit_code(tmp_path, monkeypatch):
    inputs = _write_synthetic_seq_enh_inputs(tmp_path)
    dirs = _make_stub_dirs(tmp_path)
    args = _lstm_args_for(inputs)

    tracker = _RecordingTracker()
    import vidlu.experiments as ve
    real_from_args = ve.TrainingExperiment.from_args

    def from_args_with_tracker(*a, **kw):
        experiment = real_from_args(*a, **kw)
        experiment.tracker = tracker  # the args' tracker string cannot name a test double
        return experiment

    monkeypatch.setattr(ve.TrainingExperiment, "from_args",
                        staticmethod(from_args_with_tracker))
    _train_lstm_experiment(args, dirs)
    assert tracker.exit_code == 0


def test_lstm_experiment_end_to_end(tmp_path):
    inputs = _write_synthetic_seq_enh_inputs(tmp_path)
    dirs = _make_stub_dirs(tmp_path)

    # All three sources, so the run covers the mapping form of `data` (and thus the
    # per-input encoders) rather than only the bare-tensor one.
    args = _lstm_args_for(inputs, input_types=("feats", "logits", "labels"))
    split_to_metrics = _train_lstm_experiment(args, dirs)

    # Accuracy comes from the classification problem defaults, the rest from DEFAULT_LSTM_METRICS.
    assert {"A", "mP", "mR", "mF1", "n"} <= set(split_to_metrics["val"])
    # A checkpoint was saved under the stub saved_states root.
    assert any(dirs.saved_states.rglob("model_state.pth"))
    # `resume=None` means the last checkpoint is the one evaluated, and checkpoints are ranked
    # by the main metric, so the last checkpoint's `perf` is the reported main metric value.
    perf_paths = sorted(dirs.saved_states.rglob("perf.json"),
                        key=lambda p: int(p.parent.name.split("_")[0]))
    last_perf = json.loads(perf_paths[-1].read_text())
    assert last_perf == pytest.approx(split_to_metrics["val"][IRAP_MAIN_METRIC])

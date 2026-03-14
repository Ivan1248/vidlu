import torch
import shutil
from pathlib import Path
from vidlu_irap_gaim.training.dynamic_weights import DynamicBalancedRecallWeights
from vidlu.utils.collections import NameDict
import os


# Mock classes to simulate environment
class MockDataset:
    def __init__(self):
        self.info = NameDict(
            class_counts=(2,),
            attribute_names=["attr0"],
            attr_idx_to_class_occurrence_counts={0: torch.tensor([10, 20])},
        )
        # Add required attributes for check in compute_attr_idx_to_class_occurrence_counts
        self.segment_id_to_labels = {"s1": torch.tensor([0]), "s2": torch.tensor([1])}
        self.segment_ids = ["s1", "s2"]

    def info_cache_hdd(self, *args, **kwargs):
        return self


class MockLoss:
    def set_attrs_idx(self, idx):
        pass

    def set_class_weights(self, weights):
        self.weights = weights


class MockMetric:
    def get_internal_metrics(self):
        # Return stats that would produce different weights than random initialization
        return {
            0: {"tp": torch.tensor([5.0, 0.0]), "actual": torch.tensor([5.0, 10.0]), "pos": torch.tensor([5.0, 0.0])}
        }


class MockEvent:
    def __init__(self):
        self._handlers = []

    def handler(self, f):
        self._handlers.append(f)
        return f

    def add_handler(self, f):
        self._handlers.append(f)

    def fire(self, state):
        for h in self._handlers:
            h(state)


class MockTrainer:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.data = {"train": MockDataset(), "val": MockDataset()}
        self.loss = MockLoss()
        self.metrics = [MockMetric()]
        self.evaluation = NameDict(epoch_completed=MockEvent())
        self.ext = None

    def state_dict(self):
        # Simulate Trainer.state_dict logic
        return {"extensions": {str(type(self.ext)): self.ext.state_dict()}}

    def load_state_dict(self, state_dict):
        # Simulate Trainer.load_state_dict logic
        if "extensions" in state_dict and str(type(self.ext)) in state_dict["extensions"]:
            self.ext.load_state_dict(state_dict["extensions"][str(type(self.ext))])


def test():
    cache_dir = Path("tmp_repro_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    print("--- Setting up initial training session ---")
    t1 = MockTrainer()
    # Fix: explicitly pass attrs_idx to avoid looking up standard attributes in the mock dataset
    ext1 = DynamicBalancedRecallWeights(cache_dir=cache_dir, attrs_idx=[0])
    t1.ext = ext1

    # Initialize: computes initial weights based on occurrence counts (random recalls)
    ext1.initialize(t1)

    # For verification, we can look at the weights stored in the extension.
    # Note: initialization uses _estimate_random_classifier_recalls if no result_lists are present (which is true at init).
    initial_weights = ext1.attr_idx_to_class_weights[0].clone()
    print(f"Initial weights (random recalls): {initial_weights}")

    # Simulate an evaluation phase update
    print("Simulating evaluation update...")
    t1.evaluation.epoch_completed.fire(NameDict(split_name="val"))

    updated_weights = ext1.attr_idx_to_class_weights[0].clone()
    print(f"Updated weights (from stats): {updated_weights}")

    if torch.allclose(initial_weights, updated_weights):
        print("WARNING: Weights did not change. Test might be invalid.")
    else:
        print("Weights successfully updated.")

    # Save state
    print("Saving state...")
    state = t1.state_dict()

    print("\n--- Resuming in new session ---")
    t2 = MockTrainer()
    # Fix: explicitly pass attrs_idx here too
    ext2 = DynamicBalancedRecallWeights(cache_dir=cache_dir, attrs_idx=[0])
    t2.ext = ext2

    # Initialize: resets to initial random weights
    ext2.initialize(t2)
    print(f"Initialized weights (should be initial): {ext2.attr_idx_to_class_weights[0]}")

    # Load state
    print("Loading state...")
    t2.load_state_dict(state)
    print(f"Loaded weights: {ext2.attr_idx_to_class_weights[0]}")

    resumed_weights = ext2.attr_idx_to_class_weights[0]

    # Verification
    if torch.allclose(resumed_weights, updated_weights):
        print("\n[SUCCESS] Weights were preserved correctly!")
    else:
        print("\n[FAILURE] Weights were NOT preserved (reset to initial).")
        print(f"Expected: {updated_weights}")
        print(f"Got:      {resumed_weights}")

    # Cleanup
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


if __name__ == "__main__":
    test()

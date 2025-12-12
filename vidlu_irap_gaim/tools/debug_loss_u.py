import torch
import sys
import os
from functools import partial

# Add project root to path if needed (though usually run from root)
# sys.path.append(os.getcwd())

from vidlu_irap_gaim.training import make_sequence_color_jitter, JITTER_STRONG


def debug_jitter_dimensions():
    print("Debugging make_sequence_color_jitter dimensions...")

    # 1. Test 4D Input (T, C, H, W)
    print("\n--- Test 4D Input (T, C, H, W) ---")
    T, C, H, W = 3, 3, 32, 32
    x_4d = torch.rand(T, C, H, W)
    record_4d = {"rgb": x_4d}

    jitter_fn = make_sequence_color_jitter(preset=JITTER_STRONG)
    out_4d = jitter_fn(record_4d)["rgb"]

    diff_4d = (x_4d - out_4d).abs().sum()
    print(f"4D Input Diff: {diff_4d.item()}")
    if diff_4d > 0:
        print("4D Input: PASS (Jitter applied)")
    else:
        print("4D Input: FAIL (No jitter)")

    # 2. Test 5D Input (B, T, C, H, W)
    print("\n--- Test 5D Input (B, T, C, H, W) ---")
    B = 2
    x_5d = torch.rand(B, T, C, H, W)
    record_5d = {"rgb": x_5d}

    out_5d = jitter_fn(record_5d)["rgb"]

    diff_5d = (x_5d - out_5d).abs().sum()
    print(f"5D Input Diff: {diff_5d.item()}")
    if diff_5d > 0:
        print("5D Input: PASS (Jitter applied)")
    else:
        print("5D Input: FAIL (Identity returned)")


if __name__ == "__main__":
    debug_jitter_dimensions()

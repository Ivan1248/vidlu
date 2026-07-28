import os

# Fix OMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR
from torch.optim import Adam
import math


def quarter_cos(progress):
    # progress is epoch / total_epochs
    # returns cos(progress * pi / 2)
    return math.cos(progress * math.pi / 2)


def main():
    from vidlu_irap_gaim.training.extensions import phase_lr_decay_factor

    # Parameters
    frozen_epochs = 2
    total_epochs = 10

    frozen_lr = 5e-5
    frozen_lambda = phase_lr_decay_factor(0.8 ** 2, frozen_epochs)

    finetune_lr = 1e-5
    finetune_lambda = phase_lr_decay_factor(0.88 ** 13, total_epochs - frozen_epochs)

    print(f"{'Epoch':<5} | {'Phase':<10} | {'FTF LR':<15} | {'QCos LR':<15} | {'Mult LR':<15}")
    print("-" * 75)

    # ---------------------------------------------------------
    # Simulation 1: FreezeThenFinetune
    # ---------------------------------------------------------
    params_ftf = [torch.nn.Parameter(torch.zeros(1))]
    params_ftf[0].requires_grad = True

    optimizer_ftf = None
    scheduler_ftf = None

    lrs_ftf = []
    epochs = []

    # ---------------------------------------------------------
    # Simulation 2: QuarterCosLR
    # ---------------------------------------------------------
    start_lr = 5e-5
    params_qcos = [torch.nn.Parameter(torch.zeros(1))]
    params_qcos[0].requires_grad = True
    optimizer_qcos = Adam(params_qcos, lr=start_lr)

    lr_lambda_qcos = lambda e: quarter_cos(0 if e == 0 else e / total_epochs)
    scheduler_qcos = LambdaLR(optimizer_qcos, lr_lambda=lr_lambda_qcos)

    lrs_qcos = []

    # ---------------------------------------------------------
    # Simulation 3: MultiplicativeLR (Standard)
    # ---------------------------------------------------------
    params_mult = [torch.nn.Parameter(torch.zeros(1))]
    params_mult[0].requires_grad = True
    optimizer_mult = Adam(params_mult, lr=start_lr)

    # Using 0.9 as a representative decay factor
    mult_gamma = 0.9
    scheduler_mult = MultiplicativeLR(optimizer_mult, lr_lambda=lambda e: mult_gamma)

    lrs_mult = []

    for epoch in range(total_epochs):
        epochs.append(epoch)

        # --- FreezeThenFinetune Step ---
        if epoch == 0:
            optimizer_ftf = Adam(params_ftf, lr=frozen_lr)
            scheduler_ftf = MultiplicativeLR(optimizer_ftf, lr_lambda=lambda e: frozen_lambda)
            phase = "Frozen"
        elif epoch == frozen_epochs:
            optimizer_ftf = Adam(params_ftf, lr=finetune_lr)
            scheduler_ftf = MultiplicativeLR(optimizer_ftf, lr_lambda=lambda e: finetune_lambda)
            phase = "Finetune"
        else:
            phase = "Frozen" if epoch < frozen_epochs else "Finetune"

        lr_ftf = optimizer_ftf.param_groups[0]["lr"]
        lrs_ftf.append(lr_ftf)

        # --- QuarterCosLR Step ---
        lr_qcos = optimizer_qcos.param_groups[0]["lr"]
        lrs_qcos.append(lr_qcos)

        # --- MultiplicativeLR Step ---
        lr_mult = optimizer_mult.param_groups[0]["lr"]
        lrs_mult.append(lr_mult)

        print(f"{epoch:<5} | {phase:<10} | {lr_ftf:.2e}        | {lr_qcos:.2e}        | {lr_mult:.2e}")

        # Steps
        scheduler_ftf.step()
        scheduler_qcos.step()
        scheduler_mult.step()

    # Plotting Log Scale
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs_ftf, marker="o", linestyle="-", color="b", label="FreezeThenFinetune")
    plt.plot(epochs, lrs_qcos, marker="s", linestyle="--", color="g", label="QuarterCosLR (start=5e-5)")
    plt.plot(
        epochs,
        lrs_mult,
        marker="^",
        linestyle="-.",
        color="m",
        label=f"MultiplicativeLR (start=5e-5, gamma={mult_gamma})",
    )

    plt.axvline(x=frozen_epochs - 0.5, color="r", linestyle=":", label="FTF Phase Change")

    plt.title("Learning Rate Schedule Comparison (Log Scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate (log scale)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    output_path = os.path.abspath("lr_schedule_comparison.png")
    plt.savefig(output_path)
    print(f"\nLog-scale plot saved to: {output_path}")

    # Plotting Linear Scale
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs_ftf, marker="o", linestyle="-", color="b", label="FreezeThenFinetune")
    plt.plot(epochs, lrs_qcos, marker="s", linestyle="--", color="g", label="QuarterCosLR (start=5e-5)")
    plt.plot(
        epochs,
        lrs_mult,
        marker="^",
        linestyle="-.",
        color="m",
        label=f"MultiplicativeLR (start=5e-5, gamma={mult_gamma})",
    )

    plt.axvline(x=frozen_epochs - 0.5, color="r", linestyle=":", label="FTF Phase Change")

    plt.title("Learning Rate Schedule Comparison (Linear Scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate (linear scale)")
    plt.yscale("linear")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    output_path_linear = os.path.abspath("lr_schedule_comparison_linear.png")
    plt.savefig(output_path_linear)
    print(f"Linear-scale plot saved to: {output_path_linear}")


if __name__ == "__main__":
    main()

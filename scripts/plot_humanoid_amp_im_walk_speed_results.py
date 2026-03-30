"""
Plot desired vs actual speed for humanoid_amp_im_walk configs.

For each config folder, the script reads all `speed_results*.csv` files,
computes the per-file MSE between `desired_speed` and `actual_speed`, then
averages the curves across files and reports the average MSE in the legend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_speed_results(csv_path: Path) -> tuple[pd.DataFrame, float]:
    """Return (curve_df, mse) where curve_df has columns: desired_speed, actual_speed."""
    df = pd.read_csv(csv_path)

    required = {"desired_speed", "actual_speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    # If multiple rows share the same desired_speed, average actual_speed for stability.
    # (Uses rounding to avoid float representation mismatch.)
    df = df.copy()
    df["desired_speed"] = np.round(df["desired_speed"].astype(float), 6)
    df["actual_speed"] = df["actual_speed"].astype(float)
    curve = df.groupby("desired_speed", as_index=False)["actual_speed"].mean()

    mse = np.mean((curve["desired_speed"].to_numpy() - curve["actual_speed"].to_numpy()) ** 2)
    return curve, float(mse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logs/skrl/humanoid_amp_im_walk"),
        help="Root folder containing config subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/skrl/humanoid_amp_im_walk/combined_speed_desired_vs_actual_alpha_mse.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window (blocks execution).",
    )
    args = parser.parse_args()

    root_dir: Path = args.root
    if not root_dir.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root_dir}")

    configs: list[tuple[str, str]] = [
        ("05_imitation_abs_diff_entcoeff", r"$\alpha = 0.5$"),
        ("075_imitation_abs_diff_entcoeff", r"$\alpha = 0.25$"),
        (
            "05_imitation_abs_diff_ent_imweightboosted",
            r"$\alpha = 0.5$ weight boost",
        ),
        ("no_imitation_abs_diff", "imitasyon azaltmasız"),
        ("no_imitation_reward_result", "AMP"),
    ]

    per_config_curves: dict[str, dict[str, object]] = {}
    global_min_speed = None
    global_max_speed = None

    for folder_name, legend_base in configs:
        config_dir = root_dir / folder_name
        if not config_dir.exists():
            raise FileNotFoundError(f"Missing config folder: {config_dir}")

        csv_files = sorted(config_dir.glob("speed_results*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No speed_results*.csv found in: {config_dir}")

        curves: list[pd.DataFrame] = []
        mses: list[float] = []
        for csv_path in csv_files:
            curve_df, mse = _read_speed_results(csv_path)
            curves.append(curve_df)
            mses.append(mse)

        all_points = pd.concat(curves, ignore_index=True)
        mean_curve = (
            all_points.groupby("desired_speed", as_index=False)["actual_speed"].mean().sort_values("desired_speed")
        )
        avg_mse = float(np.mean(mses))

        per_config_curves[folder_name] = {
            "legend_base": legend_base,
            "mean_curve": mean_curve,
            "avg_mse": avg_mse,
        }

        min_speed = float(mean_curve["desired_speed"].min())
        max_speed = float(mean_curve["desired_speed"].max())
        global_min_speed = min_speed if global_min_speed is None else min(global_min_speed, min_speed)
        global_max_speed = max_speed if global_max_speed is None else max(global_max_speed, max_speed)

    assert global_min_speed is not None and global_max_speed is not None

    plt.figure(figsize=(10, 8))

    # Plot each config curve with avg MSE in legend.
    for folder_name, _legend_base in configs:
        entry = per_config_curves[folder_name]
        mean_curve = entry["mean_curve"]  # type: ignore[assignment]
        avg_mse = entry["avg_mse"]  # type: ignore[assignment]
        legend_base = entry["legend_base"]  # type: ignore[assignment]

        label = f"{legend_base} (mse = {avg_mse:.6f})"
        plt.plot(
            mean_curve["desired_speed"],
            mean_curve["actual_speed"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=label,
        )

    # Reference diagonal: desired_speed == actual_speed
    plt.plot(
        [global_min_speed, global_max_speed],
        [global_min_speed, global_max_speed],
        "k--",
        linewidth=2,
        alpha=0.8,
        label="İdeal (istenilen = gerçek)",
    )

    plt.xlabel("İstenilen hız")
    plt.ylabel("Gerçek hız")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {args.output}")

    # Print MSE summary for quick checking.
    for folder_name, legend_base in configs:
        avg_mse = per_config_curves[folder_name]["avg_mse"]  # type: ignore[index]
        # Use folder_name to avoid Windows console encoding issues with Turkish characters.
        print(f"{folder_name}: mse = {avg_mse:.6f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


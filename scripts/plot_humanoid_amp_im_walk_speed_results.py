"""
Plot desired vs actual speed for humanoid_amp_im_walk configs.

The script takes a folder path as input, walks every immediate subfolder,
locates the `speed_results.csv` file inside each one and produces a single
velocity figure containing every configuration found. Each curve is labeled
with its folder name (with the prefix up to and including ``torch_`` stripped).

For each configuration the script also prints the average MSE between
``desired_speed`` and ``actual_speed`` and reports whether any trial inside the
configuration failed. The summary is mirrored to a ``.txt`` file next to the
output figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _legend_from_folder(folder_name: str) -> str:
    """Strip everything up to and including the first ``torch_`` token."""
    marker = "torch_"
    idx = folder_name.find(marker)
    if idx == -1:
        return folder_name
    return folder_name[idx + len(marker):]


def _read_speed_results(csv_path: Path) -> tuple[pd.DataFrame, float, bool, int]:
    """Return (curve_df, mse, any_failed, num_failed).

    ``curve_df`` only contains successful rows; ``mse`` is computed across the
    successful rows. A trial is considered failed when its ``success`` column
    is ``False`` (or ``actual_speed`` is NaN when the column is missing).
    """
    df = pd.read_csv(csv_path)

    required = {"desired_speed", "actual_speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    df = df.copy()
    df["desired_speed"] = np.round(df["desired_speed"].astype(float), 6)
    df["actual_speed"] = pd.to_numeric(df["actual_speed"], errors="coerce")

    if "success" in df.columns:
        success_mask = df["success"].astype(str).str.lower().isin({"true", "1", "1.0"})
    else:
        success_mask = df["actual_speed"].notna()

    num_failed = int((~success_mask).sum())
    any_failed = num_failed > 0

    success_df = df[success_mask & df["actual_speed"].notna()]
    curve = success_df.groupby("desired_speed", as_index=False)["actual_speed"].mean()
    curve = curve.sort_values("desired_speed").reset_index(drop=True)

    if curve.empty:
        mse = float("nan")
    else:
        diffs = curve["desired_speed"].to_numpy() - curve["actual_speed"].to_numpy()
        mse = float(np.mean(diffs ** 2))

    return curve, mse, any_failed, num_failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("logs/skrl/humanoid_amp_im_walk_v2"),
        help="Root folder containing config subfolders (one speed_results.csv per subfolder).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to '<root>/combined_speed_desired_vs_actual.png'.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Output TXT summary path. Defaults to '<root>/speed_results_summary.txt'.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="speed_results.csv",
        help="Name of the CSV file expected inside each subfolder.",
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

    output_path: Path = args.output or (root_dir / "combined_speed_desired_vs_actual.png")
    summary_path: Path = args.summary or (root_dir / "speed_results_summary.txt")

    subfolders = sorted(p for p in root_dir.iterdir() if p.is_dir())

    per_config: list[dict[str, object]] = []
    global_min_speed: float | None = None
    global_max_speed: float | None = None

    for config_dir in subfolders:
        csv_path = config_dir / args.csv_name
        if not csv_path.exists():
            continue

        curve, mse, any_failed, num_failed = _read_speed_results(csv_path)
        if curve.empty:
            print(f"Skipping {config_dir.name}: no successful rows in {csv_path.name}")
            continue

        per_config.append(
            {
                "folder": config_dir.name,
                "legend": _legend_from_folder(config_dir.name),
                "curve": curve,
                "mse": mse,
                "any_failed": any_failed,
                "num_failed": num_failed,
            }
        )

        cmin = float(curve["desired_speed"].min())
        cmax = float(curve["desired_speed"].max())
        global_min_speed = cmin if global_min_speed is None else min(global_min_speed, cmin)
        global_max_speed = cmax if global_max_speed is None else max(global_max_speed, cmax)

    if not per_config:
        raise FileNotFoundError(
            f"No '{args.csv_name}' files found in immediate subfolders of: {root_dir}"
        )

    assert global_min_speed is not None and global_max_speed is not None

    plt.figure(figsize=(11, 8))

    for entry in per_config:
        curve = entry["curve"]  # type: ignore[assignment]
        legend = entry["legend"]  # type: ignore[assignment]
        mse = entry["mse"]  # type: ignore[assignment]
        any_failed = entry["any_failed"]  # type: ignore[assignment]

        fail_tag = " [FAILED]" if any_failed else ""
        label = f"{legend} (mse = {mse:.6f}){fail_tag}"
        plt.plot(
            curve["desired_speed"],
            curve["actual_speed"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=label,
        )

    plt.plot(
        [global_min_speed, global_max_speed],
        [global_min_speed, global_max_speed],
        "k--",
        linewidth=2,
        alpha=0.8,
        label="Ideal (desired = actual)",
    )

    plt.xlabel("Desired speed")
    plt.ylabel("Actual speed")
    plt.title("Desired vs. actual speed across configurations")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    legend_width = max((len(str(e["legend"])) for e in per_config), default=10)
    legend_width = max(legend_width, len("config"))

    header = f"{'config'.ljust(legend_width)}  {'avg_mse':>12}  {'failed':>7}  {'num_failed':>10}"
    separator = "-" * len(header)

    summary_lines: list[str] = [
        f"Root: {root_dir}",
        f"Found {len(per_config)} configurations.",
        "",
        header,
        separator,
    ]
    for entry in per_config:
        legend = str(entry["legend"]).ljust(legend_width)
        mse = float(entry["mse"])  # type: ignore[arg-type]
        failed_str = "yes" if entry["any_failed"] else "no"
        num_failed = int(entry["num_failed"])  # type: ignore[arg-type]
        summary_lines.append(f"{legend}  {mse:12.6f}  {failed_str:>7}  {num_failed:>10}")

    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print()
    print(f"Summary saved to: {summary_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

"""
Plot ramp-angle robustness results across humanoid AMP configurations.

The root folder is expected to contain one subfolder per configuration, where
each subfolder may include a ``ramp_demo_results.csv`` produced by
``scripts/reinforcement_learning/skrl/ramp_demo.py``.

For each configuration, success rate is aggregated by ramp angle:
  * success_rate(angle) = mean(success) over all rows at that angle

Then one combined figure is produced:
  * x-axis: ramp angle (deg)
  * y-axis: success rate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _legend_from_folder(folder_name: str) -> str:
    """Strip everything up to and including first ``torch_`` token."""
    marker = "torch_"
    idx = folder_name.find(marker)
    if idx == -1:
        return folder_name
    return folder_name[idx + len(marker):]


def _read_ramp_csv(csv_path: Path) -> pd.DataFrame:
    """Load and normalize one ramp_demo_results.csv."""
    df = pd.read_csv(csv_path)

    required = {"test_angle", "success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    df = df.copy()
    df["test_angle"] = pd.to_numeric(df["test_angle"], errors="coerce")
    df["success"] = df["success"].astype(str).str.lower().isin({"true", "1", "1.0"})
    df = df[df["test_angle"].notna()].reset_index(drop=True)
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate success statistics per ramp angle."""
    out = (
        df.groupby("test_angle", as_index=False)
        .agg(
            success_mean=("success", "mean"),
            success_std=("success", "std"),
            n_trials=("success", "count"),
        )
        .sort_values("test_angle")
        .reset_index(drop=True)
    )
    out["success_std"] = out["success_std"].fillna(0.0)
    return out


def _plot(per_config: list[dict], out_path: Path) -> None:
    """Create combined ramp-angle success-rate figure across configs."""
    plt.figure(figsize=(9, 6))

    for entry in per_config:
        agg = entry["agg"]
        if agg.empty:
            continue
        x = agg["test_angle"].to_numpy()
        y = agg["success_mean"].to_numpy()
        yerr = agg["success_std"].to_numpy()
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=entry["legend"],
        )

    plt.xlabel("Ramp angle (deg)")
    plt.ylabel("Success rate")
    plt.title("Ramp angle vs success rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {out_path}")


def _write_summary(per_config: list[dict], summary_path: Path) -> None:
    """Write compact text summary with per-angle stats."""
    lines: list[str] = []
    lines.append("Ramp demo evaluation summary")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Aggregation: mean and std of success over all rows per test_angle.")
    lines.append("")

    for entry in per_config:
        lines.append("-" * 72)
        lines.append(f"Config: {entry['legend']}")
        lines.append(f"  source: {entry['csv']}")
        lines.append(f"  rows in CSV: {entry['n_rows']}")
        lines.append("")

        agg = entry["agg"]
        if agg.empty:
            lines.append("  (no aggregated rows)")
            lines.append("")
            continue

        header = f"  {'angle':>7}  {'success':>8}  {'succ_sd':>8}  {'n':>5}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for _, row in agg.iterrows():
            lines.append(
                f"  {row['test_angle']:7.2f}  {row['success_mean']:8.3f}  "
                f"{row['success_std']:8.3f}  {int(row['n_trials']):5d}"
            )
        lines.append("")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary saved to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("logs/skrl/humanoid_amp_im_walk_v2"),
        help="Root folder containing config subfolders.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="ramp_demo_results.csv",
        help="CSV filename expected inside each config folder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder for figure and summary. Defaults to <root>.",
    )
    args = parser.parse_args()

    root_dir: Path = args.root
    if not root_dir.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root_dir}")

    out_dir: Path = args.out_dir or root_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_config: list[dict] = []
    for config_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        csv_path = config_dir / args.csv_name
        if not csv_path.exists():
            continue

        df = _read_ramp_csv(csv_path)
        if df.empty:
            print(f"Skipping {config_dir.name}: empty or invalid CSV.")
            continue

        agg = _aggregate(df)
        agg.to_csv(config_dir / "ramp_demo_aggregated.csv", index=False)

        per_config.append(
            {
                "folder": config_dir.name,
                "legend": _legend_from_folder(config_dir.name),
                "csv": csv_path,
                "agg": agg,
                "n_rows": len(df),
            }
        )

    if not per_config:
        raise FileNotFoundError(
            f"No '{args.csv_name}' files found in immediate subfolders of: {root_dir}"
        )

    _plot(per_config, out_dir / "ramp_angle_vs_success_rate.png")
    _write_summary(per_config, out_dir / "ramp_demo_summary.txt")


if __name__ == "__main__":
    main()

"""
Plot noisy-plane robustness results across humanoid_amp_im_walk configurations.

Mirrors the folder layout of ``plot_speed_results.py``: the root folder
contains one subfolder per configuration, each holding a
``noisy_plane_demo_results.csv`` produced by ``noisy_plane_demo.py``.

For each configuration we aggregate the runs into two paper-ready metrics
versus noise amplitude:

  * travel_range  — mean of ``x_pos`` reached before episode end
  * success_rate  — fraction of episodes with ``success == True``

Aggregation follows the Phase 1 protocol: for every
(config, noise_type, downsampled_scale, noise_amplitude, noise_seed) cell we first
average across the desired-speed sweep to get one number per seed, then
report mean ± std across seeds. Random and wave noise are reported in
separate figures. Each figure places travel range and success rate side by
side; random noise emits one figure per ``downsampled_scale`` value so terrain
resolution is not collapsed across.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _legend_from_folder(folder_name: str) -> str:
    """Strip everything up to and including the first ``torch_`` token."""
    marker = "torch_"
    idx = folder_name.find(marker)
    if idx == -1:
        return folder_name
    return folder_name[idx + len(marker):]


def _read_noisy_csv(csv_path: Path) -> pd.DataFrame:
    """Load a noisy_plane_demo_results.csv and normalise columns/dtypes."""
    df = pd.read_csv(csv_path)

    required = {"desired_speed", "actual_speed", "x_pos",
                "noise_amplitude", "noise_type", "noise_seed", "success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    df = df.copy()
    df["noise_amplitude"] = df["noise_amplitude"].astype(float).round(6)
    df["noise_seed"]      = df["noise_seed"].astype(int)
    df["noise_type"]      = df["noise_type"].astype(str)
    df["x_pos"]           = pd.to_numeric(df["x_pos"], errors="coerce")
    df["actual_speed"]    = pd.to_numeric(df["actual_speed"], errors="coerce")
    df["success"]         = df["success"].astype(str).str.lower().isin({"true", "1", "1.0"})

    # downsampled_scale optional (wave rows may have NaN); default to NaN if missing
    if "downsampled_scale" not in df.columns:
        df["downsampled_scale"] = float("nan")
    else:
        df["downsampled_scale"] = pd.to_numeric(df["downsampled_scale"], errors="coerce")
    # round to mitigate float-printing noise from CSV
    df["downsampled_scale"] = df["downsampled_scale"].round(6)

    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Two-stage aggregation: (1) collapse speed sweep within seed,
    (2) mean ± std across seeds.

    Returns one row per (noise_type, downsampled_scale, noise_amplitude) with
    columns: range_mean, range_std, success_mean, success_std, n_seeds.
    """
    # stage 1: within-seed mean of x_pos and success fraction across speeds
    per_seed = pd.DataFrame(
        df.groupby(["noise_type", "downsampled_scale", "noise_amplitude", "noise_seed"],
                   dropna=False, as_index=False)
          .agg(range_seed=("x_pos", "mean"),
               success_seed=("success", "mean"))
    )
    # stage 2: across-seed mean and std
    out = pd.DataFrame(
        per_seed.groupby(["noise_type", "downsampled_scale", "noise_amplitude"],
                         dropna=False, as_index=False)
                .agg(range_mean=("range_seed", "mean"),
                     range_std=("range_seed", "std"),
                     success_mean=("success_seed", "mean"),
                     success_std=("success_seed", "std"),
                     n_seeds=("noise_seed", "count"))
                .reset_index(drop=True)
    )
    sort_idx = np.lexsort(
        (
            out["noise_amplitude"].to_numpy(),
            out["downsampled_scale"].fillna(np.inf).to_numpy(),
            out["noise_type"].astype(str).to_numpy(),
        )
    )
    out = out.iloc[sort_idx].reset_index(drop=True)
    # std is NaN when only one seed present — fill 0 so error bars render
    out["range_std"]   = out["range_std"].fillna(0.0)
    out["success_std"] = out["success_std"].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plot_panel(ax, configs, key_mean, key_std, ylabel, title, ylim=None):
    """Draw one (configs × amplitude) panel for either metric."""
    for entry in configs:
        sub = entry["sub"]
        if sub.empty:
            continue
        # convert amplitude m -> cm for readability (matches Phase 1 paper)
        x = sub["noise_amplitude"].to_numpy() * 100.0
        y = sub[key_mean].to_numpy()
        yerr = sub[key_std].to_numpy()
        ax.errorbar(
            x, y, yerr=yerr,
            marker="o", linewidth=2, markersize=5, capsize=3,
            label=entry["legend"],
        )
    ax.set_xlabel("Noise amplitude (cm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _make_figure(per_config, noise_type, d_scale, out_dir):
    """Produce a 1×2 figure (range | success) for one (noise_type, downsampled_scale)."""
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 5),
        squeeze=False,
        sharex=True,
    )

    sliced = []
    for entry in per_config:
        agg = entry["agg"]
        if math.isnan(d_scale):
            sub = agg[(agg["noise_type"] == noise_type) & agg["downsampled_scale"].isna()]
        else:
            sub = agg[(agg["noise_type"] == noise_type) &
                      np.isclose(agg["downsampled_scale"], d_scale)]
        sliced.append({"legend": entry["legend"], "sub": sub.reset_index(drop=True)})

    scale_tag = "" if math.isnan(d_scale) else f"  (downsampled_scale = {d_scale:g} m)"
    _plot_panel(
        axes[0, 0], sliced,
        key_mean="range_mean", key_std="range_std",
        ylabel="Avg. travel range (m)",
        title=f"Travel range vs amplitude{scale_tag}",
    )
    _plot_panel(
        axes[0, 1], sliced,
        key_mean="success_mean", key_std="success_std",
        ylabel="Success rate",
        title=f"Success rate vs amplitude{scale_tag}",
        ylim=(-0.05, 1.05),
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc="upper center", ncol=min(len(labels), 4),
                   bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)

    fig.suptitle(f"Noisy plane robustness — {noise_type} noise", fontsize=13, y=1.06)
    fig.tight_layout()

    if math.isnan(d_scale):
        out_path = out_dir / f"noisy_plane_{noise_type}.png"
    else:
        out_path = out_dir / f"noisy_plane_{noise_type}_downsampled_scale_{d_scale:g}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# summary writer
# ---------------------------------------------------------------------------
def _write_summary(per_config, summary_path: Path) -> None:
    lines: list[str] = []
    lines.append("Noisy plane evaluation summary")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Aggregation: within-seed mean across desired-speed sweep,")
    lines.append("then mean ± std across noise_seeds.")
    lines.append("")

    for entry in per_config:
        legend = entry["legend"]
        agg = entry["agg"]
        lines.append("-" * 72)
        lines.append(f"Config: {legend}")
        lines.append(f"  source: {entry['csv']}")
        lines.append(f"  rows in CSV: {entry['n_rows']}")
        lines.append("")

        if agg.empty:
            lines.append("  (no aggregated rows)")
            continue

        header = (f"  {'type':>6}  {'scale':>7}  {'amp(cm)':>8}  "
                  f"{'range_m':>10}  {'range_sd':>10}  "
                  f"{'success':>8}  {'succ_sd':>8}  {'n':>3}")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for _, r in agg.iterrows():
            scale_str = "  --" if pd.isna(r["downsampled_scale"]) else f"{r['downsampled_scale']:7.3f}"
            lines.append(
                f"  {r['noise_type']:>6}  {scale_str}  "
                f"{r['noise_amplitude']*100:8.2f}  "
                f"{r['range_mean']:10.3f}  {r['range_std']:10.3f}  "
                f"{r['success_mean']:8.3f}  {r['success_std']:8.3f}  "
                f"{int(r['n_seeds']):3d}"
            )
        lines.append("")

    text = "\n".join(lines)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(f"Summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path, nargs="?",
        default=Path("logs/skrl/humanoid_amp_im_walk_v2"),
        help="Root folder containing config subfolders.",
    )
    parser.add_argument(
        "--csv-name", type=str, default="noisy_plane_demo_results.csv",
        help="Name of the CSV file expected inside each subfolder.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output folder for figures and summary. Defaults to <root>.",
    )
    args = parser.parse_args()

    root_dir: Path = args.root
    if not root_dir.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root_dir}")

    out_dir: Path = args.out_dir or root_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    subfolders = sorted(p for p in root_dir.iterdir() if p.is_dir())

    per_config: list[dict] = []
    for config_dir in subfolders:
        csv_path = config_dir / args.csv_name
        if not csv_path.exists():
            continue
        df = _read_noisy_csv(csv_path)
        if df.empty:
            print(f"Skipping {config_dir.name}: empty CSV.")
            continue
        agg = _aggregate(df)

        # save the aggregated table next to each config (handy for the paper)
        agg_path = config_dir / "noisy_plane_aggregated.csv"
        agg.to_csv(agg_path, index=False)

        per_config.append({
            "folder": config_dir.name,
            "legend": _legend_from_folder(config_dir.name),
            "csv": csv_path,
            "agg": agg,
            "n_rows": len(df),
        })

    if not per_config:
        raise FileNotFoundError(
            f"No '{args.csv_name}' files found in immediate subfolders of: {root_dir}"
        )

    # discover the set of (noise_type, downsampled_scale) combinations actually present
    all_agg = pd.concat([e["agg"] for e in per_config], ignore_index=True)

    # Random: one figure per downsampled_scale (range | success side by side)
    random_rows = all_agg[all_agg["noise_type"] == "random"]
    if not random_rows.empty:
        random_scale_series = pd.Series(random_rows["downsampled_scale"])
        random_scales = sorted(s for s in random_scale_series.dropna().unique())
        for d_scale in random_scales:
            _make_figure(per_config, "random", d_scale, out_dir)

    # Wave: one figure (range | success side by side; downsampled_scale is NaN)
    wave_rows = all_agg[all_agg["noise_type"] == "wave"]
    if not wave_rows.empty:
        _make_figure(per_config, "wave", float("nan"), out_dir)

    _write_summary(per_config, out_dir / "noisy_plane_summary.txt")


if __name__ == "__main__":
    main()
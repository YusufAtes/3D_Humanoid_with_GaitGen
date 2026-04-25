"""
gait_generators/fft_mlp_v2.py
=============================

Drop-in generator for the v2 FFT-MLP produced by
``fft_datacreate_review_v2.py``.

Key differences vs ``fft_mlp_review.py``
----------------------------------------
1. **Two-headed model** (freq + period) instead of a single 137-wide head.
   The period head is a ``softplus(Linear(H, 1))`` so the output is always
   strictly positive (in seconds) - no more z-scored period, no more risk
   of a negative integer resample count.
2. **Optional cubic-spline residual.**  When ``use_residual=True`` the
   prior is loaded from ``gait reference phase 2/spline_prior_v2.npz`` and
   added to the MLP's freq + period predictions before the IRFFT.
3. **Variant selector** via the ``variant`` arg; the generator auto-loads
   the right checkpoint name (``FINAL_BEST_MODEL_V2_<variant>.pth``).

Checkpoint paths (produced by ``fft_datacreate_review_v2.py``)
--------------------------------------------------------------
    weights        = kfold_results/FINAL_BEST_MODEL_V2_<variant>.pth
    mean / std     = gait reference phase 2/{mean_train,std_train}.npy
    spline prior   = gait reference phase 2/spline_prior_v2.npz

BREAKING CHANGE NOTICE
----------------------
This generator uses a DIFFERENT checkpoint format and architecture from
``fft_mlp``/``fft_mlp_review``.  It is registered under the new name
``fft_mlp_v2`` (see ``registry.py``).  Existing PPO runs using ``fft_mlp``
or ``fft_mlp_review`` are unaffected.

The YAML config must set ``env.gait_generator: fft_mlp_v2`` explicitly.
See ``configs/gen_fft_mlp_v2.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from scipy.interpolate import CubicSpline

from .base import BaseGaitGenerator


_REPO = Path(__file__).resolve().parents[1]

Variant = Literal["baseline", "phase", "residual", "phase_residual"]


# ---------------------------------------------------------------------------
# Model architecture MUST match fft_datacreate_review_v2.SimpleFCNN exactly.
# Redefined here to avoid importing the training script at runtime.
# ---------------------------------------------------------------------------
class _SimpleFCNNv2(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 256,
                 freq_dim: int = 136) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1),
        )
        self.freq_head   = nn.Linear(hidden_size, freq_dim)
        self.period_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.freq_head(h), F.softplus(self.period_head(h))


class FFTMLPv2Generator(BaseGaitGenerator):
    """v2 FFT-MLP with softplus period head + optional cubic-spline residual."""

    name = "fft_mlp_v2"
    _FREQ_DIM = 136

    def __init__(
        self,
        variant: Variant = "phase_residual",
        weights_path: str | Path | None = None,
        mean_path:    str | Path = _REPO / "gait reference phase 2" / "mean_train.npy",
        std_path:     str | Path = _REPO / "gait reference phase 2" / "std_train.npy",
        spline_prior_path: str | Path = _REPO / "gait reference phase 2" / "spline_prior_v2.npz",
        dt: float = 1e-3,
        tile_repeats: int = 50,
        hidden_size: int = 256,
    ) -> None:
        super().__init__(dt=dt, tile_repeats=tile_repeats)
        self.variant = variant
        self.use_residual = variant in ("residual", "phase_residual")

        if weights_path is None:
            weights_path = (_REPO / "kfold_results"
                            / f"FINAL_BEST_MODEL_V2_{variant}.pth")

        # ---- model
        self.net = _SimpleFCNNv2(input_size=3, hidden_size=hidden_size,
                                 freq_dim=self._FREQ_DIM)
        self.net.load_state_dict(torch.load(weights_path, weights_only=True))
        self.net.eval()

        # ---- norm stats
        self.mean_perbin = np.load(mean_path).astype(np.float32)         # (136,)
        self.std_global  = float(np.load(std_path).reshape(-1)[0])

        # ---- spline prior (optional)
        self._freq_cs = self._period_cs = None
        if self.use_residual:
            d = np.load(spline_prior_path)
            ks = d["knot_speeds"].astype(np.float64)
            self._freq_cs   = CubicSpline(ks, d["freq_knots"].astype(np.float64),
                                          axis=0, bc_type="natural", extrapolate=True)
            self._period_cs = CubicSpline(ks, d["period_knots"].astype(np.float64),
                                          bc_type="natural", extrapolate=True)
            self._speed_min = float(ks[0])
            self._speed_max = float(ks[-1])

    # ------------------------------------------------------------------
    def _input_vec(self, speed: float, leg_lengths: Sequence[float]) -> torch.Tensor:
        v = np.empty(3, dtype=np.float32)
        v[0] = float(speed) / 2.4
        v[1] = float(leg_lengths[0])
        v[2] = float(leg_lengths[1])
        return torch.tensor(v, dtype=torch.float32).unsqueeze(0)

    def _apply_prior(self, speed: float,
                     pred_freq_norm: np.ndarray,
                     pred_period_s: float) -> tuple[np.ndarray, float]:
        if not self.use_residual:
            return pred_freq_norm, pred_period_s
        s = float(np.clip(speed, self._speed_min, self._speed_max))
        prior_freq_denorm = self._freq_cs(s).astype(np.float32)           # (136,)
        prior_period      = float(self._period_cs(s))
        prior_freq_norm   = (prior_freq_denorm - self.mean_perbin) / self.std_global
        return pred_freq_norm + prior_freq_norm, pred_period_s + prior_period

    def _denormalize(self, freq_norm: np.ndarray) -> np.ndarray:
        freq = freq_norm * self.std_global + self.mean_perbin            # (136,)
        return freq.reshape(17, 4, 2)

    def _ifft_to_6joints(self, freq_174x2: np.ndarray, period: float) -> np.ndarray:
        complex_pred = freq_174x2[..., 0] + 1j * freq_174x2[..., 1]      # (17, 4)
        time4 = np.fft.irfft(complex_pred, n=32, axis=0)                  # (32, 4)

        # 4 -> 6 (ankles zero; no mocap)
        time6 = np.zeros((time4.shape[0], 6), dtype=np.float64)
        time6[:, 0] = time4[:, 0]   # rhip
        time6[:, 1] = time4[:, 1]   # rknee
        time6[:, 3] = time4[:, 2]   # lhip
        time6[:, 4] = time4[:, 3]   # lknee

        # Resample to control rate. Period is ALWAYS positive now (softplus).
        n_samples = max(2, int(round(period / self.dt)))
        if n_samples != time6.shape[0]:
            time6 = resample(time6, n_samples, axis=0)
        return np.tile(time6, (self.tile_repeats, 1))

    # ------------------------------------------------------------------
    def predict(self, speed: float, leg_lengths: Sequence[float]) -> np.ndarray:
        x = self._input_vec(speed, leg_lengths)
        with torch.no_grad():
            pf_raw, pp = self.net(x)
        pred_freq_norm = pf_raw.cpu().numpy()[0]                          # (136,)
        pred_period_s  = float(pp.cpu().numpy()[0, 0])

        pred_freq_norm, pred_period_s = self._apply_prior(
            speed, pred_freq_norm, pred_period_s
        )
        freq_174x2 = self._denormalize(pred_freq_norm)
        return self._ifft_to_6joints(freq_174x2, pred_period_s)


__all__ = ["FFTMLPv2Generator"]
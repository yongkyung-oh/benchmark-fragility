"""Fragility thresholds and the violation convention.

A pair *violates* a test when its metric is at or below the threshold
(``metric <= tau``). A pair is *fragile* if it violates **any** of the three.

Defaults follow lenient, literature-standard cut-offs:
- ``tau_w = 0.60`` — win rate (Bouthillier et al. 2021 use 0.75; 0.60 is laxer).
- ``tau_d = 0.20`` — Cohen's d, the minimum "small" effect (Cohen 1988).
- ``tau_b = 0.20`` — breakdown-point ratio, the robust-statistics fragility line.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Thresholds"]


@dataclass(frozen=True)
class Thresholds:
    tau_w: float = 0.60  # consistency (win rate)
    tau_d: float = 0.20  # magnitude (Cohen's d)
    tau_b: float = 0.20  # stability (breakdown-point ratio)

    def violations(self, win_rate: float, cohens_d: float, bp_ratio: float
                   ) -> tuple[bool, bool, bool, bool]:
        """Return ``(consistency, magnitude, stability, fragile)`` flags."""
        viol_consistency = win_rate <= self.tau_w
        viol_magnitude = cohens_d <= self.tau_d
        viol_stability = bp_ratio <= self.tau_b
        fragile = viol_consistency or viol_magnitude or viol_stability
        return viol_consistency, viol_magnitude, viol_stability, fragile

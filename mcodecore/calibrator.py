"""Token estimation calibrator.

Standalone module with no dependency on context/compact, avoiding circular imports.
"""

from __future__ import annotations

import threading


class TokenCalibrator:
    """Sliding-window token estimation calibrator.

    Dynamically corrects the estimation factor using the median of
    ``actual_prompt_tokens / estimated``.
    """

    def __init__(self, window: int = 50):
        self.samples: list[tuple[int, int]] = []
        self.calibration_factor: float = 1.0
        self._window = window
        self._lock = threading.Lock()

    def record(self, estimated: int, actual_prompt_tokens: int) -> None:
        """Record an (estimated, actual) sample and refit."""
        if estimated > 1000 and actual_prompt_tokens > 0:
            with self._lock:
                self.samples.append((estimated, actual_prompt_tokens))
                if len(self.samples) > self._window:
                    self.samples.pop(0)
                self._refit()

    def _refit(self) -> None:
        ratios = [a / e for e, a in self.samples if e > 1000]
        if ratios:
            self.calibration_factor = sorted(ratios)[len(ratios) // 2]

    def calibrated(self, estimated: int) -> int:
        """Return the calibrated token estimate."""
        with self._lock:
            factor = self.calibration_factor
        return int(estimated * factor)

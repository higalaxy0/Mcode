"""Token calibrator tests.

Covers ``mcodecore.calibrator.TokenCalibrator``:
sliding-window samples, median calibration factor, and record/calibrated behavior.
"""

from __future__ import annotations

from mcodecore.calibrator import TokenCalibrator


def test_default_calibration_factor_one():
    c = TokenCalibrator()
    assert c.calibration_factor == 1.0
    assert c.calibrated(100) == 100


def test_record_updates_factor():
    c = TokenCalibrator()
    c.record(2000, 1000)  # actual=1000, est=2000 -> ratio 0.5 (requires est>1000)
    assert c.calibration_factor < 1.0


def test_calibrated_applies_factor():
    c = TokenCalibrator()
    c.calibration_factor = 0.5
    assert c.calibrated(200) == 100


def test_sliding_window_limits_samples():
    c = TokenCalibrator(window=5)
    for i in range(20):
        c.record(2000, 200)  # all ratio 0.1 (est>1000)
    assert len(c.samples) <= 5


def test_factor_is_median_of_samples():
    c = TokenCalibrator(window=5)
    c.record(2000, 1000)  # 0.5
    c.record(2000, 600)   # 0.3
    c.record(2000, 1400)  # 0.7
    # median = 0.5
    assert abs(c.calibration_factor - 0.5) < 0.01


def test_zero_estimate_does_not_crash():
    c = TokenCalibrator()
    c.record(0, 0)
    # should not raise
    assert isinstance(c.calibration_factor, float)


def test_calibrated_with_no_samples():
    c = TokenCalibrator()
    assert c.calibrated(42) == 42

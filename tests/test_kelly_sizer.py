"""Kelly sizer tests."""

from strategies.kelly_sizer import combined_size, kelly_fraction, size_from_confidence


def test_kelly_fraction_zero_when_bad_prob():
    assert kelly_fraction(-0.1, 1.5) == 0.0


def test_kelly_fraction_positive_with_edge():
    f = kelly_fraction(0.55, 1.5, max_fraction=0.5, min_fraction=0.0)
    assert f >= 0.0


def test_size_from_confidence_hold():
    assert size_from_confidence(0.9, signal="hold") == 0.0


def test_combined_size_respects_cap():
    x = combined_size(0.6, 1.2, 0.9, signal="buy", max_fraction=0.15)
    assert 0.0 <= x <= 0.15

"""Assert the scored path never references a blocked/bias-proxy feature."""

from pathlib import Path

from airecruiter.fairness import (
    SCORED_SOURCE_MODULES,
    assert_no_blocked_features,
)

PKG = Path(__file__).resolve().parents[1] / "airecruiter"


def test_scored_path_has_no_blocked_features():
    all_violations = []
    for mod in SCORED_SOURCE_MODULES:
        src = (PKG / mod).read_text(encoding="utf-8")
        all_violations += assert_no_blocked_features(src, mod)
    assert not all_violations, "Blocked features referenced:\n" + "\n".join(all_violations)


def test_guard_detects_a_planted_violation():
    bad = "def s(rec):\n    return rec['education'][0]['tier']\n"
    assert assert_no_blocked_features(bad, "fake.py"), "guard should catch a tier reference"

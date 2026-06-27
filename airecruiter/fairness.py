"""Fairness by design.

We name the fields that must never influence rank, and provide a guard that the
test suite uses to assert the scorer never reads them. The principle: rank on
demonstrated, job-relevant evidence — not on identity or prestige proxies.

Note on location: the target role is an explicitly hybrid Pune/Noida position and
the JD itself lists in-scope cities, so *city / relocation-willingness* is treated
as a bona-fide occupational requirement, used only as a small tie-shaper. We do
NOT use `country`/nationality as an ethnicity proxy, nor education tier/grade,
name, gender, age, or graduation year.
"""

from __future__ import annotations

# Fields that are off-limits as ranking signals. The names map to keys in the
# candidate record; the guard below also forbids these substrings appearing as
# attribute access in the scoring source.
BLOCKED_FEATURES = frozenset(
    {
        "anonymized_name",
        "tier",            # education prestige tier — explicit bias proxy
        "grade",           # GPA/percentage — proxy, not job-relevant here
        "institution",     # college name — prestige proxy
        "gender",
        "age",
        "birth_year",
        "marital_status",
        "religion",
        "nationality",
    }
)

# Modules that constitute the scored path. The fairness test scans these for any
# textual reference to a blocked feature.
SCORED_SOURCE_MODULES = (
    "features.py",
    "honeypot.py",
    "score.py",
    "jd.py",
)


def assert_no_blocked_features(source_text: str, module_name: str = "") -> list[str]:
    """Return a list of blocked features referenced in `source_text` (empty = ok).

    We look for the field name used as a dict key or attribute, e.g. `["tier"]`,
    `.tier`, `'grade'`. Comments are stripped first so documentation that merely
    *names* a blocked field (to explain why it is excluded) does not trip the
    guard.
    """
    violations: list[str] = []
    lines = []
    for raw in source_text.splitlines():
        # Drop everything after the first unquoted '#'. Simple but sufficient:
        # our source never puts '#' inside string literals on scored-path lines.
        code = raw.split("#", 1)[0]
        lines.append(code)
    code_only = "\n".join(lines)
    for feat in BLOCKED_FEATURES:
        for token in (f'["{feat}"]', f"['{feat}']", f".{feat}", f'"{feat}"', f"'{feat}'"):
            if token in code_only:
                violations.append(f"{module_name}: references blocked feature via {token}")
                break
    return violations

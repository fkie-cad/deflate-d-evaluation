"""Apply DEFLATE-D tiers to decompiler output and expose the placeholder map.

The cumulative tiers T0--T4 come from :mod:`deflated.transform`. For the
variable-naming task we additionally need the placeholder->short-name mapping
that T3's ``CompressPlaceholderNames`` produced, so a prediction on the
compressed code (where ``v4`` became, say, ``a``) can be mapped back to the
original placeholder identity that the DWARF ground truth is keyed by.

``tiered_versions`` returns, for one function, the five rendered strings plus
the placeholder map for each tier (empty for tiers below T3).
"""

from __future__ import annotations

from deflated.transforms import build_pipeline
from deflated.transforms.contextual import CompressPlaceholderNames

TIERS = ("T0", "T1", "T2", "T3", "T4")


def _pipeline_for(tier: str):
    return build_pipeline(tier, include_stubs=True)


def tiered_versions(raw: str) -> dict[str, dict]:
    """Render ``raw`` at every tier.

    Returns ``{tier: {"code": str, "placeholder_map": {orig: short}}}``. The map
    is the *original->compressed* placeholder rename; invert it to go from a
    compressed name back to the original placeholder.
    """
    out: dict[str, dict] = {}
    for tier in TIERS:
        if tier == "T0":
            out[tier] = {"code": raw, "placeholder_map": {}}
            continue
        pipe = _pipeline_for(tier)
        code = pipe.apply(raw)
        # Find the placeholder transform instance (only present at T3+) and read
        # the map it recorded during ``apply``.
        pmap: dict[str, str] = {}
        for t in pipe.transforms:
            if isinstance(t, CompressPlaceholderNames):
                pmap = dict(t.last_mapping)
                break
        out[tier] = {"code": code, "placeholder_map": pmap}
    return out
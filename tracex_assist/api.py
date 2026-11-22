"""One import point for the dashboard and llm.py (Member 3).

Uses Member 1's profile.py and Member 2's evidence.py when they exist, otherwise the stub. `USING_STUB` tells
the UI to label its numbers as placeholders.
"""
from __future__ import annotations

try:
    from .profile import build_profile, demo_users, place_index  # Member 1
    _profile_stub = False
except ImportError:
    from .stub import build_profile, demo_users, place_index
    _profile_stub = True

try:
    from .evidence import evidence_to_dict, gather  # Member 2
    _evidence_stub = False
except ImportError:
    from .stub import gather
    from .contract import to_dict as evidence_to_dict
    _evidence_stub = True

USING_STUB = {"profile": _profile_stub, "evidence": _evidence_stub}

__all__ = ["build_profile", "demo_users", "place_index", "gather", "evidence_to_dict", "USING_STUB"]

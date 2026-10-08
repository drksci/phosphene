"""End-to-end: asciicast -> keyframes -> VTM roles -> incremental A2UI JSONL (+ stats)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from vtcore.cast import Cast
from vtcore.emulator import keyframes
from vtcore.frame import Frame
from vttui.reconcile import Reconciler, msg_bytes


def cast_to_a2ui(cast: Cast, labeler: Callable[[Frame], np.ndarray], max_frames: int = 400) -> tuple[list[list[dict]], list[dict]]:
    """``labeler`` is any frame -> role-map function (trained VTM, heuristics, or gold labels).

    Returns per-keyframe message lists and per-keyframe stats comparing the A2UI delta stream with
    the raw VT bytes and a naive full re-render."""
    rec = Reconciler()
    out, stats = [], []
    prev_t, vt_bytes = 0.0, 0
    evs = iter(e for e in cast.events if e.code == "o")
    pending = next(evs, None)
    for f in keyframes(cast, max_frames=max_frames):
        while pending is not None and pending.t <= f.t:  # raw VT bytes that produced this keyframe
            vt_bytes += len(pending.data.encode("utf-8", "replace"))
            pending = next(evs, None)
        roles = labeler(f)
        msgs = rec.step(f, roles)
        full = Reconciler().step(f, rec.roles)
        out.append(msgs)
        stats.append({"t": f.t, "vt_bytes": vt_bytes, "a2ui_bytes": msg_bytes(msgs), "full_bytes": msg_bytes(full),
                      "n_msgs": len(msgs), "structural": any("updateComponents" in m for m in msgs)})
        vt_bytes = 0
        prev_t = f.t
    return out, stats

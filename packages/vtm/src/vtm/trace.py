"""Export a recording's trip through the pipeline as a compact JSON trace for the web player.

Per keyframe: cell deltas (what changed on screen), role deltas (what the matcher/template said),
gateway state, the A2UI messages actually emitted and the byte counts at each stage. The player
redraws the terminal from cell deltas, so it never needs a VT parser.
"""

from __future__ import annotations

import json
import re
from typing import Callable

import numpy as np

from vtcore import Cast, keyframes
from vtcore.frame import Frame
from vttui import Gateway, Reconciler, TemplateCache, msg_bytes

# identities: user and host names are collected from the whole recording first (prompts, user@host,
# home paths), then every occurrence is replaced at the same width, in screen text and in A2UI
_ID_PATTERNS = [
    re.compile(r"\b([A-Za-z][\w.-]{1,31})@([A-Za-z][\w.-]{1,63})"),
    re.compile(r"/(?:home|Users)/([\w.-]{2,32})"),
    re.compile(r"(?m)^\$?\s*([a-z][\w.-]{2,31}) [~/]"),  # zsh/omz style "name ~/path"
]
_KEEP = {"root", "user", "host", "localhost", "home", "users", "www", "git", "admin", "ubuntu", "debian", "docker"}


def find_identities(text: str) -> list[str]:
    names = set()
    for rx in _ID_PATTERNS:
        for m in rx.finditer(text):
            names.update(g for g in m.groups() if g)
    names = {n for n in names if n.lower() not in _KEEP and len(n) >= 3 and not n.isdigit()}
    return sorted(names, key=len, reverse=True)


class Masker:
    def __init__(self, names: list[str]):
        self.rx = re.compile("|".join(re.escape(n) for n in names)) if names else None

    def text(self, s: str) -> str:
        if not self.rx:
            return s
        return self.rx.sub(lambda m: ("user" + "x" * len(m.group(0)))[: len(m.group(0))] if len(m.group(0)) <= 4
                           else ("user" + "·" * len(m.group(0)))[: len(m.group(0))], s)

    def frame(self, f: Frame) -> Frame:
        if not self.rx:
            return f
        cp = f.cp.copy()
        for r in range(cp.shape[0]):
            row = f.row_text(r)
            new = self.text(row)
            if new != row:
                cp[r] = [ord(ch) if cp[r, i] != 0 else 0 for i, ch in enumerate(new)]
        return Frame(cp, f.fg, f.bg, f.attr, f.cursor, f.t, f.damage, f.rec_id, dict(f.meta))


def _cells(f: Frame, mask: np.ndarray) -> list[list]:
    """Changed cells as [row, col, char, fg, bg, attr] (char "" for wide-char continuations)."""
    rs, cs = np.nonzero(mask)
    return [[int(r), int(c), chr(f.cp[r, c]) if f.cp[r, c] else "", int(f.fg[r, c]), int(f.bg[r, c]), int(f.attr[r, c])]
            for r, c in zip(rs, cs)]


def _role_runs(roles: np.ndarray, prev: np.ndarray | None) -> list[list[int]]:
    """Changed role cells as horizontal runs [row, col, length, role]."""
    diff = np.ones(roles.shape, bool) if prev is None or prev.shape != roles.shape else roles != prev
    out = []
    for r in np.nonzero(diff.any(1))[0]:
        c, w = 0, roles.shape[1]
        while c < w:
            if not diff[r, c]:
                c += 1
                continue
            e = c + 1
            while e < w and diff[r, e] and roles[r, e] == roles[r, c]:
                e += 1
            out.append([int(r), c, e - c, int(roles[r, c])])
            c = e
    return out


def export_trace(cast: Cast, segmenter: Callable[[Frame], np.ndarray], title: str = "", max_frames: int = 240,
                 cache: TemplateCache | None = None) -> dict:
    gw = Gateway(segmenter, cache=cache)
    mask = Masker(find_identities("".join(e.data for e in cast.events if e.code == "o")))
    evs = iter(e for e in cast.events if e.code == "o")
    pend = next(evs, None)
    prev_f: Frame | None = None
    prev_roles = None
    frames = []
    for f in keyframes(cast, max_frames=max_frames):
        vt = 0
        while pend is not None and pend.t <= f.t:
            vt += len(pend.data.encode("utf-8", "replace"))
            pend = next(evs, None)
        f = mask.frame(f)
        calls_before = gw.stats.model_calls
        msgs = gw.step(f)
        roles = gw.rec.roles
        full = msg_bytes(Reconciler().step(f, roles))
        changed = np.ones(f.shape, bool) if prev_f is None or prev_f.shape != f.shape else (
            (prev_f.cp != f.cp) | (prev_f.fg != f.fg) | (prev_f.bg != f.bg) | (prev_f.attr != f.attr))
        shown = [json.loads(mask.text(json.dumps(m, ensure_ascii=False))) for m in msgs]
        frames.append({
            "t": round(f.t, 3), "shape": list(f.shape), "cursor": list(f.cursor),
            "cells": _cells(f, changed), "roles": _role_runs(roles, prev_roles),
            "state": gw.state, "model": gw.stats.model_calls > calls_before,
            "vt": vt, "a2ui": msg_bytes(msgs), "full": full,
            "structural": any("updateComponents" in m for m in msgs), "msgs": shown,
        })
        prev_f, prev_roles = f, roles.copy()
    return {"title": title, "cols": cast.cols, "rows": cast.rows, "duration": round(frames[-1]["t"] if frames else 0, 3),
            "stats": gw.stats.summary(), "frames": frames}

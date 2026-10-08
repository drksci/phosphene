"""Progressive TUI -> A2UI gateway.

The client only ever renders A2UI; VT parsing and understanding stay on the server. The learned
segmenter is a *matcher*, called only when the layout is new; per-frame work afterwards is a
deterministic slice of the cell grid against a locked template, cheap enough for WASM.

    RAW ──settled keyframe──▶ MATCHING ──regions stable for N keyframes──▶ LOCKED
     ▲      (template cache hit skips straight to LOCKED, no model call)     │
     └──────── resize ◀──────────── DRIFT (ink outside slots / frame broken) ┘
                                     drifted cells go back to raw, re-match in the background

* Partial locking: regions that are stable lock individually; everything else is streamed as a
  ``Terminal`` component (styled cell runs, not VT bytes), so the UI upgrades piece by piece.
* Locked frames emit ``updateDataModel`` patches only (content), never re-layout.
* ``action()`` maps A2UI events from the client back to keystrokes for the pty.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from vtcore.encode import layout_signature
from vtcore.frame import DEFAULT_COLOR, REVERSE, Frame
from vtcore.roles import ROLE_ID

from .compile import LISTISH, RAW, Region, build_tree
from .keys import edit_bytes, key_bytes, select_bytes
from .reconcile import Reconciler, msg_bytes

R = ROLE_ID
SELECTABLE = (R["menu_item"], R["selected"])
FILL = (R["status_bar"], R["progress"])

Segmenter = Callable[[Frame], np.ndarray]


@dataclass
class Template:
    slots: np.ndarray  # (H, W) role a cell takes when it has ink (0 = unclaimed)
    fill: np.ndarray  # (H, W) bool: blank cells that still take the slot role (bars, gauges)
    hits: int = 0


@dataclass
class TemplateCache:
    """Layout fingerprint -> template. Persist it per app and repeat sessions lock without the model."""
    items: dict[tuple, Template] = field(default_factory=dict)

    @staticmethod
    def key(f: Frame) -> tuple:
        return (f.shape, layout_signature(f))

    def get(self, f: Frame) -> Template | None:
        t = self.items.get(self.key(f))
        if t is not None:
            t.hits += 1
        return t

    def put(self, f: Frame, t: Template) -> None:
        self.items[self.key(f)] = t


def _leaves(node: Region) -> list[Region]:
    out = []
    for ch in node.children:
        out.append(ch)
        if ch.children:
            out.extend(_leaves(ch))
    return out


def region_keys(f: Frame, roles: np.ndarray) -> dict[tuple, Region]:
    """Identity of each region across frames: (role, top, left). Bottoms may grow (logs, lists)."""
    return {(r.role, r.top, r.left): r for r in _leaves(build_tree(f, roles)) if r.role != RAW}


def make_template(f: Frame, roles: np.ndarray, keep: set[tuple] | None = None) -> Template:
    """Slot map from a role map. Only regions in ``keep`` are included (partial locking)."""
    h, w = f.shape
    slots = np.zeros((h, w), np.int8)
    fill = np.zeros((h, w), bool)
    regs = region_keys(f, roles)
    order = sorted(regs.items(), key=lambda kv: kv[1].role != R["border"])  # boxes first, content on top
    for key, reg in order:
        if keep is not None and key not in keep:
            continue
        t, b, l, r = reg.top, reg.bottom, reg.left, reg.right
        if reg.role == R["border"]:
            box = roles[t:b + 1, l:r + 1] == R["border"]
            slots[t:b + 1, l:r + 1][box] = R["border"]
            continue
        if reg.role in LISTISH or reg.role in SELECTABLE:
            # room to grow: claim empty rows below (and columns to the right) until something else is there
            while b + 1 < h and not slots[b + 1, l:r + 1].any() and not (roles[b + 1, l:r + 1] != 0).any():
                b += 1
            while r + 1 < w and not slots[t:b + 1, r + 1].any() and not (roles[t:b + 1, r + 1] != 0).any():
                r += 1
        area = roles[t:b + 1, l:r + 1]
        sl = slots[t:b + 1, l:r + 1]
        sl[:] = np.where(area != 0, area, reg.role)
        if reg.role in FILL:
            fill[t:b + 1, l:r + 1] = True
    return Template(slots, fill)


def apply_template(t: Template, f: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic per-frame roles from a locked template; returns (roles, drift mask)."""
    ink = (f.cp != 32) & (f.cp != 0)
    roles = np.where(ink | t.fill, t.slots, 0).astype(np.int8)
    drift = ink & (t.slots == 0)
    if f.damage is not None:  # a frame edge changing means the layout moved
        drift |= f.damage & (t.slots == R["border"]) & ~np.isin(f.cp, list(map(ord, "─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬+-|=")))
    # selection moves within a list: re-derive it from highlight attributes every frame
    hl = ((f.attr & REVERSE) > 0) | (f.bg != DEFAULT_COLOR)
    sel_slot = np.isin(t.slots, SELECTABLE)
    for row in np.nonzero(sel_slot.any(1))[0]:
        cols = sel_slot[row]
        on = hl[row, cols].mean() > 0.5
        target = R["selected"] if on else R["menu_item"]
        roles[row, cols] = np.where(ink[row, cols] | on, target, 0)
    return roles, drift


@dataclass
class Stats:
    keyframes: int = 0
    model_calls: int = 0
    cache_hits: int = 0
    locked: int = 0  # keyframes served fully from a template (no model, no raw)
    partial: int = 0  # some regions locked, rest raw
    raw: int = 0  # whole screen raw
    drift: int = 0
    structural_msgs: int = 0  # keyframes that needed updateComponents
    a2ui_bytes: int = 0

    def summary(self) -> dict:
        n = max(self.keyframes, 1)
        return {**self.__dict__, "locked_frac": round(self.locked / n, 3), "model_per_keyframe": round(self.model_calls / n, 3)}


class Gateway:
    def __init__(self, segmenter: Segmenter, cache: TemplateCache | None = None, lock_after: int = 2,
                 surface_id: str = "term"):
        self.segment = segmenter
        self.cache = cache if cache is not None else TemplateCache()
        self.lock_after = lock_after
        self.rec = Reconciler(surface_id, stabilize_roles=False)
        self.template: Template | None = None
        self.history: deque[set[tuple]] = deque(maxlen=lock_after)
        self.state = "raw"
        self.shape: tuple[int, int] | None = None
        self.stats = Stats()

    # -- server: frames in, A2UI out ------------------------------------------------
    def step(self, f: Frame) -> list[dict]:
        s = self.stats
        s.keyframes += 1
        if f.shape != self.shape:
            self.shape, self.template, self.state = f.shape, None, "raw"
            self.history.clear()
        roles = None
        if self.state == "locked":
            roles, drift = apply_template(self.template, f)
            if drift.any():
                s.drift += 1
                self.state = "matching"
                self.history.clear()
                roles[drift] = RAW  # only the drifted cells fall back; the rest stays locked this frame
                self._record(f)  # re-match (in production: asynchronously, off the frame path)
                s.partial += 1
            else:
                s.locked += 1
                self.cache.put(f, self.template)  # every locked variant of the screen becomes a cache key
        if roles is None:
            cached = self.cache.get(f)
            if cached is not None:
                r, drift = apply_template(cached, f)
                if not drift.any():
                    s.cache_hits += 1
                    self.template, self.state, roles = cached, "locked", r
                    s.locked += 1
            if roles is None:
                roles = self._observe(f)
        msgs = self.rec.step(f, roles)
        s.a2ui_bytes += msg_bytes(msgs)
        s.structural_msgs += any("updateComponents" in m for m in msgs)
        return msgs

    def _record(self, f: Frame) -> tuple[np.ndarray, dict]:
        pred = np.asarray(self.segment(f), np.int8)
        self.stats.model_calls += 1
        regs = region_keys(f, pred)
        self.history.append(set(regs))
        return pred, regs

    def _observe(self, f: Frame) -> np.ndarray:
        """Run the matcher, lock the regions that have been stable for ``lock_after`` keyframes."""
        s = self.stats
        pred, regs = self._record(f)
        stable = set.intersection(*self.history) if len(self.history) == self.lock_after else set()
        ink = (f.cp != 32) & (f.cp != 0)
        if stable and stable == set(regs):  # everything stable: lock the whole screen
            self.template = make_template(f, pred)
            self.cache.put(f, self.template)
            self.state = "locked"
            s.locked += 1
            return apply_template(self.template, f)[0]
        self.state = "matching"
        if not stable:
            s.raw += 1
            return np.where(ink, RAW, 0).astype(np.int8)
        part = make_template(f, pred, keep=stable)
        roles, drift = apply_template(part, f)
        roles[drift] = RAW
        s.partial += 1
        return roles

    # -- client: A2UI actions in, keystrokes out -------------------------------------
    def action(self, event: dict) -> str:
        name, ctx = event.get("name"), event.get("context", {})
        data = self.rec.data.get("r", {})
        if name == "key":
            return key_bytes(str(ctx.get("key", "")))
        if name == "select":
            rows = data.get(ctx.get("list"), {}).get("rows", [])
            cur = next((i for i, r in enumerate(rows) if r.get("selected")), None)
            return select_bytes(cur, int(ctx.get("row", 0)), bool(ctx.get("activate")))
        if name == "edit":
            old = data.get(ctx.get("field"), {}).get("value", "")
            return edit_bytes(old, str(ctx.get("value", "")), bool(ctx.get("submit")))
        return ""

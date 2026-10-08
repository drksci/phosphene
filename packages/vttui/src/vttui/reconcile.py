"""Reconciler: keeps the last emitted UI and streams only what changed, as A2UI v0.9 messages.

  frame N ──compile──▶ (components, data) ──diff vs N-1──▶ updateComponents (changed structure only)
                                                     └──▶ updateDataModel  (changed leaf paths only)

Role maps are stabilised first: cells the VT stream didn't touch (damage == False) keep last frame's
role, so model jitter on unchanged content can never cause UI churn.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vtcore.frame import Frame

from .compile import compile_frame
from .style import stylesheet, theme

VERSION = "v0.9.1"
CATALOG_ID = "https://github.com/drksci/phosphene/blob/main/docs/catalog/vttui_v0_1.json"
_DELETE = object()


def _esc(k: str) -> str:
    return str(k).replace("~", "~0").replace("/", "~1")


def diff_data(old: Any, new: Any, path: str = "") -> list[tuple[str, Any]]:
    """Minimal JSON-pointer patches turning ``old`` into ``new`` (upserts + deletes)."""
    if old == new:
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in new:
            if k not in old:
                out.append((f"{path}/{_esc(k)}", new[k]))
            else:
                out.extend(diff_data(old[k], new[k], f"{path}/{_esc(k)}"))
        out.extend((f"{path}/{_esc(k)}", _DELETE) for k in old if k not in new)
        # many changed children: one replace of the parent is smaller and atomic
        if len(out) > 1 and len(out) >= max(4, len(new) // 2):
            return [(path or "/", new)]
        return out
    if isinstance(old, list) and isinstance(new, list):
        n = min(len(old), len(new))
        out = []
        for i in range(n):
            out.extend(diff_data(old[i], new[i], f"{path}/{i}"))
        if len(new) < len(old) or len(out) > max(3, n // 2):
            return [(path, new)]  # shrink or heavy churn: replace the list
        out.extend((f"{path}/{i}", new[i]) for i in range(n, len(new)))  # appends
        return out
    return [(path or "/", new)]


def stabilize(prev_roles: np.ndarray | None, roles: np.ndarray, f: Frame) -> np.ndarray:
    if prev_roles is None or prev_roles.shape != roles.shape or f.damage is None:
        return roles
    out = roles.copy()
    keep = ~f.damage
    out[keep] = prev_roles[keep]
    return out


class Reconciler:
    def __init__(self, surface_id: str = "term", stabilize_roles: bool = True, styles: bool = True):
        self.surface_id = surface_id
        self.components: dict[str, dict] = {}
        self.data: dict = {}
        self.roles: np.ndarray | None = None
        self.shape: tuple[int, int] | None = None
        self.stabilize_roles = stabilize_roles
        self.styles = styles

    def _msg(self, kind: str, body: dict) -> dict:
        return {"version": VERSION, kind: {"surfaceId": self.surface_id, **body}}

    def step(self, f: Frame, roles: np.ndarray) -> list[dict]:
        if self.stabilize_roles:
            roles = stabilize(self.roles, roles, f)
        self.roles = roles
        comps, data = compile_frame(f, roles)
        if not self.styles:
            data.pop("s", None)
            for c in comps.values():
                if isinstance(c.get("style"), dict) and set(c["style"]) == {"path"}:
                    c.pop("style")
        msgs: list[dict] = []
        if self.shape != f.shape:  # first frame or resize: (re)create the surface with the theme
            body = {"catalogId": CATALOG_ID}
            if self.styles:
                body["theme"] = theme(f)
            msgs.append(self._msg("createSurface", body))
            self.components, self.data, self.shape = {}, {}, f.shape
        # data first so freshly bound components never render empty
        for path, value in diff_data(self.data, data):
            msgs.append(self._msg("updateDataModel", {"path": path} if value is _DELETE else {"path": path, "value": value}))
        changed = [c for cid, c in comps.items() if self.components.get(cid) != c]
        if changed:
            # root last: children exist before the tree that references them is re-rendered
            changed.sort(key=lambda c: c["id"] == "root")
            msgs.append(self._msg("updateComponents", {"components": changed}))
        self.components, self.data = comps, data
        return msgs

    def stylesheet(self) -> str:
        return stylesheet(self.components, theme(_shape_frame(self.shape)), self.data) if self.shape else ""


def _shape_frame(shape):
    h, w = shape
    z = np.zeros((h, w), np.uint8)
    return Frame(np.zeros((h, w), np.int32), z, z, z)


def msg_bytes(msgs: list[dict]) -> int:
    return sum(len(json.dumps(m, ensure_ascii=False, separators=(",", ":")).encode()) + 1 for m in msgs)


def to_jsonl(msgs: list[dict]) -> str:
    return "".join(json.dumps(m, ensure_ascii=False, separators=(",", ":")) + "\n" for m in msgs)

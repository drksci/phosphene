"""Deterministic compiler: (Frame, role map) -> A2UI component tree + data model.

Structure (components) and content (data model) are deliberately separated: components reference
content through JSON-Pointer bindings, so when only text changes, the reconciler streams small
``updateDataModel`` patches and the component tree is untouched — the same split as a virtual DOM
(structure) vs. reactive signals (content).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from vtcore.frame import Frame
from vtcore.roles import ROLE_ID, ROLES

from .style import line_style, region_style

R = ROLE_ID
LINE_ROLES = {R[r] for r in ("text", "log", "error", "code", "table", "menu_item", "selected")}
LISTISH = {R[r] for r in ("menu_item", "table", "text", "code", "log")}
TEXT_VARIANT = {"text": "body", "log": "code", "error": "body", "code": "code", "table": "code", "menu_item": "body"}
TONE = {"error": "error", "log": "muted"}


@dataclass
class Region:
    role: int
    top: int
    left: int
    bottom: int
    right: int  # inclusive bbox
    lines: list[tuple[int, int, int, bool]] = field(default_factory=list)  # (row, c0, c1 exclusive, selected)
    children: list["Region"] = field(default_factory=list)  # for boxes
    title: str | None = None
    key: str = ""

    @property
    def name(self) -> str:
        return "box" if self.role == R["border"] else ROLES[self.role]

    def contains(self, o: "Region") -> bool:
        return self.top <= o.top and self.bottom >= o.bottom and self.left <= o.left and self.right >= o.right and self is not o


# ---------------------------------------------------------------------------
# 1. segmentation into regions
def _row_runs(roles_row: np.ndarray, gap: int = 2) -> list[tuple[int, int, int]]:
    """Runs of equal non-blank role; blank gaps <= ``gap`` inside a run are absorbed."""
    runs: list[list[int]] = []
    w = len(roles_row)
    c = 0
    while c < w:
        r = int(roles_row[c])
        if r == 0:
            c += 1
            continue
        if runs and runs[-1][0] == r and c - runs[-1][2] <= gap:
            runs[-1][2] = c + 1
        else:
            runs.append([r, c, c + 1])
        c += 1
    return [tuple(x) for x in runs]  # type: ignore[misc]


def _boxes(roles: np.ndarray) -> list[Region]:
    """Rectangular frames from border cells: a top-left corner scan over connected border runs."""
    h, w = roles.shape
    border = roles == R["border"]
    boxes = []
    seen = np.zeros_like(border)
    for r in range(h):
        for c in range(w):
            if not border[r, c] or seen[r, c]:
                continue
            # flood fill the border component
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and border[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys, xs = zip(*cells)
            t, b, l, rr = min(ys), max(ys), min(xs), max(xs)
            boxes.append(Region(R["border"], t, l, b, rr))
    return boxes


def segment(f: Frame, roles: np.ndarray) -> list[Region]:
    h, w = roles.shape
    regions: list[Region] = []
    open_: list[Region] = []  # line regions that may still grow downward
    for r in range(h):
        runs = _row_runs(roles[r])
        nxt_open = []
        for role, c0, c1 in runs:
            if role in (R["border"],):
                continue
            target = None
            if role in LINE_ROLES:
                for reg in open_:
                    if reg.bottom != r - 1 or not (c0 <= reg.right and c1 - 1 >= reg.left):
                        continue
                    if reg.role == role or (role == R["selected"] and reg.role in LISTISH) or (
                            reg.role == R["selected"] and role in LISTISH):
                        target = reg
                        break
            if target is None or role not in LINE_ROLES:
                target = Region(role, r, c0, r, c1 - 1)
                regions.append(target)
            else:
                if target.role == R["selected"] and role != R["selected"]:
                    target.role = role
                if target.lines and target.lines[-1][0] == r:  # second run of same region on this row: extend
                    row, a, b, sel = target.lines[-1]
                    target.lines[-1] = (row, min(a, c0), max(b, c1), sel or role == R["selected"])
                    target.left, target.right = min(target.left, c0), max(target.right, c1 - 1)
                    nxt_open.append(target)
                    continue
                target.bottom = r
                target.left, target.right = min(target.left, c0), max(target.right, c1 - 1)
            target.lines.append((r, c0, c1, role == R["selected"]))
            nxt_open.append(target)
        open_ = nxt_open
    for reg in regions:
        if reg.role == R["selected"]:
            reg.role = R["menu_item"]
    # merge prompt + input on the same row into one "prompt" region (input kept as a line span)
    out, by_row = [], {}
    for reg in regions:
        if reg.role == R["prompt"]:
            by_row[reg.top] = reg
    for reg in regions:
        p = by_row.get(reg.top)
        if reg.role == R["input"] and p is not None and reg.left >= p.right:
            p.lines.append((reg.top, reg.left, reg.right + 1, True))
            p.right = reg.right
            continue
        out.append(reg)
    return out


def build_tree(f: Frame, roles: np.ndarray) -> Region:
    h, w = roles.shape
    root = Region(-1, 0, 0, h - 1, w - 1)
    boxes = [b for b in _boxes(roles) if b.bottom - b.top >= 2 and b.right - b.left >= 2]
    rules = [b for b in _boxes(roles) if b.bottom == b.top and b.right - b.left >= 4]
    leaves = segment(f, roles)
    for d in rules:
        d.role = R["border"]
        d.key = "divider"
    # titles that sit on a box's top edge become the box title
    for b in boxes:
        for reg in list(leaves):
            if reg.role == R["title"] and reg.top == b.top and b.left < reg.left <= reg.right < b.right:
                b.title = f.row_text(reg.top)[reg.left:reg.right + 1].strip()
                leaves.remove(reg)
    nodes = sorted(boxes, key=lambda b: (b.bottom - b.top) * (b.right - b.left), reverse=True)
    for item in [*nodes, *leaves, *rules]:
        parent = root
        for b in reversed(nodes):  # smallest containing box first
            if b.contains(item):
                parent = b
                break
        parent.children.append(item)
    _assign_keys(root, "root")
    return root


def _assign_keys(node: Region, key: str) -> None:
    node.key = key
    counts: dict[str, int] = {}
    node.children.sort(key=lambda c: (c.top, c.left))
    for ch in node.children:
        nm = "divider" if ch.key == "divider" else ch.name
        i = counts.get(nm, 0)
        counts[nm] = i + 1
        _assign_keys(ch, f"{key}.{nm}{i}" if key != "root" else f"{nm}{i}")


# ---------------------------------------------------------------------------
# 2. emission: A2UI v0.9 flat components + data model
RE_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%")
RE_HINT = re.compile(r"(\^[A-Z\\_\]]|M-\w|F\d{1,2}|<[^>]{1,12}>|\[[^\]]{1,12}\]|[A-Za-z?/]{1,5}(?=\s))\s?([A-Za-z][\w\- ]{0,14}?)(?=\s{2,}|\s*$|\s?\^|\s?F\d|\s?<|\s?\[)")
FILLED = set("#=|█▉▊▋▌▍▎▏■▓⣿>")


def _span_text(f: Frame, row: int, c0: int, c1: int) -> str:
    return "".join(chr(c) for c in f.cp[row, c0:c1] if c).rstrip()


def _progress(text: str, f: Frame, row: int, c0: int, c1: int) -> float | None:
    m = RE_PCT.search(text)
    if m:
        return min(1.0, float(m.group(1)) / 100)
    inner = re.search(r"[\[|]?([#=|█▉▊▋▌▍▎▏░▒▓■□⣿>\-. ]{4,})[\]|]?", text)
    if inner:
        s = inner.group(1)
        return round(sum(ch in FILLED for ch in s) / max(len(s.rstrip(" ")) or len(s), 1), 3)
    # coloured gauge (dialog style): fraction of cells with a different bg than the bar's end
    bg = f.bg[row, c0:c1]
    if len(bg) > 2 and bg[0] != bg[-1]:
        return round(float((bg == bg[0]).mean()), 3)
    return None


def _hints(text: str) -> list[dict]:
    items = [{"key": k.strip(), "label": l.strip()} for k, l in RE_HINT.findall(text) if l.strip()]
    if not items:
        items = [{"key": "", "label": t} for t in re.split(r"\s{2,}", text.strip()) if t]
    return items


class Emitter:
    def __init__(self, f: Frame):
        self.f = f
        self.components: dict[str, dict] = {}
        self.data: dict[str, dict] = {}
        self.styles: dict[str, dict] = {}

    def add(self, cid: str, comp: str, **props) -> str:
        self.components[cid] = {"id": cid, "component": comp, **props}
        return cid

    def node(self, n: Region) -> str:
        cid = self._node(n)
        if n.role != -1:  # style lives in the data model: geometry/colour changes stay data patches
            self.styles[cid] = region_style(self.f, n)
            self.components[cid]["style"] = {"path": f"/s/{cid}"}
        return cid

    def _node(self, n: Region) -> str:
        f, k = self.f, n.key
        if n.role == -1:
            return self.add("root", "Column", children=self._bands(n.children))
        if n.key == "divider" or (n.role == R["border"] and n.top == n.bottom):
            return self.add(k, "Divider", axis="horizontal")
        if n.role == R["border"]:
            kids = self._bands(n.children)
            if n.title is not None:
                self.data[k] = {"title": n.title}
                kids = [self.add(f"{k}:title", "Text", text={"path": f"/r/{k}/title"}, variant="h4"), *kids]
            col = self.add(f"{k}:col", "Column", children=kids)
            return self.add(k, "Card", child=col)
        role = ROLES[n.role]
        if role in ("title", "status_bar"):
            txt = " ".join(_span_text(f, r, a, b) for r, a, b, _ in n.lines).strip()
            self.data[k] = {"text": txt}
            return self.add(k, "Text", text={"path": f"/r/{k}/text"}, variant="h3" if role == "title" else "caption",
                            **({"tone": "bar"} if role == "status_bar" else {}))
        if role == "prompt":
            r0, a0, b0, _ = n.lines[0]
            prompt = _span_text(f, r0, a0, b0)
            rest = [(r, a, b) for r, a, b, inp in n.lines[1:] if inp]
            value = " ".join(_span_text(f, r, a, b) for r, a, b in rest)
            self.data[k] = {"prompt": prompt, "value": value}
            p = self.add(f"{k}:p", "Text", text={"path": f"/r/{k}/prompt"}, variant="code", tone="prompt")
            active = f.cursor[0] == r0
            v = (self.add(f"{k}:in", "TextField", label="", value={"path": f"/r/{k}/value"}, variant="shortText") if active
                 else self.add(f"{k}:v", "Text", text={"path": f"/r/{k}/value"}, variant="code"))
            return self.add(k, "Row", children=[p, v], align="center")
        if role == "input":
            r0, a0, b0, _ = n.lines[0]
            self.data[k] = {"value": _span_text(f, r0, a0, b0)}
            return self.add(k, "TextField", label="", value={"path": f"/r/{k}/value"}, variant="shortText")
        if role == "progress":
            items = []
            for r, a, b, _ in n.lines:
                t = _span_text(f, r, a, b)
                items.append({"label": RE_PCT.sub("", re.sub(r"[\[\]#=|█▉▊▋▌▍▎▏░▒▓■□⣿>]{2,}", "", t)).strip(), "value": _progress(t, f, r, a, b)})
            self.data[k] = {"items": items}
            tpl = self.add(f"{k}:item", "Progress", value={"path": "value"}, label={"path": "label"})
            return self.add(k, "Column", children={"path": f"/r/{k}/items", "componentId": tpl})
        if role == "key_hint":
            items = _hints(" ".join(_span_text(f, r, a, b) for r, a, b, _ in n.lines))
            self.data[k] = {"items": items}
            lbl = self.add(f"{k}:lbl", "Text", text={"path": "label"}, variant="caption")
            btn = self.add(f"{k}:btn", "Button", child=lbl, variant="borderless", shortcut={"path": "key"},
                           action={"event": {"name": "key", "context": {"key": {"path": "key"}}}})
            return self.add(k, "Row", children={"path": f"/r/{k}/items", "componentId": btn})
        # line-oriented regions -> List with an item template; one data entry per terminal line
        rows = [{"text": _span_text(f, r, a, b), "selected": sel, **line_style(f, r, a, b)} for r, a, b, sel in n.lines]
        self.data[k] = {"rows": rows}
        tpl = self.add(f"{k}:row", "Text", text={"path": "text"}, variant=TEXT_VARIANT.get(role, "body"),
                       highlight={"path": "selected"}, style={"fg": {"path": "fg"}, "bg": {"path": "bg"}, "bold": {"path": "bold"}},
                       **({"tone": TONE[role]} if role in TONE else {}))
        return self.add(k, "List", children={"path": f"/r/{k}/rows", "componentId": tpl}, role=role)

    def _bands(self, children: list[Region]) -> list[str]:
        """Children whose row spans overlap are laid out side by side (Row); others stack."""
        out, i = [], 0
        while i < len(children):
            band = [children[i]]
            bottom = children[i].bottom
            j = i + 1
            while j < len(children) and children[j].top <= bottom:
                band.append(children[j])
                bottom = max(bottom, children[j].bottom)
                j += 1
            ids = [self.node(c) for c in sorted(band, key=lambda c: c.left)]
            if len(ids) > 1:
                out.append(self.add(f"{band[0].key}:band", "Row", children=ids, align="start"))
            else:
                out.extend(ids)
            i = j
        return out


def compile_frame(f: Frame, roles: np.ndarray) -> tuple[dict[str, dict], dict]:
    """Returns (components by id, data model) for one frame."""
    em = Emitter(f)
    em.node(build_tree(f, roles))
    return em.components, {"r": em.data, "s": em.styles}

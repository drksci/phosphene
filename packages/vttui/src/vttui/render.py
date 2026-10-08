"""Reference A2UI client: applies a message stream to a surface and renders HTML.

Two modes:
  native   — flow layout, the way a generic A2UI renderer would draw the components
  faithful — uses the emitted style layer: same proportions/colours as the source terminal
Also used by tests to prove that incremental streams reconstruct exactly the full-frame UI.
"""

from __future__ import annotations

import copy
import html
from typing import Any

from .style import resolve_style, stylesheet

NATIVE_CSS = """
.a2ui-native{font-family:system-ui,sans-serif;font-size:14px;color:#ddd;background:#16181d;padding:12px;border-radius:8px}
.a2ui-native .Column,.a2ui-native .List{display:flex;flex-direction:column;gap:2px}
.a2ui-native .Row{display:flex;flex-direction:row;gap:12px;align-items:flex-start;flex-wrap:wrap}
.a2ui-native .Card{border:1px solid #3a3f4b;border-radius:10px;padding:8px 10px;margin:4px 0;background:#1d2027}
.a2ui-native .Text.h3,.a2ui-native .Text.h4{font-weight:600;color:#fff}
.a2ui-native .Text.code{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;white-space:pre}
.a2ui-native .Text.caption{font-size:12px;color:#aaa}
.a2ui-native .Text[data-tone=bar]{background:#2b3240;padding:2px 8px;border-radius:4px}
.a2ui-native .Text[data-tone=error]{color:#ff6b6b}
.a2ui-native .Text[data-tone=muted]{color:#9aa}
.a2ui-native .Text[data-tone=prompt]{color:#4ade80}
.a2ui-native .sel{background:#3b82f6;color:#fff;border-radius:4px}
.a2ui-native .TextField{font-family:ui-monospace,monospace;background:#0f1115;border:1px solid #3b82f6;border-radius:4px;padding:1px 6px;min-width:20ch}
.a2ui-native .Button{background:#2b3240;border-radius:4px;padding:1px 6px;font-size:12px}
.a2ui-native .Button kbd{color:#fbbf24;margin-right:4px}
.a2ui-native progress{width:220px;vertical-align:middle;margin-right:8px}
.a2ui-native .Divider{border-top:1px solid #3a3f4b;margin:4px 0}
"""


def _ptr(path: str) -> list[str]:
    if path in ("", "/"):
        return []
    return [p.replace("~1", "/").replace("~0", "~") for p in path.split("/")[1:]]


class Surface:
    def __init__(self):
        self.components: dict[str, dict] = {}
        self.data: Any = {}
        self.theme: dict | None = None

    def apply(self, msg: dict) -> None:
        if "createSurface" in msg:
            self.components, self.data = {}, {}
            self.theme = msg["createSurface"].get("theme")
        elif "updateComponents" in msg:
            for c in msg["updateComponents"]["components"]:
                self.components[c["id"]] = copy.deepcopy(c)
        elif "updateDataModel" in msg:
            body = msg["updateDataModel"]
            self._set(body.get("path", "/"), body.get("value"), "value" not in body)

    def _set(self, path: str, value: Any, delete: bool) -> None:
        keys = _ptr(path)
        if not keys:
            self.data = {} if delete else copy.deepcopy(value)
            return
        node = self.data
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node.setdefault(k, {})
        last = keys[-1]
        if isinstance(node, list):
            i = int(last)
            if delete:
                del node[i]
            elif i == len(node):
                node.append(copy.deepcopy(value))
            else:
                node[i] = copy.deepcopy(value)
        elif delete:
            node.pop(last, None)
        else:
            node[last] = copy.deepcopy(value)

    def get(self, path: str, scope: Any = None) -> Any:
        """Absolute paths resolve against the data model; relative ones against a template item."""
        node = self.data if path.startswith("/") or scope is None else scope
        for k in _ptr(path if path.startswith("/") else "/" + path):
            if isinstance(node, list):
                node = node[int(k)] if int(k) < len(node) else None
            elif isinstance(node, dict):
                node = node.get(k)
            else:
                return None
        return node

    def resolve(self, v: Any, scope: Any = None) -> Any:
        if isinstance(v, dict) and set(v) == {"path"}:
            return self.get(v["path"], scope)
        return v

    # -----------------------------------------------------------------------
    def render(self, faithful: bool = False) -> str:
        body = self._render("root", None, faithful) if "root" in self.components else ""
        if faithful and self.theme:
            return f'<style>{stylesheet(self.components, self.theme, self.data)}</style><div class="a2ui-term">{body}</div>'
        return f'<style>{NATIVE_CSS}</style><div class="a2ui-native">{body}</div>'

    def _children(self, c: dict, scope: Any) -> list[tuple[str, Any]]:
        ch = c.get("children")
        if isinstance(ch, list):
            return [(i, scope) for i in ch]
        if isinstance(ch, dict):
            return [(ch["componentId"], it) for it in (self.get(ch["path"], scope) or [])]
        if "child" in c:
            return [(c["child"], scope)]
        return []

    def _render(self, cid: str, scope: Any, faithful: bool) -> str:
        c = self.components.get(cid)
        if c is None:
            return ""
        kind = c["component"]
        st0 = resolve_style(c, self.data)
        placed = faithful and isinstance(st0, dict) and "area" in st0
        attr = f' data-id="{html.escape(cid)}"' if placed else ""
        inner = "".join(self._render(i, s, faithful) for i, s in self._children(c, scope))
        tone = c.get("tone")
        if kind == "Text":
            txt = html.escape(str(self.resolve(c.get("text"), scope) or ""))
            sel = bool(self.resolve(c.get("highlight"), scope))
            if scope is not None and faithful:  # template row inside a placed List: per-line colours
                st = c.get("style", {})
                fg, bg, bold = (self.resolve(st.get(k), scope) for k in ("fg", "bg", "bold"))
                css = (f"color:var(--{fg});" if fg and fg != "default" else "") + (
                    f"background:var(--{bg});" if bg and bg != "default" else "") + ("font-weight:700;" if bold else "")
                return f'<span class="row-item{" sel" if sel else ""}" style="{css}">{txt}</span>'
            cls = f'Text {c.get("variant", "body")}{" sel" if sel else ""}'
            return f'<div{attr} class="{cls}"{f" data-tone={tone}" if tone else ""}>{txt}</div>'
        if kind == "TextField":
            val = html.escape(str(self.resolve(c.get("value"), scope) or ""))
            return f'<span{attr} class="TextField">{val}<span class="caret">▏</span></span>'
        if kind == "Progress":
            v = self.resolve(c.get("value"), scope)
            lbl = html.escape(str(self.resolve(c.get("label"), scope) or ""))
            pct = "" if v is None else f"{v * 100:.0f}%"
            return f'<div class="Progress"><progress value="{v or 0}" max="1"></progress>{pct} {lbl}</div>'
        if kind == "Button":
            key = html.escape(str(self.resolve(c.get("shortcut"), scope) or ""))
            return f'<span class="Button">{f"<kbd>{key}</kbd>" if key else ""}{inner}</span>'
        if kind == "Divider":
            return f'<div{attr} class="Divider"></div>'
        if kind == "Card":
            if faithful:  # frame is a positioned sibling so children stay placed relative to the surface
                return f'<div{attr} class="card-frame"></div><div class="flow">{inner}</div>'
            return f'<div class="Card">{inner}</div>'
        if faithful and not placed:
            return f'<div class="flow">{inner}</div>'
        return f'<div{attr} class="{kind}">{inner}</div>'

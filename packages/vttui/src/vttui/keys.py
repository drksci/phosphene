"""A2UI client actions -> terminal keystrokes (the return path of the gateway).

The client never speaks VT: it sends A2UI ``action`` events from the components the gateway
emitted, and these functions turn them into the bytes a user would have typed.

  Button  {"name": "key",    "context": {"key": "^X"}}                  -> "\\x18"
  List    {"name": "select", "context": {"list": "box0.menu_item0", "row": 4}} -> arrows (+ Enter if activate)
  TextField value change  (old -> new)                                   -> edit keys (+ "\\r" on submit)
"""

from __future__ import annotations

import re

ESC = "\x1b"
NAMED = {
    "enter": "\r", "ret": "\r", "return": "\r", "tab": "\t", "esc": ESC, "escape": ESC, "space": " ", "spc": " ",
    "bs": "\x7f", "backspace": "\x7f", "del": f"{ESC}[3~", "delete": f"{ESC}[3~", "ins": f"{ESC}[2~",
    "up": f"{ESC}[A", "down": f"{ESC}[B", "right": f"{ESC}[C", "left": f"{ESC}[D", "home": f"{ESC}[H", "end": f"{ESC}[F",
    "pgup": f"{ESC}[5~", "pgdn": f"{ESC}[6~", "pageup": f"{ESC}[5~", "pagedown": f"{ESC}[6~",
    "↑": f"{ESC}[A", "↓": f"{ESC}[B", "→": f"{ESC}[C", "←": f"{ESC}[D",
}
FKEYS = {1: f"{ESC}OP", 2: f"{ESC}OQ", 3: f"{ESC}OR", 4: f"{ESC}OS", 5: f"{ESC}[15~", 6: f"{ESC}[17~", 7: f"{ESC}[18~",
         8: f"{ESC}[19~", 9: f"{ESC}[20~", 10: f"{ESC}[21~", 11: f"{ESC}[23~", 12: f"{ESC}[24~"}


def key_bytes(label: str) -> str:
    """Key label as printed by a TUI ("^X", "M-x", "F1", "<Enter>", "[ OK ]", "q", "Esc") -> bytes."""
    k = label.strip()
    inner = re.fullmatch(r"[<\[]\s*(.+?)\s*[>\]]", k)
    if inner:
        k = inner.group(1)
    low = k.lower()
    if low in NAMED:
        return NAMED[low]
    if m := re.fullmatch(r"\^(.)", k):
        return chr(ord(m.group(1).upper()) & 0x1F) if m.group(1) != "?" else "\x7f"
    if m := re.fullmatch(r"(?:C-|Ctrl-|Ctrl\+)(.)", k, re.I):
        return chr(ord(m.group(1).upper()) & 0x1F)
    if m := re.fullmatch(r"(?:M-|Alt-|Alt\+|Meta-)(.+)", k, re.I):
        return ESC + key_bytes(m.group(1))
    if m := re.fullmatch(r"F(\d{1,2})", k, re.I):
        return FKEYS.get(int(m.group(1)), "")
    if len(k) == 1:
        return k
    # button captions ("OK", "Cancel") are activated by their first letter in dialog/whiptail, Enter otherwise
    return "\r" if low in ("ok", "yes", "continue", "next") else k[0].lower()


def edit_bytes(old: str, new: str, submit: bool = False) -> str:
    """Keys that turn the field's ``old`` text into ``new``, assuming the cursor is at its end."""
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    keys = "\x7f" * (len(old) - p) + new[p:] if len(old) - p <= 32 else "\x15" + new  # ^U kills a long line
    return keys + ("\r" if submit else "")


def select_bytes(current: int | None, target: int, activate: bool = False) -> str:
    """Arrow keys moving a list selection from ``current`` to ``target`` (+ Enter to activate)."""
    if current is None:
        return ""
    d = target - current
    return (NAMED["down"] * d if d > 0 else NAMED["up"] * -d) + ("\r" if activate else "")

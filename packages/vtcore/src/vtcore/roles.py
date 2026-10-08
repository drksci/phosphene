"""Shared semantic vocabulary for terminal cells (model output == compiler input)."""

ROLES = [
    "blank",       # 0  empty background
    "text",        # 1  plain output / prose
    "prompt",      # 2  shell / REPL prompt prefix (PS1, >>>, ❯ ...)
    "input",       # 3  the line being edited at the cursor (command or field value)
    "border",      # 4  box-drawing frames, rules, separators
    "title",       # 5  headings, window/panel titles
    "status_bar",  # 6  full-width highlighted bar (top/bottom), mode lines
    "menu_item",   # 7  item of a selectable list / menu / file tree
    "selected",    # 8  the highlighted (current) item
    "table",       # 9  column-aligned tabular data
    "progress",    # 10 progress bars, spinners, percentages
    "code",        # 11 editor buffer / source code / line-number gutter
    "log",         # 12 timestamped or levelled log lines
    "error",       # 13 errors / warnings / tracebacks
    "key_hint",    # 14 key bindings & buttons ("^X Exit", "[ OK ]", "F1 Help")
]
ROLE_ID = {r: i for i, r in enumerate(ROLES)}
N_ROLES = len(ROLES)

# roles whose region "owns" its whitespace (spaces inside get the role instead of blank)
FILL_ROLES = {ROLE_ID[r] for r in ("status_bar", "selected", "input", "progress")}

APP_KINDS = ["shell", "editor", "monitor", "menu", "installer", "repl", "pager", "dashboard", "other"]
APP_ID = {a: i for i, a in enumerate(APP_KINDS)}

# palette for visualisation (hex), index-aligned with ROLES
ROLE_COLORS = [
    "#00000000", "#9aa5b1", "#4ade80", "#22d3ee", "#64748b", "#f472b6", "#a78bfa", "#fbbf24",
    "#f97316", "#38bdf8", "#34d399", "#c084fc", "#94a3b8", "#ef4444", "#facc15",
]

"""Frame = one settled terminal screen, as dense per-cell arrays."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# attribute bits
BOLD, ITALIC, UNDERLINE, REVERSE, BLINK, STRIKE = 1, 2, 4, 8, 16, 32
# colour indices: 0..15 ANSI, 16 = default, 17 = unmappable
DEFAULT_COLOR = 16
N_COLORS = 18


@dataclass
class Frame:
    cp: np.ndarray  # (H, W) int32 codepoint; 0 = right half of a wide char
    fg: np.ndarray  # (H, W) uint8
    bg: np.ndarray  # (H, W) uint8
    attr: np.ndarray  # (H, W) uint8 bitfield
    cursor: tuple[int, int] = (-1, -1)  # (row, col); (-1, -1) when hidden
    t: float = 0.0
    damage: np.ndarray | None = None  # (H, W) bool, cells changed vs previous keyframe
    rec_id: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.cp.shape  # type: ignore[return-value]

    def row_text(self, r: int) -> str:
        """Row as a string where index == column (wide-char continuations become spaces)."""
        return "".join(chr(c) if c else " " for c in self.cp[r])

    def lines(self) -> list[str]:
        return [self.row_text(r) for r in range(self.shape[0])]

    def display_lines(self) -> list[str]:
        """Rows as they'd display (continuation cells dropped, trailing space stripped)."""
        return ["".join(chr(c) for c in row if c).rstrip() for row in self.cp]

    def content_hash(self) -> str:
        h = hashlib.blake2b(digest_size=12)
        for a in (self.cp, self.fg, self.bg, self.attr):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# storage: one .npz per shard, cell arrays concatenated with per-frame offsets


def save_frames(path: str | Path, frames: list[Frame], extra: dict[str, np.ndarray] | None = None) -> None:
    """Save frames (plus optional per-frame HxW arrays in ``extra``, e.g. labels) to one .npz."""
    shapes = np.array([f.shape for f in frames], dtype=np.int32).reshape(-1, 2)
    cat = lambda xs: np.concatenate([x.ravel() for x in xs]) if xs else np.zeros(0)
    arrays = {
        "shapes": shapes,
        "cp": cat([f.cp for f in frames]).astype(np.int32),
        "fg": cat([f.fg for f in frames]).astype(np.uint8),
        "bg": cat([f.bg for f in frames]).astype(np.uint8),
        "attr": cat([f.attr for f in frames]).astype(np.uint8),
        "damage": cat([f.damage if f.damage is not None else np.ones(f.shape, bool) for f in frames]).astype(bool),
        "meta": np.array(json.dumps([{"cursor": list(f.cursor), "t": f.t, "rec_id": f.rec_id, **f.meta} for f in frames])),
    }
    for k, xs in (extra or {}).items():
        arrays[f"x_{k}"] = cat(list(xs))
    np.savez_compressed(path, **arrays)


def load_frames(path: str | Path) -> tuple[list[Frame], dict[str, list[np.ndarray]]]:
    z = np.load(path, allow_pickle=False)
    shapes = z["shapes"]
    metas = json.loads(str(z["meta"]))
    offs = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])])
    frames, extra = [], {k[2:]: [] for k in z.files if k.startswith("x_")}
    cols = {k: z[k] for k in ("cp", "fg", "bg", "attr", "damage")}
    xcols = {k: z[f"x_{k}"] for k in extra}
    for i, (h, w) in enumerate(shapes):
        s = slice(offs[i], offs[i + 1])
        m = metas[i]
        frames.append(
            Frame(
                cp=cols["cp"][s].reshape(h, w),
                fg=cols["fg"][s].reshape(h, w),
                bg=cols["bg"][s].reshape(h, w),
                attr=cols["attr"][s].reshape(h, w),
                damage=cols["damage"][s].reshape(h, w),
                cursor=tuple(m.pop("cursor")),
                t=m.pop("t"),
                rec_id=m.pop("rec_id"),
                meta=m,
            )
        )
        for k in extra:
            extra[k].append(xcols[k][s].reshape(h, w))
    return frames, extra

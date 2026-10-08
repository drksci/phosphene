"""VTM: a small per-cell segmentation transformer over the terminal grid.

Input  per cell: char id, fg, bg, attr bits, damage (changed since last keyframe), cursor flags,
       plus row/col position embeddings (row also measured from the bottom: status bars live there).
Body   conv stem (local glyph shapes) + N axial blocks (row attention, then column attention), so
       every cell sees its whole row and whole column at O(HW(H+W)) cost — enough to find aligned
       table columns, box edges and full-width bars.
Output per-cell role logits (ROLES) + per-frame app-kind logits (APP_KINDS).

~1-3M params: trains on a free Colab T4 in minutes, exports to ONNX (onnxruntime-web / WebGPU), and
its argmax is a pure deterministic function of the screen — the "learned transformer" half of the
pipeline; the compiler/reconciler (vttui) is the deterministic half.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vtcore.encode import CHAR_VOCAB
from vtcore.frame import N_COLORS
from vtcore.roles import APP_KINDS, N_ROLES

MAX_ROWS, MAX_COLS = 128, 320


class AxialBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int = 4, drop: float = 0.0):
        super().__init__()
        self.heads = heads
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv_r, self.qkv_c = nn.Linear(d, 3 * d), nn.Linear(d, 3 * d)
        self.o_r, self.o_c = nn.Linear(d, d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))
        self.drop = drop

    def _attn(self, x, qkv, out, mask):
        # x: (N, L, d), mask: (N, L) True = valid
        n, l, d = x.shape
        q, k, v = qkv(x).view(n, l, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        am = mask[:, None, None, :] if mask is not None else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=self.drop if self.training else 0.0)
        return out(y.transpose(1, 2).reshape(n, l, d))

    def forward(self, x, mask):
        b, h, w, d = x.shape
        xr = self.n1(x).reshape(b * h, w, d)
        mr = mask.reshape(b * h, w)
        x = x + self._attn(xr, self.qkv_r, self.o_r, mr | ~mr.any(1, keepdim=True)).view(b, h, w, d)
        xc = self.n2(x).transpose(1, 2).reshape(b * w, h, d)
        mc = mask.transpose(1, 2).reshape(b * w, h)
        x = x + self._attn(xc, self.qkv_c, self.o_c, mc | ~mc.any(1, keepdim=True)).view(b, w, h, d).transpose(1, 2)
        return x + self.mlp(self.n3(x))


class VTM(nn.Module):
    def __init__(self, d: int = 128, layers: int = 4, heads: int = 4, n_roles: int = N_ROLES, n_apps: int = len(APP_KINDS)):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, n_roles=n_roles, n_apps=n_apps)
        self.char = nn.Embedding(CHAR_VOCAB, d)
        self.fg = nn.Embedding(N_COLORS, d)
        self.bg = nn.Embedding(N_COLORS, d)
        self.attr = nn.Linear(6, d, bias=False)
        self.dmg = nn.Embedding(2, d)
        self.cur = nn.Embedding(3, d)
        self.row = nn.Embedding(MAX_ROWS, d)
        self.row_b = nn.Embedding(MAX_ROWS, d)
        self.col = nn.Embedding(MAX_COLS, d)
        self.col_r = nn.Embedding(MAX_COLS, d)
        self.stem = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, groups=d), nn.Conv2d(d, d, 1), nn.GELU(),
            nn.Conv2d(d, d, (1, 5), padding=(0, 2), groups=d), nn.Conv2d(d, d, 1),
        )
        self.blocks = nn.ModuleList(AxialBlock(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_roles)
        self.app_head = nn.Linear(d, n_apps)

    def forward(self, char, fg, bg, attr, damage, cursor, mask):
        """All inputs (B, H, W) int64 except mask (B, H, W) bool (True = real cell)."""
        b, h, w = char.shape
        bits = torch.stack([(attr >> i) & 1 for i in range(6)], -1).float()
        # per-sample true height/width for bottom/right-relative positions
        hs = mask.any(2).sum(1).clamp(min=1)  # (B,)
        ws = mask.any(1).sum(1).clamp(min=1)
        ri = torch.arange(h, device=char.device)
        ci = torch.arange(w, device=char.device)
        rb = (hs[:, None] - 1 - ri[None]).clamp(0, MAX_ROWS - 1)  # (B, H)
        cr = (ws[:, None] - 1 - ci[None]).clamp(0, MAX_COLS - 1)  # (B, W)
        x = (self.char(char) + self.fg(fg) + self.bg(bg) + self.attr(bits) + self.dmg(damage) + self.cur(cursor)
             + self.row(ri.clamp(max=MAX_ROWS - 1))[None, :, None] + self.row_b(rb)[:, :, None]
             + self.col(ci.clamp(max=MAX_COLS - 1))[None, None] + self.col_r(cr)[:, None])
        x = x * mask[..., None]
        x = x + self.stem(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        pooled = (x * mask[..., None]).sum((1, 2)) / mask.sum((1, 2)).clamp(min=1)[:, None]
        return self.head(x), self.app_head(pooled)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

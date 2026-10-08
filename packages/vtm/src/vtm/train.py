"""Training, evaluation, inference and ONNX export for the VTM."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vtcore.encode import features
from vtcore.frame import Frame
from vtcore.roles import N_ROLES

from .data import GridDataset, collate
from .metrics import confusion, report
from .model import VTM


def class_weights(ds: GridDataset, power: float = 0.5) -> torch.Tensor:
    counts = np.ones(N_ROLES)
    for _, r, _, _ in ds.items:
        counts += np.bincount(r.ravel(), minlength=N_ROLES)
    w = (counts.sum() / counts) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train(
    train_shards: list[str], val_shards: list[str] | None = None, out_dir: str = "runs/vtm", epochs: int = 8,
    batch_size: int = 16, lr: float = 2e-3, d: int = 128, layers: int = 4, device: str | None = None, log=print,
) -> VTM:
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ds = GridDataset(train_shards)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    model = VTM(d=d, layers=layers).to(device)
    cw = class_weights(ds).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    log(f"train frames={len(ds)} params={model.n_params / 1e6:.2f}M device={device}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, step = [], 0
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, app_logits = model(b["char"], b["fg"], b["bg"], b["attr"], b["damage"], b["cursor"], b["mask"])
            ce = F.cross_entropy(logits.float().permute(0, 3, 1, 2), b["roles"], weight=cw, ignore_index=-100, reduction="none")
            loss = (ce * b["weight"]).sum() / b["weight"].sum().clamp(min=1)
            has_app = b["app"] >= 0
            if has_app.any():
                loss = loss + 0.1 * F.cross_entropy(app_logits.float()[has_app], b["app"][has_app])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            step += 1
        rec = {"epoch": ep, "loss": tot / max(len(dl), 1), "sec": round(time.time() - t0, 1)}
        if val_shards:
            rec |= {k: v for k, v in evaluate(model, val_shards, device).items() if k != "per_class"}
        history.append(rec)
        log(json.dumps(rec))
    save(model, out / "vtm.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    return model


@torch.no_grad()
def predict(model: VTM, f: Frame, device: str | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (roles HxW int8, confidence HxW float32, app id)."""
    device = device or next(model.parameters()).device
    model.eval()
    feats = features(f)
    t = {k: torch.from_numpy(v)[None].to(device) for k, v in feats.items()}
    mask = torch.ones_like(t["char"], dtype=torch.bool)
    logits, app = model(t["char"], t["fg"].clamp(max=17), t["bg"].clamp(max=17), t["attr"], t["damage"], t["cursor"], mask)
    p = logits[0].float().softmax(-1)
    conf, roles = p.max(-1)
    return roles.cpu().numpy().astype(np.int8), conf.cpu().numpy(), int(app[0].argmax())


@torch.no_grad()
def evaluate(model: VTM, shards: list[str], device: str | None = None) -> dict:
    ds = GridDataset(shards, augment=False, max_rows=128, max_cols=320)
    preds, golds = [], []
    for f, r, _, _ in ds.items:
        p, _, _ = predict(model, f, device)
        preds.append(p)
        golds.append(r)
    return report(confusion(preds, golds))


def save(model: VTM, path: str | Path) -> None:
    torch.save({"cfg": model.cfg, "state": model.state_dict()}, path)


def load(path: str | Path, device: str = "cpu") -> VTM:
    ck = torch.load(path, map_location=device)
    m = VTM(**{k: ck["cfg"][k] for k in ("d", "layers", "heads", "n_roles", "n_apps")})
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def export_onnx(model: VTM, path: str | Path, rows: int = 24, cols: int = 80) -> None:
    """Dynamic H/W ONNX graph -> runs in onnxruntime-web (WebGPU/WASM) for in-browser streaming."""
    model = model.cpu().eval()
    z = torch.zeros(1, rows, cols, dtype=torch.long)
    mask = torch.ones(1, rows, cols, dtype=torch.bool)
    names = ["char", "fg", "bg", "attr", "damage", "cursor", "mask"]
    axes = {n: {1: "rows", 2: "cols"} for n in names} | {"roles": {1: "rows", 2: "cols"}}
    torch.onnx.export(model, (z, z, z, z, z, z, mask), str(path), input_names=names, output_names=["roles", "app"],
                      dynamic_axes=axes, opset_version=18, dynamo=False)


if __name__ == "__main__":  # python -m vtm.train '{"train": [...], "val": [...], "out": "...", "epochs": 5}'
    import sys

    cfg = json.loads(sys.argv[1])
    train(cfg["train"], cfg.get("val"), out_dir=cfg["out"], epochs=cfg.get("epochs", 8), batch_size=cfg.get("bs", 16),
          lr=cfg.get("lr", 2e-3), d=cfg.get("d", 128), layers=cfg.get("layers", 4))

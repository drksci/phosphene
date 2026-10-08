"""Generates the Colab notebooks from plain Python cell lists (keeps notebooks diff-friendly).

    python notebooks/_build.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
COLAB = "https://colab.research.google.com/github/drksci/phosphene/blob/main/notebooks/"

SETUP = r'''#@title Setup: clone + install (re-run safe)
import os, sys
if not os.path.exists("/content/phosphene"):
    !git clone -q https://github.com/drksci/phosphene /content/phosphene
%cd /content/phosphene
!git pull -q
!pip install -q -e packages/vtcore -e packages/vttui -e "packages/vtm[train,data,label]"

#@markdown Persist data/models to Google Drive (survives runtime resets) or keep them in /content.
USE_DRIVE = False  #@param {type:"boolean"}
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA = "/content/drive/MyDrive/phosphene"
else:
    DATA = "/content/phosphene_data"
os.makedirs(DATA, exist_ok=True)
print("DATA =", DATA)'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def nb(name, cells, gpu=False):
    meta = {"colab": {"provenance": [], "toc_visible": True}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    badge = md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}{name})")
    doc = {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": [badge, *cells]}
    for c in doc["cells"]:
        c["source"] = [l + "\n" for l in c["source"].split("\n")]
        c["source"][-1] = c["source"][-1].rstrip("\n")
    (HERE / name).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
nb("01_data_prep.ipynb", [
    md("""
# 01 · Data prep — keyframes, dedupe, free labels

**phosphene / vtm** turns terminal byte streams into incremental A2UI. This notebook builds the corpus:

1. Stream a subset of [asciinema_terminal_recordings](https://huggingface.co/datasets/James4Ever0/asciinema_terminal_recordings) (79k casts, 18 GB uncompressed — we never extract it all).
2. Emulate each cast (pyte) → **keyframes** (screen settled ≥150 ms) → **dedupe** by layout signature across the corpus.
3. **Weak labels** for every frame from heuristic labeling functions (free) + per-cell confidence.
4. **Synthetic TUIs** with *exact* labels (free): shells, htop-likes, dialogs, editors, installers, pagers, dashboards — rendered as real ANSI through the same emulator.

Outputs `.npz` shards in `DATA/`. No API calls, CPU runtime is fine.
"""),
    code(SETUP),
    code('''#@title Parameters
N_RECORDINGS = 4000   #@param {type:"integer"}
SPREAD = False        #@param {type:"boolean"}
PER_RECORDING = 12    #@param {type:"integer"}
N_SYNTH_TRAIN = 3000  #@param {type:"integer"}
N_SYNTH_VAL = 300     #@param {type:"integer"}
SHARD_SIZE = 4000'''),
    code('''#@title Fetch a subset of recordings (size-filtered, never the full 18 GB)
from vtm.corpus import fetch_recordings
paths = fetch_recordings("/content/recordings", n=N_RECORDINGS, spread=SPREAD)
print(len(paths), "recordings")'''),
    code('''#@title Keyframes → dedupe → weak labels → shards
import glob, time
from tqdm.auto import tqdm
from vtcore import iter_corpus
from vtm.data import build_weak_shard

for p in glob.glob(f"{DATA}/weak_*.npz"): os.remove(p)
seen, buf, n_shard, t0 = set(), [], 0, time.time()
for f in tqdm(iter_corpus(paths, per_recording=PER_RECORDING, seen=seen), desc="frames"):
    buf.append(f)
    if len(buf) == SHARD_SIZE:
        build_weak_shard(f"{DATA}/weak_{n_shard:03d}.npz", buf); buf, n_shard = [], n_shard + 1
if buf:
    build_weak_shard(f"{DATA}/weak_{n_shard:03d}.npz", buf)
print(f"{len(seen)} unique layouts in {time.time()-t0:.0f}s")'''),
    code('''#@title Synthetic TUIs with exact labels
from vtm.data import build_synth_shard
n_tr = build_synth_shard(f"{DATA}/synth_train.npz", range(N_SYNTH_TRAIN))
n_va = build_synth_shard(f"{DATA}/synth_val.npz", range(10**6, 10**6 + N_SYNTH_VAL))
print("synth frames:", n_tr, "train /", n_va, "val")'''),
    md("## Corpus stats"),
    code('''import numpy as np, matplotlib.pyplot as plt, collections
from vtcore import load_frames, ROLES
from vtm.labeling.heuristics import frame_confidence

frames, roles, confs = [], [], []
for p in sorted(glob.glob(f"{DATA}/weak_*.npz")):
    fr, ex = load_frames(p); frames += fr; roles += ex["roles"]; confs += ex["conf"]
fc = np.array([frame_confidence(r, c.astype(np.float32)) for r, c in zip(roles, confs)])
sizes = collections.Counter(f.shape for f in frames).most_common(8)
hist = np.bincount(np.concatenate([r.ravel() for r in roles]), minlength=len(ROLES))

fig, ax = plt.subplots(1, 3, figsize=(16, 3.5))
ax[0].bar(ROLES[1:], hist[1:]); ax[0].tick_params(axis="x", rotation=60); ax[0].set_title("weak-label role cells")
ax[1].hist(fc, bins=40); ax[1].set_title("heuristic frame confidence (low → send to LLM)")
ax[2].barh([f"{h}x{w}" for (h, w), _ in sizes], [n for _, n in sizes]); ax[2].set_title("terminal sizes")
plt.tight_layout(); print(len(frames), "real frames")'''),
    md("## Look at it\nRole overlay legend is shown above each gallery. Real frames carry *weak* labels; synthetic ones are exact."),
    code('''from vtm.viz import compare_html, show, frame_html, legend_html
import random
rng = random.Random(1)
for i in rng.sample(range(len(frames)), 4):
    show(compare_html(frames[i], {"frame": None, f"heuristics (conf {fc[i]:.2f})": roles[i]}))'''),
    code('''sfr, sex = load_frames(f"{DATA}/synth_val.npz")
show(legend_html() + "".join(frame_html(sfr[i], sex["roles"][i], title=f"synthetic #{i}", font_px=8) for i in range(0, 60, 12)))'''),
    code('''#@title Heuristic baseline vs exact synthetic labels
from vtm.labeling.heuristics import label
from vtm.metrics import confusion, report
from vtm.viz import plot_iou
rep_h = report(confusion([label(f)[0] for f in sfr], sex["roles"]))
plot_iou({"heuristics @ synth_val": rep_h}); print({k: round(v, 3) for k, v in rep_h.items() if k != "per_class"})'''),
])

# ---------------------------------------------------------------------------
nb("02_llm_labeling.ipynb", [
    md("""
# 02 · LLM labeling — cheapest effective path

Spend tokens only where free labels are weak:

| stage | what | cost |
|---|---|---|
| 0 | dedupe by layout signature (01) | free |
| 1 | heuristics + synthetic exact labels (01) | free |
| 2 | **select** ~1–2k frames: 60% lowest heuristic confidence + 40% k-center diversity | free |
| 3 | **label** with Claude Sonnet 5.5 · Batch API (−50%) · cached system prompt · thinking off · region (not per-cell) JSON | ≈ $0.003/frame |
| 4 | **review** with Claude Opus 5.5 only where Sonnet ≠ heuristics (top ~15%) + a 100-frame gold test set | ≈ $0.02/frame on ~15% |
| 5 | train (03), then repeat 2–4 on frames *the model* is unsure about (active learning) | — |

Expected first round for 1,500 frames: **≈ $5–10** total. A hard budget gate refuses to submit above `MAX_USD`.
Needs `ANTHROPIC_API_KEY` in Colab secrets (🔑 sidebar).
"""),
    code(SETUP),
    code('''#@title Parameters
N_LABEL = 1500        #@param {type:"integer"}
REVIEW_FRAC = 0.15    #@param {type:"number"}
N_GOLD = 100          #@param {type:"integer"}
MAX_USD = 20.0        #@param {type:"number"}
LABELER = "claude-sonnet-5-5"  #@param ["claude-sonnet-5-5", "claude-haiku-4-5"]
REVIEWER = "claude-opus-5-5"   #@param ["claude-opus-5-5", "claude-sonnet-5-5"]
#@markdown `api`: Message Batches with your key. `claude-code`: no key — an agent session (Claude Code + colab-mcp)
#@markdown reads `DATA/tasks_*.jsonl`, labels with cheap subagents, reviews, and writes `DATA/labels_*.jsonl`.
LABEL_MODE = "claude-code"  #@param ["claude-code", "api"]'''),
    code('''if LABEL_MODE == "api":
    from google.colab import userdata
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")'''),
    code('''#@title Select frames worth paying for
import glob, json, numpy as np
from vtcore import load_frames
from vtm.labeling.heuristics import frame_confidence
from vtm.labeling.select import select_frames

frames, hroles, confs = [], [], []
for p in sorted(glob.glob(f"{DATA}/weak_*.npz")):
    fr, ex = load_frames(p); frames += fr; hroles += ex["roles"]; confs += [frame_confidence(r, c.astype(np.float32)) for r, c in zip(ex["roles"], ex["conf"])]
# optional: after 03 has run, use model confidence instead (active learning round 2+)
if os.path.exists(f"{DATA}/model_conf.json"):
    confs = json.load(open(f"{DATA}/model_conf.json")); print("using model confidences")
idx = select_frames(frames, confs, N_LABEL, roles=hroles)
sel = [frames[i] for i in idx]
print(len(sel), "selected of", len(frames))'''),
    code('''#@title Cost estimate + budget gate
from vtm.labeling import llm
est = llm.estimate_cost(sel, LABELER)
est_review = llm.estimate_cost(sel[: int(len(sel) * REVIEW_FRAC) + N_GOLD], REVIEWER, out_tokens=1500)
print("labeler:", est); print("reviewer (upper bound):", est_review)
assert est["usd"] + est_review["usd"] <= MAX_USD, "over budget: lower N_LABEL"'''),
    code('''#@title Stage 3 (claude-code mode) · export tasks, then wait for the agent to write labels
if LABEL_MODE == "claude-code":
    n = llm.export_tasks({f"f{i}": frames[i] for i in idx}, f"{DATA}/tasks_label.jsonl")
    open(f"{DATA}/LABELING_GUIDE.md", "w").write(llm.guide_markdown())
    print(n, "tasks →", f"{DATA}/tasks_label.jsonl", "· the agent writes", f"{DATA}/labels_label.jsonl")'''),
    code('''#@title Stage 3 · label with the cheap model (Batch API, or import the agent's labels)
JOBS = f"{DATA}/jobs.json"
jobs = json.load(open(JOBS)) if os.path.exists(JOBS) else {}
if LABEL_MODE == "claude-code":
    labels = llm.import_labels(f"{DATA}/labels_label.jsonl"); print(len(labels), "labels imported")
else:
    reqs = {f"f{i}": llm.labeler_request(frames[i], LABELER) for i in idx}
    if "label" not in jobs:  # resumable: re-running never double-submits
        job = llm.submit_batch(reqs, max_usd=MAX_USD, est=est)
        jobs["label"] = {"id": job.id, "ids": job.custom_ids}; json.dump(jobs, open(JOBS, "w"))
    job = llm.BatchJob(jobs["label"]["id"], jobs["label"]["ids"])
    llm.wait_batch(job)
    labels, usage = llm.collect_batch(job)
    pi, po = llm.PRICES[LABELER]
    print(len(labels), "labeled", usage, f"≈ ${0.5 * (usage['input'] * pi + usage['output'] * po + usage['cache_read'] * pi * 0.1) / 1e6:.2f}")'''),
    code('''#@title Stage 4 · route disagreements (+ a gold set) to the reviewer
import random
from vtm.labeling.select import disagreement
cheap = {int(k[1:]): llm.rasterize(v["regions"], frames[int(k[1:])].shape, frames[int(k[1:])]) for k, v in labels.items()}
dis = {i: disagreement(cheap[i], hroles[i]) for i in cheap}
by_dis = sorted(dis, key=lambda i: -dis[i][0])
review_ids = by_dis[: int(len(cheap) * REVIEW_FRAC)]
rest = [i for i in cheap if i not in set(review_ids)]
gold_ids = random.Random(0).sample(rest, min(N_GOLD, len(rest)))
print("review:", len(review_ids), "gold:", len(gold_ids), "median disagreement", np.median([d for d, _ in dis.values()]).round(3))
if LABEL_MODE == "claude-code":
    jobs["review"] = {"gold": gold_ids}; json.dump(jobs, open(JOBS, "w"))
    rpath = f"{DATA}/labels_review.jsonl"
    if not os.path.exists(rpath):
        llm.export_tasks({f"f{i}": frames[i] for i in review_ids + gold_ids}, f"{DATA}/tasks_review.jsonl",
                         proposals=labels, disagree_rows={f"f{i}": dis[i][1] for i in review_ids + gold_ids})
        raise SystemExit(f"exported review tasks → {DATA}/tasks_review.jsonl; re-run this cell once {rpath} exists")
    reviewed = llm.import_labels(rpath); print(len(reviewed), "reviews imported")
elif "review" not in jobs:
    rreqs = {f"f{i}": llm.review_request(frames[i], labels[f"f{i}"], dis[i][1], REVIEWER) for i in review_ids + gold_ids}
    job = llm.submit_batch(rreqs)
    jobs["review"] = {"id": job.id, "ids": job.custom_ids, "gold": gold_ids}; json.dump(jobs, open(JOBS, "w"))
if LABEL_MODE == "api":
    rjob = llm.BatchJob(jobs["review"]["id"], jobs["review"]["ids"])
    llm.wait_batch(rjob)
    reviewed, rusage = llm.collect_batch(rjob)
    print(len(reviewed), "reviewed", rusage)'''),
    code('''#@title How good is the cheap labeler? (Sonnet vs Opus on the random gold set)
from vtm.metrics import confusion, report
gold_ids = jobs["review"]["gold"]
good = [i for i in gold_ids if f"f{i}" in reviewed]
opus = {i: llm.rasterize(reviewed[f"f{i}"]["regions"], frames[i].shape, frames[i]) for i in [int(k[1:]) for k in reviewed]}
rep_cheap = report(confusion([cheap[i] for i in good], [opus[i] for i in good]))
rep_heur = report(confusion([hroles[i] for i in good], [opus[i] for i in good]))
print(f"cheap labeler vs reviewer: mIoU {rep_cheap['macro_iou']:.3f}   heuristics vs reviewer: mIoU {rep_heur['macro_iou']:.3f}")'''),
    code('''#@title Save labeled shards (gold set held out for 04)
from vtcore import save_frames
from vtcore.roles import APP_ID
def pack(ids, src):
    fs, rs, cs = [], [], []
    for i in ids:
        lab = reviewed.get(f"f{i}") or labels.get(f"f{i}")
        if lab is None: continue
        f = frames[i]; f.meta.update(source="llm", app=APP_ID.get(lab.get("app"), APP_ID["other"]), label_src=src(i))
        fs.append(f); rs.append(opus.get(i, cheap.get(i))); cs.append(np.ones(f.shape, np.float16))
    return fs, rs, cs
train_ids = [i for i in cheap if i not in set(gold_ids)]
random.Random(1).shuffle(train_ids)
n_val = max(50, len(train_ids) // 10)
src = lambda i: "reviewed" if f"f{i}" in reviewed else "labeler"
for name, ids in [("llm_val", train_ids[:n_val]), ("llm_train", train_ids[n_val:]), ("gold_test", good)]:
    fs, rs, cs = pack(ids, src); save_frames(f"{DATA}/{name}.npz", fs, {"roles": rs, "conf": cs}); print(name, len(fs))'''),
    code('''#@title Spot-check: heuristics vs cheap labeler vs reviewer
from vtm.viz import compare_html, show
for i in by_dis[:3]:
    maps = {"heuristics": hroles[i], LABELER: cheap[i]}
    if i in opus: maps[REVIEWER] = opus[i]
    show(compare_html(frames[i], maps))'''),
])

# ---------------------------------------------------------------------------
nb("03_train_vtm.ipynb", [
    md("""
# 03 · Train the VTM

Per-cell segmentation transformer (conv stem + axial row/column attention, ~2M params) trained on
`synth_train` (exact) + `weak_*` (heuristic, confidence-weighted ×0.5) + `llm_train` (×2 oversampled).
Evaluated on the held-out reviewer-labeled **gold set** against the heuristic baseline, then exported
to ONNX for in-browser streaming (onnxruntime-web / WebGPU). Use a **T4 GPU** runtime.
"""),
    code(SETUP),
    code('''#@title Parameters
EPOCHS = 8      #@param {type:"integer"}
BATCH = 16      #@param {type:"integer"}
D_MODEL = 128   #@param {type:"integer"}
LAYERS = 4      #@param {type:"integer"}
USE_WEAK = True #@param {type:"boolean"}'''),
    code('''import glob, json
have = lambda n: os.path.exists(f"{DATA}/{n}.npz")
train_shards = [f"{DATA}/synth_train.npz"]
if USE_WEAK: train_shards += sorted(glob.glob(f"{DATA}/weak_*.npz"))
if have("llm_train"): train_shards += [f"{DATA}/llm_train.npz"] * 2
val_shards = [f"{DATA}/synth_val.npz"] + ([f"{DATA}/llm_val.npz"] if have("llm_val") else [])
print(train_shards, val_shards)'''),
    code('''from vtm.train import train
model = train(train_shards, val_shards, out_dir=f"{DATA}/runs/vtm", epochs=EPOCHS, batch_size=BATCH, d=D_MODEL, layers=LAYERS)'''),
    code('''#@title Evaluate: model vs heuristics (gold = reviewer labels; synth = exact)
from vtcore import load_frames
from vtm.train import predict
from vtm.labeling.heuristics import label
from vtm.metrics import confusion, report
results = {}
for split in ["gold_test", "synth_val"]:
    if not have(split): continue
    fr, ex = load_frames(f"{DATA}/{split}.npz")
    pm = [predict(model, f)[0] for f in fr]
    ph = [label(f)[0] for f in fr]
    results[split] = {"model": report(confusion(pm, ex["roles"])), "heuristics": report(confusion(ph, ex["roles"]))}
    print(split, {k: round(v["macro_iou"], 3) for k, v in results[split].items()})
json.dump(results, open(f"{DATA}/runs/vtm/eval.json", "w"), indent=1)'''),
    code('''#@title Active learning: model confidence on the unlabeled pool → feed back into 02
import numpy as np
from vtcore import load_frames
conf = []
for p in sorted(glob.glob(f"{DATA}/weak_*.npz")):
    for f in load_frames(p)[0]:
        r, c, _ = predict(model, f)
        m = r != 0
        conf.append(float(c[m].mean()) if m.any() else 1.0)
json.dump(conf, open(f"{DATA}/model_conf.json", "w"))
print("pool mean confidence", np.mean(conf).round(3), "→ re-run 02 to label the least-confident frames")'''),
    code('''#@title Export ONNX (browser / edge inference)
from vtm.train import export_onnx
export_onnx(model, f"{DATA}/runs/vtm/vtm.onnx")
!ls -la {DATA}/runs/vtm'''),
], gpu=True)

# ---------------------------------------------------------------------------
nb("04_visualize_results.ipynb", [
    md("""
# 04 · Visualise results

Companion to 03: metrics, error gallery, and the end-to-end **terminal → A2UI** stream —
how many bytes the incremental A2UI stream needs per keyframe vs. the raw VT stream and a naive
full re-render, plus an interactive replay with **native** and **faithful** (style-layer) renders.
"""),
    code(SETUP),
    code('''import json, glob, numpy as np, matplotlib.pyplot as plt
from vtcore import load_frames
from vtm.train import load, predict
from vtm.labeling.heuristics import label
from vtm.metrics import confusion, report
from vtm.viz import plot_iou, plot_confusion, compare_html, show, replay
model = load(f"{DATA}/runs/vtm/vtm.pt", device="cuda" if __import__("torch").cuda.is_available() else "cpu")
hist = json.load(open(f"{DATA}/runs/vtm/history.json"))
split = "gold_test" if os.path.exists(f"{DATA}/gold_test.npz") else "synth_val"
fr, ex = load_frames(f"{DATA}/{split}.npz"); gold = ex["roles"]
pm = [predict(model, f)[0] for f in fr]; ph = [label(f)[0] for f in fr]
rm, rh = report(confusion(pm, gold)), report(confusion(ph, gold))
print(split, "model mIoU", round(rm["macro_iou"], 3), "| heuristics", round(rh["macro_iou"], 3))'''),
    md("## Training curve + per-class IoU"),
    code('''fig, ax = plt.subplots(1, 2, figsize=(16, 3.5), gridspec_kw={"width_ratios": [1, 3]})
ax[0].plot([h["loss"] for h in hist], label="train loss")
if "macro_iou" in hist[0]: ax[0].plot([h["macro_iou"] for h in hist], label="val mIoU")
ax[0].legend(); ax[0].set_xlabel("epoch")
plot_iou({"VTM": rm, "heuristics": rh}, ax=ax[1]); plt.tight_layout()'''),
    code('''fig, ax = plt.subplots(1, 2, figsize=(15, 6))
plot_confusion(confusion(pm, gold), ax[0], "VTM"); plot_confusion(confusion(ph, gold), ax[1], "heuristics"); plt.tight_layout()'''),
    md("## Error gallery — worst frames for the model"),
    code('''err = [float(((p != g) & ((p > 0) | (g > 0))).sum() / max(((p > 0) | (g > 0)).sum(), 1)) for p, g in zip(pm, gold)]
for i in np.argsort(err)[::-1][:4]:
    print(f"frame {i}  error {err[i]:.2f}")
    show(compare_html(fr[i], {"gold": gold[i], "VTM": pm[i], "heuristics": ph[i]}))'''),
    md("## Terminal → A2UI stream\nPick a real recording (or a synthetic one) and stream it through VTM → compiler → reconciler."),
    code('''from vtcore import load_cast
from vtm.synth import synth_cast
from vtm.stream import cast_to_a2ui
from vtcore.emulator import keyframes
paths = sorted(glob.glob("/content/recordings/**/record.*", recursive=True))
cast = load_cast(paths[7]) if paths else synth_cast(42)
labeler = lambda f: predict(model, f)[0]
msgs, stats = cast_to_a2ui(cast, labeler, max_frames=120)
kf = list(keyframes(cast, max_frames=120))
vt, inc, full = (np.array([s[k] for s in stats]) for k in ("vt_bytes", "a2ui_bytes", "full_bytes"))
fig, ax = plt.subplots(1, 2, figsize=(16, 3.5))
ax[0].plot(full, label="A2UI full re-render"); ax[0].plot(inc, label="A2UI delta (phosphene)"); ax[0].plot(vt, label="raw VT bytes", alpha=.6)
ax[0].set_yscale("log"); ax[0].set_xlabel("keyframe"); ax[0].set_ylabel("bytes"); ax[0].legend()
ax[1].plot(np.cumsum(full), label="full"); ax[1].plot(np.cumsum(inc), label="delta"); ax[1].plot(np.cumsum(vt), label="VT"); ax[1].legend(); ax[1].set_title("cumulative bytes")
print(f"delta/full = {inc[1:].sum() / max(full[1:].sum(), 1):.2%}   structural updates on {np.mean([s['structural'] for s in stats[1:]]):.0%} of keyframes")'''),
    code('''#@title Interactive replay (terminal | A2UI native | A2UI faithful)
from vttui.reconcile import Reconciler
rec, roles_seq = Reconciler(), []
for f, m in zip(kf, msgs):
    rec.step(f, labeler(f)); roles_seq.append(rec.roles)
replay(kf, roles_seq, msgs)'''),
    code('''#@title Export stream: JSONL + stylesheet (open in packages/viewer/index.html)
from vttui.reconcile import to_jsonl
open(f"{DATA}/stream.a2ui.jsonl", "w").write("".join(to_jsonl(m) for m in msgs))
open(f"{DATA}/stream.css", "w").write(rec.stylesheet())
print(f"{DATA}/stream.a2ui.jsonl")'''),
], gpu=True)

print("notebooks written")

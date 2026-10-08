# phosphene

> *phosphene (n.)* — light you see that isn't there. Here: UI that the terminal never drew.

**phosphene** is an experiment in a **Virtual Terminal Model (VTM)**. It learns to turn raw
VT/TUI byte streams (shells, htop, vim, installers, dialogs…) into a structured,
**incremental [A2UI](https://a2ui.org) stream**. When the screen changes, only the changed
components and data are sent, the same way a virtual DOM and signals work in a modern frontend.

```
 asciicast / pty bytes
        │  vtcore: VT emulation (pyte) → keyframes (screen settled) + damage mask
        ▼
 Frame  (H×W chars · fg · bg · attrs · cursor · damage)
        │  vtm: per-cell segmentation transformer (axial attention, ~2M params, ONNX/WebGPU)
        ▼
 Role map (15 roles: prompt · input · border · title · status_bar · menu_item · selected ·
           table · progress · code · log · error · key_hint · text · blank)
        │  vttui: deterministic compiler  — regions → boxes → layout bands → A2UI components
        │         + style layer (cell-grid area, ANSI palette tokens) for faithful rendering
        ▼
 Reconciler (keeps last UI; undamaged cells keep their roles → no flicker)
        │  updateComponents  only when *structure* changes
        │  updateDataModel   JSON-pointer patches for *content* (one log line, one %, one keystroke)
        ▼
 A2UI v0.9 JSONL  →  any A2UI renderer (native look)  or  phosphene viewer (faithful look)
```

The learned part is small and a pure function of the screen. Everything after it is
deterministic code, so the system as a whole is a streaming transformer you can run in a browser.

## Monorepo

| package | what |
|---|---|
| [`packages/vtcore`](packages/vtcore) | asciicast v1/v2/v3 parsing, VT emulation, keyframes, dedupe signatures, model features, compact LLM rendering, shared role vocabulary |
| [`packages/vttui`](packages/vttui) | **vt → A2UI**: compiler, reconciler (delta stream), style layer + CSS, reference renderer |
| [`packages/vtm`](packages/vtm) | the model, synthetic TUI generator (exact labels), heuristic labelers, LLM labeling (Batch API), active selection, training, ONNX export, notebook viz |
| [`packages/viewer`](packages/viewer) | zero-dependency HTML player for exported `.a2ui.jsonl` streams (native / faithful toggle) |
| [`notebooks`](notebooks) | Colab pipeline, generated from `notebooks/_build.py` |

## Run it (Colab)

| | notebook | runtime |
|---|---|---|
| 1 | [01 · data prep](https://colab.research.google.com/github/drksci/phosphene/blob/main/notebooks/01_data_prep.ipynb): subset of 79k asciinema recordings → keyframes → dedupe → weak + synthetic labels | CPU |
| 2 | [02 · LLM labeling](https://colab.research.google.com/github/drksci/phosphene/blob/main/notebooks/02_llm_labeling.ipynb): select → Sonnet (batch) → Opus review on disagreements → gold set | CPU + `ANTHROPIC_API_KEY` |
| 3 | [03 · train VTM](https://colab.research.google.com/github/drksci/phosphene/blob/main/notebooks/03_train_vtm.ipynb): train, evaluate against heuristics, active learning signal, ONNX export | T4 |
| 4 | [04 · visualise results](https://colab.research.google.com/github/drksci/phosphene/blob/main/notebooks/04_visualize_results.ipynb): IoU, confusion, error gallery, bytes/keyframe, interactive A2UI replay | T4 |

Notebooks 1, 3 and 4 need no API key. Training on synthetic + weak labels alone already works.
Notebook 2 adds real-world precision.

## Cheapest labeling that works

Most of the labels cost nothing. LLM tokens go only where the free labels are weak. Details are in
[docs/LABELING.md](docs/LABELING.md).

1. **Dedupe** keyframes by layout signature (letters/digits abstracted away). This removes most frames.
2. **Synthetic TUIs** rendered through the real emulator give *exact* per-cell labels for free.
3. **Heuristic labeling functions** label every real frame for free and give a per-cell confidence.
4. **Active selection** picks about 1–2k frames for the LLM: the lowest-confidence ones plus a k-center diversity sample.
5. **Claude Sonnet 5.5** labels them through the Batch API (50% off). It returns compact regions rather than per-cell labels, with a cached system prompt and thinking off. That's about $0.003 per frame.
6. **Claude Opus 5.5** reviews only the frames where Sonnet and the heuristics disagree (about 15%), plus a 100-frame gold test set.
7. After training, the model's own uncertainty picks the next round of frames.

A first round of about 1,500 frames should cost roughly **$5–10**. The notebook refuses to submit if its estimate is over the `MAX_USD` budget.

## Style layer

Each A2UI component carries a `style` with its `area` (terminal cell rectangle), `fg`/`bg` (ANSI
palette tokens) and `bold`/`inverse`. `createSurface.theme` carries the palette and grid size.
`vttui.stylesheet()` (or the viewer) renders a **faithful** view with the same proportions and
colours as the source terminal. A plain A2UI renderer can ignore all of it and render natively.
The catalog extension is described in [docs/catalog/vttui_v0_1.json](docs/catalog/vttui_v0_1.json).

## Dev

```bash
uv sync && uv run pytest
```

CI runs the tests on every push. The heavy work (dataset, training) runs only in Colab.

## Status

This is a prototype. What's next is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#roadmap): temporal context, in-browser
ONNX streaming, distillation to WGSL shaders, and A2UI → keystroke actions for driving TUIs from the UI.

# Architecture

## 1. vtcore — from bytes to frames

- **Casts**: asciicast v1 (relative delays, one JSON document), v2 (absolute times) and v3
  (relative intervals, `term.cols/rows`, resize events) all become one `Event(t, code, data)` list.
- **Emulation**: `pyte` with a dense-array mirror. Only the rows in pyte's dirty set are re-read.
  Colours are quantised to the 16 ANSI colours plus default and other (256/truecolour map to the
  nearest one).
- **Keyframes**: a frame is emitted when output pauses for ≥150 ms (the screen *settled*) or at
  least every 3 s during continuous output. Unchanged screens are skipped. Each frame carries a
  **damage** mask, the cells that changed since the previous keyframe.
- **Dedupe**: `layout_signature` hashes per-row runs of character *classes* (letter, box, block,
  highlight) with columns quantised. The same layout with different text collapses to one entry.

## 2. vtm — the learned part

Per-cell features are a char id from a 288-symbol structural vocabulary (ASCII, each box-drawing
and block glyph, braille, CJK and nerd-font buckets), fg, bg, attribute bits, damage and cursor,
plus position embeddings for row, row-from-bottom, column and column-from-right.

The network is a depthwise conv stem for local glyph shapes, then N **axial blocks** (row
attention, then column attention, then an MLP). Each cell sees its whole row and column at
O(HW·(H+W)) cost. That's enough to find aligned table columns, box edges, full-width bars and
bottom-row key hints. The heads are per-cell role logits plus a frame-level app kind (shell,
editor, monitor, menu, installer, repl, pager, dashboard, other).

The model is ~2M params at d=128 with 4 layers. It trains in minutes on a T4 and exports to
ONNX with dynamic H/W.

## 3. vttui — the deterministic part

**Compiler** (`compile.py`)
1. Row runs of equal role (absorbing small blank gaps) merge vertically into regions. Selected
   rows merge into the list or table they belong to, as a per-line `selected` flag.
2. Border cells become flood-filled components. Rectangles become **boxes** (`Card`) and single
   lines become `Divider`s. A title on a box's top edge becomes the card's title.
3. Each region is assigned to its smallest enclosing box. Within a container, children whose row
   spans overlap form a **band** and are laid out side by side in a `Row`.
4. Each region maps to A2UI components:

| role | A2UI |
|---|---|
| text / log / error / code / table / menu_item (+selected) | `List` with a template item `Text` bound to `/r/<id>/rows` (one data entry per terminal line) |
| prompt (+input) | `Row[Text(prompt), TextField(value)]` (`Text` once the line is history) |
| status_bar / title | `Text` (`caption` with `tone: bar` / `h3`) |
| progress | `Column` of template `Progress{value,label}`, with the value parsed from `%` or bar fill |
| key_hint | `Row` of template `Button{shortcut, label}` with a `key` action |
| box / rule | `Card` (+ title) / `Divider` |

IDs are **keys derived from position in the tree** (`box0.table1`). This works like keyed
children in React: the structure stays stable while the content changes.

**Reconciler** (`reconcile.py`)
- *Stabilisation*: cells outside the damage mask keep the previous frame's role, so the model can
  only change its mind where the terminal actually changed.
- `updateComponents` carries only the components whose JSON changed.
- `updateDataModel` carries minimal JSON-pointer patches: leaf upserts and list appends (a log
  line is one small message), with parent replacement when churn is heavy. Data is sent before
  structure, so bound components never render empty.
- `createSurface` (with the theme) is sent on the first frame and on every resize.

**Style layer** (`style.py`): per-component `area` (in cells), palette tokens and bold/inverse,
plus a theme (palette, grid, font metrics) and a CSS generator for the faithful renderer.

## 4. Verification

`tests/` checks that, for every synthetic scene, applying the incremental stream to an empty
surface reproduces exactly the components and data of a full compile. In other words, the delta
stream is lossless. It also checks that deltas are much smaller than full re-renders.
Notebook 04 measures bytes per keyframe on real recordings.

## Roadmap

- **Temporal context**: feed the previous frame's predicted roles as an input channel, and add a
  recurrent or windowed-attention variant for spinners and scrolling.
- **In-browser streaming**: xterm.js parser → features → onnxruntime-web (WebGPU) → a JS port of
  vttui. All of `vttui` is pure functions over arrays, so the port is mechanical.
- **Shader distillation**: distil the axial transformer into a conv-only student and compile it
  to WGSL compute shaders, for per-frame segmentation at screen refresh rate.
- **Round-tripping**: map A2UI actions (Button `key`, TextField edits, list selection) back to
  keystrokes, so the generated UI can *drive* the TUI.
- **Richer semantics**: table column parsing (`Row` per cell), tree views, tabs, modal detection,
  and form fields in dialog/whiptail screens.

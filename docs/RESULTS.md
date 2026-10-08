# Results log

All runs are on Colab (T4, 2 vCPU). mIoU is the macro IoU over the 14 non-blank roles.

## Corpus (2026-10-08)

| | |
|---|---|
| recordings processed | 4,000 (size-filtered subset of the 79k-recording asciinema archive) |
| unique keyframes after layout dedupe | **44,168** |
| synthetic frames (exact labels) | 14,223 train · 1,466 val |
| agent-labeled real frames (round 1) | 600 (Sonnet subagents) · 90 reviewed (Opus subagents) |

## Label quality (90 reviewed real frames, Opus review = reference)

| labeler | mIoU | accuracy (non-blank cells) |
|---|---|---|
| heuristics | 0.195 | — |
| Sonnet labeler (before review) | 0.743 | 0.781 |

These frames were *selected* for maximal heuristic/labeler disagreement, so they are the hard cases.
Most common labeler errors found in review:
- `input` regions running past the typed text (now clipped automatically in `rasterize`)
- grey autosuggestions labeled as input
- off-by-one column bounds
- aligned output labeled as text instead of table
- progress bars inside logs

## Model runs

| run | data | epochs | synth val mIoU | real val+test mIoU | notes |
|---|---|---|---|---|---|
| smoke | synth only | 1 | 0.992 | — | loop/eval check; synth is in-distribution and easy |
| v1 | synth + 16k weak + 460 LLM ×3 | 1 (stopped) | — | 0.352 (acc 0.576) | weak labels dominate the real-frame data; the model copies the heuristics |
| v2 | synth + 4k weak + 460 LLM ×10 | 5 | | **0.511** (macro-F1 0.648, acc 0.73) | best epoch 3 of 5; ~9 min/epoch |

Throughput: ~150–300 ms/step at batch 16 (≤64×200 crops). The GPU is the bottleneck: a large
per-cell activation map in memory-bound kernels. The attention padding mask is not a factor (profiled).

## Gateway (model v2, 2026-10-08)

193 random recordings, ≤150 settled screens each, one shared template cache (`docs/bench_v2.json`):
13,967 screens · **40% locked** · 0.70 model calls/screen · A2UI 106.3 MB vs VT 4.2 MB (25×) ·
incremental A2UI = 43% of re-sending the full UI every screen.

Demo traces (`python -m vtm.demos`, from each app's first screen):

| app | locked | model calls/screen | VT KB | A2UI KB | full KB |
|---|---|---|---|---|---|
| less | 92% | 0.11 | 14.9 | 227.4 | 905.5 |
| dialog | 89% | 0.12 | 69.2 | 287.4 | 839.4 |
| vim | 77% | 0.31 | 22.2 | 338.0 | 2,009.2 |
| emacs | 68% | 0.34 | 55.3 | 611.3 | 1,882.0 |
| tig (93 screens) | 52% | 0.51 | 33.6 | 1,704.1 | 3,244.3 |
| top | 26% | 0.84 | 34.0 | 1,582.8 | 4,074.1 |
| htop | 24% | 0.78 | 41.9 | 1,456.6 | 2,936.1 |
| nano | 14% | 0.94 | 16.5 | 3,697.2 | 8,595.1 |

The wire format is the weak point: JSON components and the styled runs of unmatched regions are far
larger than VT. Published at https://drksci.com/labs-phosphene.

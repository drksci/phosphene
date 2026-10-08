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
| v2 | synth + 4k weak + 460 LLM ×10 | 5 | | | running |

Throughput: ~150–300 ms/step at batch 16 (≤64×200 crops). The GPU is the bottleneck: a large
per-cell activation map in memory-bound kernels. The attention padding mask is not a factor (profiled).

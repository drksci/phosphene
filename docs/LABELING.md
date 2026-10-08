# Labeling strategy: maximum signal per token

The goal is per-cell role labels for real terminal screens, at the lowest possible token spend.

## Label sources (cheapest first)

| source | coverage | quality | cost |
|---|---|---|---|
| **synthetic TUIs** (`vtm.synth`) | unlimited | exact | free |
| **heuristics** (`vtm.labeling.heuristics`) | every real frame | weak, with per-cell confidence | free |
| **Sonnet 5.5 labeler** (`vtm.labeling.llm`) | ~1–2k selected frames / round | good | ≈ $0.003 / frame (batch) |
| **Opus 5.5 reviewer** | disagreement frames (~15%) + 100-frame gold set | best available | ≈ $0.02 / frame (batch) |

Training weights: synthetic and LLM labels count 1.0. Weak labels count 0.5 × the rule's
confidence. LLM-labeled shards are also oversampled 2×.

## Why region output (not per-cell)

A 24×80 screen has 1,920 cells. Asking for a role per cell costs about 2–4k output tokens.
Asking for **rectangles** (`{role, r0, c0, r1, c1}`, later ones override earlier ones) costs
about 200–400 tokens. `rasterize()` turns them back into per-cell labels. Whitespace inside a
region stays blank, except for roles that own their whitespace (status bars, selections, inputs,
progress bars).

## Prompt economics

- Frame rendering (`vtcore.render_for_llm`): rows are prefixed with their index, with a column
  ruler. Trailing spaces are stripped and runs of empty rows are collapsed. Colours and attributes
  are a short span list on the side, not inline escape codes. A typical frame is 0.5–1.2k tokens.
- The system prompt is identical for every request and marked `cache_control`.
- **Batch API**: 50% off. Results are keyed by `custom_id` and are resumable (`jobs.json`).
- Sonnet runs with `thinking: {type: "between_tools"}` and `effort: low`. There are no tools, so
  no thinking tokens are billed. Structured outputs guarantee valid JSON.
- A hard budget gate (`MAX_USD`) checks the estimate before submitting.

## Selection

`select_frames()` spends 60% of the budget on the **lowest-confidence** frames: heuristic
confidence in round 1, the model's own confidence from round 2 on. It spends the remaining 40%
on **k-center diversity** over a cheap layout descriptor (ink, box and highlight profiles plus a
role histogram). That keeps rare layouts like dashboards, dialogs and REPLs represented.

## Review routing

After the cheap labeler, frames are ranked by Sonnet-vs-heuristics disagreement. The top
`REVIEW_FRAC` go to Opus together with the rows where the two disagree. A random gold set is
also reviewed. It's held out, and it measures both the model and the cheap labeler itself. If
Sonnet's mIoU against Opus on the gold set is high, lower `REVIEW_FRAC` next round. If it's low,
raise it.

## Cost estimate for round 1 (1,500 frames)

| | tokens | $ |
|---|---|---|
| Sonnet input (frames + cached system) | ~1.5M | ~1.5 |
| Sonnet output (regions) | ~0.5M | ~2.6 |
| Opus review (~325 frames incl. gold) | ~0.6M in / ~0.5M out | ~5 |
| **total** | | **≈ $9** |

`estimate_cost()` prints the real numbers for your selection before anything is submitted.

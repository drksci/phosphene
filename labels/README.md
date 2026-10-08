# Labels

Per-cell role labels for real terminal keyframes (from asciinema recordings), produced in-session by
Claude Code: Sonnet subagents label, Opus subagents review disagreements.

`roundN/`
- `frames.npz` — the labeled frames (`vtcore.load_frames`), with `heuristic_roles` for comparison
- `ids.json` — task id → index in the round's selection pool
- `label_KK.jsonl` — labeler output, one `{id, app, regions}` per line (rasterize with `vtm.labeling.llm.rasterize`)
- `review_KK.jsonl` — reviewer-corrected labels (when present; these supersede `label_*`)
- `review_split.json` — which ids were routed to review, and the held-out gold test set
- `tasks_label.jsonl`, `LABELING_GUIDE.md` — exactly what the labelers saw

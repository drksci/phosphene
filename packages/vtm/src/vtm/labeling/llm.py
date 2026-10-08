"""LLM labeling: frame -> region list, via the Claude Message Batches API (50% off).

Cost levers (in order of impact):
  1. Only send frames that need it: heuristics label everything for free; `select.py` picks the
     low-confidence / high-diversity frames (typically 2-5% of the deduplicated corpus).
  2. Region output, not per-cell output: ~10-30 short JSON objects per frame (~200-400 tokens).
  3. Compact frame rendering: trailing spaces stripped, empty-row runs collapsed, styles as spans.
  4. Batch API (-50%) + a frozen system prompt marked for prompt caching.
  5. Thinking off for the bulk labeler (Sonnet `between_tools` + low effort); the stronger reviewer
     only sees the frames where labeler, heuristics and model disagree.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from vtcore.encode import render_for_llm
from vtcore.frame import Frame
from vtcore.roles import APP_KINDS, FILL_ROLES, ROLE_ID, ROLES

LABELER_MODEL = "claude-sonnet-5-5"
REVIEWER_MODEL = "claude-opus-5-5"

# $ per 1M tokens (input, output); batch = 50% of these. Cache reads ~10% of input.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-4-5": (1.0, 5.0)}

ROLE_GUIDE = """\
blank: empty background (never emit)
text: plain program output / prose
prompt: the shell or REPL prompt prefix only (e.g. "user@host:~$ ", "❯ ", ">>> ")
input: text being typed at the active prompt or into a form field (usually the cursor row)
border: box-drawing frames, separators, rules (┌─┐ │ +---+ =====)
title: headings, panel/window titles (including titles embedded in a frame's top edge)
status_bar: full-width highlighted bar at top/bottom; mode lines; pager status (":", "(END)")
menu_item: an item in a selectable list, menu or file tree (not currently highlighted)
selected: the currently highlighted/selected item or row
table: column-aligned tabular rows (incl. header row), `ls` column listings
progress: progress bars, gauges, meters, spinners, percentages, transfer rates
code: source code / editor buffer, line-number gutters, vim "~" filler rows
log: timestamped or level-prefixed log lines
error: errors, warnings, tracebacks, failure messages
key_hint: key bindings, shortcuts, buttons ("^X Exit", "F1Help", "<  OK  >", "[ Cancel ]", "q quit")"""

SYSTEM = f"""You label terminal screenshots for training a UI-segmentation model.

The user message is one terminal screen: a header with size and cursor, a 2-line column ruler, then
each row as `NN|<row text>` (rows indexed from 00, columns from 0, so the character right after `|`
is column 0). `NN-MM| (empty)` marks a run of empty rows. A `styles:` section lists styled spans as
`r<row> c<from>-<to> <attrs>` (rev = reverse video, bg/fg colours, bold) — reverse/coloured spans
usually mean status bars, selected items, buttons or table headers.

Cover every non-blank cell with exactly one region. A region is a rectangle
{{role, r0, c0, r1, c1}} with inclusive row/column bounds. Prefer few, large regions: one per
log block, table, list or paragraph, not one per line. Only non-blank cells inside a region take
its role, so rectangles may include whitespace; for status_bar, selected, input and progress the
whitespace inside the region also takes the role.
Later regions override earlier ones where they overlap. For a framed box, emit one border region
covering the whole box first, then regions for everything inside it and its title.

Roles:
{ROLE_GUIDE}

Also classify the whole screen as one app kind: {", ".join(APP_KINDS)}.
Respond with JSON only, matching the schema."""

SCHEMA = {
    "type": "object",
    "properties": {
        "app": {"type": "string", "enum": APP_KINDS},
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": [r for r in ROLES if r != "blank"]},
                    "r0": {"type": "integer"}, "c0": {"type": "integer"},
                    "r1": {"type": "integer"}, "c1": {"type": "integer"},
                },
                "required": ["role", "r0", "c0", "r1", "c1"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["app", "regions"],
    "additionalProperties": False,
}

REVIEW_SYSTEM = SYSTEM + """

You are the REVIEWER. You also receive a proposed labeling (from a cheaper labeler) and a list of
rows where an independent heuristic labeler disagreed with it. Return the corrected full labeling
(same schema). Keep regions that are right; fix or replace those that are wrong."""


def _params(model: str, system: str, user: str, max_tokens: int = 4096) -> dict:
    p: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
    }
    if model.startswith("claude-sonnet-5-5"):
        p["thinking"] = {"type": "between_tools"}  # no tools here -> no thinking tokens billed
        p["output_config"]["effort"] = "low"
    elif model.startswith("claude-opus-5-5"):
        p["output_config"]["effort"] = "medium"  # reviewer: thinking stays adaptive (can't be disabled)
    return p


def labeler_request(f: Frame, model: str = LABELER_MODEL) -> dict:
    return _params(model, SYSTEM, render_for_llm(f))


def review_request(f: Frame, proposal: dict, disagree_rows: list[int], model: str = REVIEWER_MODEL) -> dict:
    user = (f"{render_for_llm(f)}\n\nproposed labeling:\n{json.dumps(proposal, separators=(',', ':'))}\n\n"
            f"rows where the heuristic labeler disagrees: {disagree_rows[:40]}")
    return _params(model, REVIEW_SYSTEM, user, max_tokens=16000)


# ---------------------------------------------------------------------------
def rasterize(regions: list[dict], shape: tuple[int, int], f: Frame | None = None) -> np.ndarray:
    """Region list -> per-cell role map (later regions win; whitespace rule as in the prompt)."""
    h, w = shape
    roles = np.zeros((h, w), np.int8)
    for reg in regions:
        rid = ROLE_ID.get(reg.get("role", ""), 0)
        r0, r1 = max(0, int(reg["r0"])), min(h - 1, int(reg["r1"]))
        c0, c1 = max(0, int(reg["c0"])), min(w - 1, int(reg["c1"]))
        if r1 < r0 or c1 < c0:
            continue
        roles[r0:r1 + 1, c0:c1 + 1] = rid
    if f is not None:
        blank = (f.cp == 32) & ~np.isin(roles, list(FILL_ROLES))
        roles[blank] = 0
        # any ink the LLM forgot to cover: plain text
        roles[(f.cp != 32) & (roles == 0)] = ROLE_ID["text"]
    return roles


def estimate_cost(frames: list[Frame], model: str = LABELER_MODEL, batch: bool = True, out_tokens: int = 350) -> dict:
    """Rough upper bound before spending anything (≈3.2 chars/token for this kind of text)."""
    inp = sum(len(render_for_llm(f)) for f in frames) / 3.2
    sys_tok = len(SYSTEM) / 3.2
    pi, po = PRICES.get(model, (5.0, 25.0))
    k = 0.5 if batch else 1.0
    n = len(frames)
    # first request writes the cache (1.25x), the rest read it (0.1x) when it caches
    usd = k * ((inp + sys_tok * (1.25 + 0.1 * max(n - 1, 0))) * pi + n * out_tokens * po) / 1e6
    return {"frames": n, "input_tokens": int(inp + n * sys_tok), "output_tokens": n * out_tokens, "usd": round(usd, 2)}


@dataclass
class BatchJob:
    id: str
    custom_ids: list[str]


def submit_batch(requests: dict[str, dict], max_usd: float | None = None, est: dict | None = None) -> BatchJob:
    """requests: custom_id -> params. Refuses to submit if the estimate exceeds ``max_usd``."""
    import anthropic

    if max_usd is not None and est is not None and est["usd"] > max_usd:
        raise RuntimeError(f"estimated ${est['usd']} exceeds budget ${max_usd}; select fewer frames")
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=[{"custom_id": cid, "params": p} for cid, p in requests.items()])
    return BatchJob(batch.id, list(requests))


def wait_batch(job: BatchJob, poll: int = 30, log=print):
    import anthropic

    client = anthropic.Anthropic()
    while True:
        b = client.messages.batches.retrieve(job.id)
        if b.processing_status == "ended":
            return b
        log(f"{b.processing_status}: {b.request_counts.processing} processing, {b.request_counts.succeeded} done")
        time.sleep(poll)


def collect_batch(job: BatchJob) -> tuple[dict[str, dict], dict]:
    """Returns ({custom_id: {"app", "regions"}}, usage totals). Failed/refused items are skipped."""
    import anthropic

    client = anthropic.Anthropic()
    out, usage = {}, {"input": 0, "output": 0, "cache_read": 0, "failed": 0}
    for res in client.messages.batches.results(job.id):  # results arrive in any order: key by custom_id
        if res.result.type != "succeeded":
            usage["failed"] += 1
            continue
        msg = res.result.message
        u = msg.usage
        usage["input"] += u.input_tokens
        usage["output"] += u.output_tokens
        usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        if msg.stop_reason in ("refusal", "max_tokens"):
            usage["failed"] += 1
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            out[res.custom_id] = json.loads(text)
        except json.JSONDecodeError:
            usage["failed"] += 1
    return out, usage


def label_sync(f: Frame, model: str = LABELER_MODEL) -> dict | None:
    """One-off synchronous labeling (debugging / tiny sets). Server-side refusal fallback enabled."""
    import anthropic

    client = anthropic.Anthropic()
    p = labeler_request(f, model)
    msg = client.beta.messages.create(**p, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if msg.stop_reason in ("refusal", "max_tokens"):
        return None
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return json.loads(text)

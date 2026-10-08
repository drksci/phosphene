/* phosphene replay player.
 *
 * Plays pre-generated traces (vtm/trace.py): for each settled screen ("keyframe") the cells that
 * changed, the role of every cell (from the model, or from a locked template), the gateway state and
 * the A2UI messages the gateway emitted. The console is drawn from cell deltas: this page has no VT
 * parser, which is the point.
 *
 * Layout: pipeline row on top (VT cell grid → roles → A2UI stream); the stage split into the VT
 * screen (native, from cells) and the A2UI client view, in sync; then the A2UI elements
 * row (each top-level element, rendered or as source, scrollable).
 * Embed protocol (same as the drksci Netscraper player, so the site's SessionEmbed drives it):
 *   in:  {type: "netscraper:play" | "netscraper:pause" | "netscraper:seek", t}
 *   out: {type: "netscraper:ready", duration}, {type: "netscraper:ended"}
 */
(() => {
  "use strict";

  const ANSI = ["#1d1d1b", "#d0675f", "#8fb573", "#d8b665", "#6f9fd0", "#b689c6", "#6db8b5", "#d9d5cc",
    "#6a6862", "#e88a80", "#a9cf8d", "#ecd08a", "#93bbe6", "#cfa7dd", "#8fd2cf", "#f6f3ea"];
  const TERM_BG = "#121211", TERM_FG = "#e9e6dd";
  const ROLE_NAMES = ["blank", "text", "prompt", "input", "border", "title", "status bar", "menu item", "selected",
    "table", "progress", "code", "log", "error", "key hint", "not yet matched"];
  const ROLE_TONE = [null, "#8a8780", "#4f8a5f", "#3f74a8", "#b9b6ad", "#8a5fa8", "#46443f", "#b08a3a", "#d0652a",
    "#3f8a96", "#5f9a4a", "#7a68b0", "#77736a", "#c0392b", "#c09a2a", "#cbc7be"];
  const RAW = 15;
  const INK = "#0c0c0b", FAINT = "#8a8780", HAIR = "#cbc7be";
  const REVERSE = 8, BOLD = 1, UNDERLINE = 4;
  const GLOW_MS = 900;

  const $ = (id) => document.getElementById(id);
  const MONO = getComputedStyle(document.documentElement).getPropertyValue("--mono").trim() || "monospace";
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const params = new URLSearchParams(location.search);
  const embedded = window.parent !== window;

  let traces = [], trace = null, F = [], disp = [], total = 0;
  let k = 0, clock = 0, playing = false, speed = 1, mode = "grid", last = 0;
  let H = 0, W = 0, cp, fg, bg, at, age, amp, roles, cursor = [-1, -1];
  let view = "ui", cardState = new Map(), userScroll = 0, prevText = null;
  let cum = { vt: 0, full: 0, inc: 0 };
  let particles = [];
  let surface = null;

  // ── loading ───────────────────────────────────────────────────────────────
  async function boot() {
    const idxUrl = new URL(params.get("index") || "traces/index.json", location.href);
    const idx = await (await fetch(idxUrl, { cache: "no-cache" })).json();
    traces = idx.recordings.map((r) => ({ ...r, file: new URL(r.file, idxUrl).href }));
    traces.forEach((r, i) => {
      const b = document.createElement("button");
      b.textContent = r.label;
      b.title = r.title || r.label;
      b.onclick = () => select(i, playing);
      $("recs").appendChild(b);
    });
    await select(0, false);
    post({ type: "netscraper:ready", duration: total });
    if (!embedded && !reduced) setPlaying(true);
    requestAnimationFrame(loop);
  }

  async function select(i, autoplay) {
    [...$("recs").children].forEach((b, j) => b.classList.toggle("on", i === j));
    const meta = traces[i];
    trace = meta.data || (meta.data = await (await fetch(meta.file)).json());
    F = trace.frames;
    disp = [];
    let d = 0;
    F.forEach((f, j) => {  // compressed time: idle gaps are capped, every keyframe stays on screen a moment
      if (j) d += Math.min(Math.max(f.t - F[j - 1].t, 0.18), 1.4);
      disp.push(d);
    });
    total = d + 1.5;
    seek(0);
    layout();
    setPlaying(autoplay);
  }

  // ── applying the trace ───────────────────────────────────────────────────
  function reset(shape) {
    [H, W] = shape;
    cp = new Uint32Array(H * W).fill(32);
    fg = new Uint8Array(H * W).fill(16);
    bg = new Uint8Array(H * W).fill(16);
    at = new Uint8Array(H * W);
    age = new Float64Array(H * W).fill(-1e9);
    amp = new Float32Array(H * W);
    roles = new Uint8Array(H * W);
  }

  function apply(j, live) {
    const f = F[j];
    if (f.shape[0] !== H || f.shape[1] !== W) reset(f.shape);
    const now = performance.now();
    // a full redraw (clear screen, app start) lights every cell: keep that glow quiet, keep small edits bright
    const strength = Math.max(0.18, Math.min(1, 0.12 * H * W / Math.max(f.cells.length, 1)));
    for (const [r, c, ch, f0, b0, a0] of f.cells) {
      const i = r * W + c;
      cp[i] = ch ? ch.codePointAt(0) : 0;
      fg[i] = f0; bg[i] = b0; at[i] = a0;
      if (live) { age[i] = now; amp[i] = strength; }
    }
    for (const [r, c, n, role] of f.roles) roles.fill(role, r * W + c, r * W + c + n);
    cursor = f.cursor;
    cum.vt += f.vt; cum.full += f.full; cum.inc += f.a2ui;
    for (const m of f.msgs) surface.apply(m);
    if (live) { spawn(f); logMsgs(f.msgs); }
  }

  function seek(target) {
    surface = new Surface();
    $("msgs").textContent = "";
    prevText = null;
    cardState.forEach((st) => st.el.remove());
    cardState.clear();
    cum = { vt: 0, full: 0, inc: 0 };
    particles = [];
    reset(F[0].shape);
    for (let j = 0; j <= target; j++) apply(j, false);
    logMsgs(F[target].msgs);
    k = target;
    clock = disp[target];
    metrics();
    renderA2UI();
  }

  // ── main loop ─────────────────────────────────────────────────────────────
  function loop(ts) {
    const dt = Math.min((ts - (last || ts)) / 1000, 0.1);
    last = ts;
    if (playing && trace) {
      clock += dt * speed;
      let changed = false;
      while (k + 1 < F.length && disp[k + 1] <= clock) { apply(++k, true); changed = true; }
      if (changed) { metrics(); renderA2UI(); }
      if (clock >= total) { setPlaying(false); post({ type: "netscraper:ended" }); }
    }
    drawScreen();
    drawMinis();
    drawFlow();
    $("head").style.left = `${Math.min(clock / total, 1) * 100}%`;
    requestAnimationFrame(loop);
  }

  function setPlaying(p) {
    if (p && clock >= total - 0.01) seek(0);
    playing = p;
    $("play").textContent = p ? "❚❚" : "▶";
    $("play").setAttribute("aria-label", p ? "Pause" : "Play");
  }

  // ── geometry ──────────────────────────────────────────────────────────────
  function fitCanvas(cv) {
    const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
    if (cv.width !== Math.round(r.width * d) || cv.height !== Math.round(r.height * d)) {
      cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(d, 0, 0, d, 0, 0);
    return [ctx, r];
  }
  // console cells fill the whole stage ("cover"); glyphs are centred in their cell
  function consoleGeom() {
    const r = $("screen").getBoundingClientRect();
    const pad = 10, cw = (r.width - 2 * pad) / W, ch = (r.height - 2 * pad) / H;
    return { r, pad, cw, ch, fpx: Math.max(4, Math.min(cw / 0.6, ch / 1.12, 18)) };
  }
  // a miniature of the grid inside a pipeline box, cells in terminal proportion
  function miniGeom(cv) {
    const r = cv.getBoundingClientRect(), pad = 6;
    const s = Math.min((r.width - 2 * pad) / W, (r.height - 2 * pad) / (H * 1.6));
    const w = W * s, h = H * s * 1.6;
    return { r, x: (r.width - w) / 2, y: (r.height - h) / 2, cw: s, ch: s * 1.6 };
  }
  function layout() { drawTicks(); fitClient(); }
  addEventListener("resize", layout);

  // ── console ───────────────────────────────────────────────────────────────
  const color = (idx, dflt) => (idx < 16 ? ANSI[idx] : dflt);
  function drawScreen() {
    if (!H) return;
    const [ctx] = fitCanvas($("screen"));
    const { r, pad, cw, ch, fpx } = consoleGeom();
    const now = performance.now();
    ctx.fillStyle = TERM_BG;
    ctx.fillRect(0, 0, r.width, r.height);
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    const grid = mode === "grid", showRoles = mode === "roles";
    for (let row = 0; row < H; row++) {
      for (let c = 0; c < W; c++) {
        const i = row * W + c;
        let f0 = color(fg[i], TERM_FG), b0 = color(bg[i], null);
        if (at[i] & REVERSE) { const t = f0; f0 = b0 || TERM_BG; b0 = t; }
        const x = pad + c * cw, y = pad + row * ch;
        if (b0) { ctx.fillStyle = b0; ctx.fillRect(x, y, cw + 0.4, ch + 0.4); }
        if (showRoles && roles[i]) {
          if (roles[i] === RAW) {
            ctx.strokeStyle = "rgba(203,199,190,.30)"; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x, y + ch); ctx.lineTo(x + cw, y); ctx.stroke();
          } else { ctx.fillStyle = ROLE_TONE[roles[i]] + "60"; ctx.fillRect(x, y, cw + 0.4, ch + 0.4); }
        }
        const g = reduced ? 0 : Math.max(0, 1 - (now - age[i]) / GLOW_MS);
        if (g > 0 && !grid) { ctx.fillStyle = `rgba(255,236,190,${0.16 * g * g * amp[i]})`; ctx.fillRect(x, y, cw + 0.4, ch + 0.4); }
        const code = cp[i];
        if (code > 32) {
          ctx.font = `${at[i] & BOLD ? 600 : 400} ${fpx}px ${MONO}`;
          ctx.fillStyle = showRoles ? "rgba(233,230,221,.6)" : f0;
          ctx.fillText(String.fromCodePoint(code), x + cw / 2, y + ch / 2 + 0.5);
          if (at[i] & UNDERLINE) ctx.fillRect(x + 1, y + ch - 2, cw - 2, 1);
        }
      }
    }
    if (grid) {  // the VT cell lattice itself: every tenth column / fifth row a little brighter, written cells outlined
      ctx.lineWidth = 1;
      for (let c = 0; c <= W; c++) {
        ctx.strokeStyle = c % 10 === 0 ? "rgba(233,230,221,.13)" : "rgba(233,230,221,.05)";
        const x = Math.round(pad + c * cw) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, pad + H * ch); ctx.stroke();
      }
      for (let row = 0; row <= H; row++) {
        ctx.strokeStyle = row % 5 === 0 ? "rgba(233,230,221,.13)" : "rgba(233,230,221,.05)";
        const y = Math.round(pad + row * ch) + 0.5;
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad + W * cw, y); ctx.stroke();
      }
      if (!reduced) for (let i = 0; i < H * W; i++) {
        const g = 1 - (now - age[i]) / GLOW_MS;
        if (g <= 0) continue;
        const x = pad + (i % W) * cw, y = pad + Math.floor(i / W) * ch;
        ctx.strokeStyle = `rgba(236,208,138,${0.9 * g * amp[i]})`;
        ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
      }
    }
    if (cursor[0] >= 0 && Math.floor(now / 530) % 2 === 0) {
      ctx.fillStyle = "rgba(233,230,221,.8)";
      ctx.fillRect(pad + cursor[1] * cw, pad + cursor[0] * ch + ch - 3, cw, 2);
    }
  }

  // ── pipeline row ─────────────────────────────────────────────────────────
  function drawMinis() {
    if (!H) return;
    const now = performance.now();
    for (const [id, kind] of [["mini0", "cells"], ["mini1", "roles"]]) {
      const cv = $(id), [ctx, r] = fitCanvas(cv), m = miniGeom(cv);
      ctx.clearRect(0, 0, r.width, r.height);
      const gap = m.cw >= 3.2 ? 0.75 : 0.25;
      for (let row = 0; row < H; row++) for (let c = 0; c < W; c++) {
        const i = row * W + c, x = m.x + c * m.cw, y = m.y + row * m.ch;
        let fill = null;
        if (kind === "cells") {
          const g = reduced ? 0 : Math.max(0, 1 - (now - age[i]) / GLOW_MS);
          if (g > 0) fill = `rgba(12,12,11,${0.22 + 0.78 * g * amp[i]})`;
          else if (cp[i] > 32) fill = "rgba(12,12,11,.22)";
          else if (bg[i] < 16 || at[i] & REVERSE) fill = "rgba(12,12,11,.10)";
        } else if (roles[i]) {
          fill = roles[i] === RAW ? ((row + c) % 2 ? HAIR : null) : ROLE_TONE[roles[i]];
        }
        if (fill) { ctx.fillStyle = fill; ctx.fillRect(x + 0.25, y + 0.25, m.cw - gap, m.ch - gap); }
      }
    }
  }

  // changed cells lift out of the console and travel across the pipeline row
  function spawn(f) {
    if (reduced || !f.cells.length) return;
    const n = Math.min(f.cells.length, f.cells.length > 0.3 * H * W ? 18 : 36), now = performance.now();
    for (let q = 0; q < n; q++) {
      const [r, c] = f.cells[Math.floor((q / n) * f.cells.length)];
      particles.push({ r, c, born: now + Math.random() * 160, role: roles[r * W + c], dur: 1300 / Math.sqrt(speed) });
    }
    if (particles.length > 360) particles.splice(0, particles.length - 360);
  }
  const ease = (x) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);
  function drawFlow() {
    const [ctx, app] = fitCanvas($("flow"));
    ctx.clearRect(0, 0, app.width, app.height);
    if (!particles.length || !H) return;
    const t = performance.now();
    const con = consoleGeom(), g0 = miniGeom($("mini0")), g1 = miniGeom($("mini1"));
    const a2 = $("msgs").getBoundingClientRect();
    const inMini = (g, r, c) => [g.r.left - app.left + g.x + (c + 0.5) * g.cw, g.r.top - app.top + g.y + (r + 0.5) * g.ch];
    particles = particles.filter((p) => t - p.born < p.dur + 30);
    for (const p of particles) {
      const x = (t - p.born) / p.dur;
      if (x < 0) continue;
      const stops = [
        [con.r.left - app.left + con.pad + (p.c + 0.5) * con.cw, con.r.top - app.top + con.pad + (p.r + 0.5) * con.ch],
        inMini(g0, p.r, p.c), inMini(g1, p.r, p.c),
        [a2.left - app.left + 10 + (p.c / W) * 40, a2.top - app.top + 10],
      ];
      // the first hop (console → cell grid) takes half the time; the two pipeline hops share the rest
      const ends = [0.5, 0.75, 1];
      const seg = x < ends[0] ? 0 : x < ends[1] ? 1 : 2;
      const a = seg ? ends[seg - 1] : 0, u = ease((x - a) / (ends[seg] - a));
      const [x0, y0] = stops[seg], [x1, y1] = stops[seg + 1];
      const px = x0 + (x1 - x0) * u, py = y0 + (y1 - y0) * u - Math.sin(u * Math.PI) * (seg ? 5 : 0);
      ctx.globalAlpha = Math.min(1, (1 - x) * 4, x * 8);
      ctx.fillStyle = seg === 0 ? "#ecd08a" : p.role && p.role !== RAW ? ROLE_TONE[p.role] : INK;
      const sz = seg === 0 ? 3 : 2.5;
      ctx.fillRect(px - sz / 2, py - sz / 2, sz, sz);
    }
    ctx.globalAlpha = 1;
  }

  // ── A2UI client (a generic renderer: what a non-terminal client draws) ──
  class Surface {
    constructor() { this.components = {}; this.data = {}; }
    apply(m) {
      if (m.createSurface) { this.components = {}; this.data = {}; }
      else if (m.updateComponents) for (const c of m.updateComponents.components) this.components[c.id] = c;
      else if (m.updateDataModel) { const b = m.updateDataModel; this.set(b.path || "/", b.value, !("value" in b)); }
    }
    ptr(p) { return !p || p === "/" ? [] : p.split("/").slice(1).map((s) => s.replace(/~1/g, "/").replace(/~0/g, "~")); }
    set(path, value, del) {
      const ks = this.ptr(path);
      if (!ks.length) { this.data = del ? {} : structuredClone(value); return; }
      let n = this.data;
      for (const key of ks.slice(0, -1)) n = Array.isArray(n) ? n[+key] : (n[key] ??= {});
      const lk = ks.at(-1);
      if (Array.isArray(n)) { if (del) n.splice(+lk, 1); else n[+lk] = structuredClone(value); }
      else if (del) delete n[lk]; else n[lk] = structuredClone(value);
    }
    get(path, scope) {
      let n = path.startsWith("/") || scope === undefined ? this.data : scope;
      for (const key of this.ptr(path.startsWith("/") ? path : "/" + path)) n = n == null ? undefined : n[Array.isArray(n) ? +key : key];
      return n;
    }
    res(v, scope) { return v && typeof v === "object" && "path" in v && Object.keys(v).length === 1 ? this.get(v.path, scope) : v; }
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function kids(c, scope) {
    const x = c.children;
    if (Array.isArray(x)) return x.map((id) => [id, scope]);
    if (x && typeof x === "object") return (surface.get(x.path, scope) || []).map((it) => [x.componentId, it]);
    if (c.child) return [[c.child, scope]];
    return [];
  }
  function node(id, scope) {
    const c = surface.components[id];
    if (!c) return "";
    const inner = kids(c, scope).map(([i, s]) => node(i, s)).join("");
    const tone = (c.tone ? ` data-tone="${c.tone}"` : "") + ` data-c="${c.component}" data-id="${esc(id)}"`;
    switch (c.component) {
      case "Text": {
        const sel = surface.res(c.highlight, scope);
        const t = String(surface.res(c.text, scope) ?? "");
        return `<div class="Text ${c.variant || "body"}${sel ? " sel" : ""}"${tone} style="--w:${Math.min(t.trim().length, 80)}ch">${esc(t)}</div>`;
      }
      case "TextField": return `<span class="TextField"${tone}>${esc(surface.res(c.value, scope))}▏</span>`;
      case "Progress": {
        const v = surface.res(c.value, scope);
        return `<div class="Progress"${tone}><span class="track"><span class="fill" style="width:${(v ?? 0) * 100}%"></span></span>${v == null ? "" : Math.round(v * 100) + "%"} ${esc(surface.res(c.label, scope))}</div>`;
      }
      case "Button": { const key = surface.res(c.shortcut, scope); return `<span class="Button"${tone}>${key ? `<kbd>${esc(key)}</kbd>` : ""}${inner}</span>`; }
      case "Divider": return `<div class="Divider"${tone}></div>`;
      case "Card": return `<div class="Card"${tone}>${inner}</div>`;
      case "Terminal": {
        const rows = surface.res(c.rows, scope) || [];
        const body = rows.map((r) => " ".repeat(Math.min(r.col, 40)) + r.runs.map((x) => x.t).join("")).join("\n");
        return `<div${tone}><div class="tag">not yet matched · sent as cells</div><div class="Terminal">${esc(body)}</div></div>`;
      }
      default: return `<div class="${c.component}"${tone}>${inner}</div>`;
    }
  }
  function renderA2UI() {
    const el = $("a2ui");
    el.innerHTML = surface.components.root ? placed("root") : "";
    // flash the leaf components whose text changed: what a client repaints for this frame
    const seen = new Map(), text = new Map();
    for (const n of el.querySelectorAll("[data-c]")) {
      const id = n.dataset.id, q = (seen.get(id) || 0) + 1;
      seen.set(id, q);
      const key = `${id}#${q}`, t = n.textContent;
      text.set(key, t);
      if (prevText && prevText.get(key) !== t && !n.querySelector("[data-c]")) n.classList.add("fresh");
    }
    prevText = text;
    requestAnimationFrame(() => requestAnimationFrame(() => el.querySelectorAll(".fresh").forEach((n) => n.classList.remove("fresh"))));
    fitClient();
    renderCards();
  }
  // The client view uses the style layer: every component that carries a cell rectangle (/s/<id>/area)
  // is placed on a grid with the terminal's rows and columns, outlined and named, and drawn natively
  // inside. So the two halves of the stage line up element for element.
  function placed(id, scope) {
    const c = surface.components[id];
    if (!c) return "";
    const st = surface.get(`/s/${id}`);
    if (st && Array.isArray(st.area)) {
      const [r0, c0, r1, c1] = st.area, w = c1 - c0;
      const name = c.component === "Text" && c.variant ? c.variant : c.component;
      return `<div class="box" data-k="${c.component}" style="grid-area:${r0 + 1}/${c0 + 1}/${r1 + 1}/${c1 + 1}">` +
        `${w >= 6 ? `<span class="nm">${esc(name)}</span>` : ""}${node(id, scope)}</div>`;
    }
    return kids(c, scope).map(([i, s]) => placed(i, s)).join("");
  }
  function fitClient() {
    const wrap = $("a2ui-wrap"), el = $("a2ui");
    if (!H || !wrap.clientWidth) return;
    const pad = 10, cw = (wrap.clientWidth - 2 * pad) / W, ch = (wrap.clientHeight - 2 * pad) / H;
    el.style.gridTemplateColumns = `repeat(${W}, ${cw}px)`;
    el.style.gridTemplateRows = `repeat(${H}, ${ch}px)`;
    el.style.setProperty("--ch", `${ch}px`);
    el.style.setProperty("--fs", `${Math.max(5, Math.min(ch * 0.82, cw / 0.62, 14))}px`);
  }
  addEventListener("resize", () => fitClient());
  $("wire").onclick = () => {
    const on = $("a2ui").classList.toggle("wire");
    $("wire").setAttribute("aria-pressed", String(on));
  };

  // ── A2UI stream: the messages this frame put on the wire, newest first ──
  function summary(m) {
    if (m.createSurface) return `<b>createSurface</b> <i>${esc(m.createSurface.surfaceId || "")}</i>`;
    if (m.updateComponents) {
      const cs = m.updateComponents.components;
      return `<b>updateComponents</b> ${cs.length} <i>${esc(cs.slice(0, 4).map((c) => c.component).join(" "))}</i>`;
    }
    if (m.updateDataModel) {
      const b = m.updateDataModel, v = "value" in b ? JSON.stringify(b.value) : "∅";
      return `<b>updateDataModel</b> ${esc(b.path || "/")} <i>${esc(v.length > 60 ? v.slice(0, 60) + "…" : v)}</i>`;
    }
    return esc(JSON.stringify(m).slice(0, 80));
  }
  function logMsgs(msgs) {
    const box = $("msgs");
    const pick = msgs.length > 10 ? [...msgs.slice(0, 4), null, ...msgs.slice(-5)] : msgs;
    for (const m of pick.reverse()) {  // newest frame on top, its messages in wire order
      const d = document.createElement("div");
      d.className = "msg new";
      d.innerHTML = m ? `${summary(m)} <i>${kb(JSON.stringify(m).length)}</i>` : `<i>… ${msgs.length - 9} more</i>`;
      box.prepend(d);
      requestAnimationFrame(() => requestAnimationFrame(() => d.classList.remove("new")));
    }
    while (box.children.length > 24) box.lastChild.remove();
  }

  // ── A2UI elements row: each top-level element of the surface, rendered or as its source ──
  function subtree(id, out = []) {
    const c = surface.components[id];
    if (!c || out.includes(c)) return out;
    out.push(c);
    const x = c.children;
    if (Array.isArray(x)) x.forEach((i) => subtree(i, out));
    else if (x && typeof x === "object") subtree(x.componentId, out);
    if (c.child) subtree(c.child, out);
    return out;
  }
  function boundPaths(v, out) {
    if (v && typeof v === "object") {
      if (typeof v.path === "string" && v.path.startsWith("/") && !v.path.startsWith("/s/")) out.add(v.path);
      for (const x of Object.values(v)) boundPaths(x, out);
    }
    return out;
  }
  const clip = (v) => Array.isArray(v) && v.length > 4 ? [...v.slice(0, 4), `… ${v.length - 4} more`] : v;
  function source(id) {
    const comps = subtree(id).map(({ style, ...c }) => c);
    const data = {};
    for (const p of boundPaths(comps, new Set())) data[p] = clip(surface.get(p));
    const json = JSON.stringify(Object.keys(data).length ? { components: comps, data } : { components: comps }, null, 1)
      .replace(/\n\s*/g, (m) => "\n" + " ".repeat(Math.max(0, m.length - 2)));
    return esc(json.length > 2400 ? json.slice(0, 2400) + "\n…" : json)
      .replace(/(&quot;[^&]*?&quot;)(:)/g, '<span class="k">$1</span>$2')
      .replace(/(: )(&quot;.*?&quot;)/g, '$1<span class="s">$2</span>');
  }
  function label(c, scope) {
    const n = c.component, items = c.children && !Array.isArray(c.children) ? (surface.get(c.children.path, scope) || []).length : 0;
    const txt = surface.res(c.text, scope);
    return [n === "Text" && c.variant ? `${n} · ${c.variant}` : n, items ? `${items} items` : typeof txt === "string" ? txt.trim().slice(0, 24) : c.id];
  }
  function renderCards() {
    const box = $("cards"), root = surface.components.root, now = performance.now();
    const top = root ? kids(root) : [];
    const seen = new Set();
    let firstFresh = null, nFresh = 0;
    top.forEach(([id, scope], i) => {
      const c = surface.components[id];
      if (!c) return;
      const key = scope === undefined ? id : `${id}#${i}`;
      seen.add(key);
      let st = cardState.get(key);
      if (!st) {
        const el = document.createElement("div");
        el.className = "card";
        el.innerHTML = `<div class="ch"><b></b><i></i></div><div class="cb"></div>`;
        st = { el, ui: "", src: "", shown: "" };
        cardState.set(key, st);
      }
      const ui = node(id, scope), src = view === "src" ? source(id) : st.src;
      const changed = st.ui && ui !== st.ui;
      st.ui = ui; st.src = src;
      const [name, sub] = label(c, scope);
      st.el.querySelector("b").textContent = name;
      st.el.querySelector("i").textContent = sub;
      const html = view === "src" ? `<pre>${src}</pre>` : `<div class="ui">${ui}</div>`;
      if (html !== st.shown) { st.el.querySelector(".cb").innerHTML = html; st.shown = html; }
      if (changed) {
        nFresh++;
        firstFresh ??= st.el;
        st.el.classList.add("fresh");
        clearTimeout(st.t);
        st.t = setTimeout(() => st.el.classList.remove("fresh"), 220);
      }
      if (box.children[i] !== st.el) box.insertBefore(st.el, box.children[i] || null);
    });
    for (const [key, st] of cardState) if (!seen.has(key)) { st.el.remove(); cardState.delete(key); }
    $("m3").textContent = `${top.length} elements${nFresh ? ` · ${nFresh} updated` : ""}`;
    // follow the change, unless the reader has been scrolling the row themselves
    if (firstFresh && now - userScroll > 4000 && playing) {
      const l = firstFresh.offsetLeft - box.offsetLeft, r = l + firstFresh.offsetWidth;
      if (l < box.scrollLeft || r > box.scrollLeft + box.clientWidth) box.scrollTo({ left: Math.max(0, l - 12), behavior: reduced ? "auto" : "smooth" });
    }
  }
  $("cards").addEventListener("wheel", (e) => {
    userScroll = performance.now();
    if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { e.preventDefault(); $("cards").scrollLeft += e.deltaY; }
  }, { passive: false });
  $("cards").addEventListener("pointerdown", () => { userScroll = performance.now(); });
  for (const b of $("view").children) b.onclick = () => {
    view = b.dataset.v;
    [...$("view").children].forEach((x) => x.classList.toggle("on", x === b));
    renderCards();
  };

  // ── timeline ──────────────────────────────────────────────────────────────
  function drawTicks() {
    if (!F.length) return;
    const [ctx, r] = fitCanvas($("ticks"));
    ctx.clearRect(0, 0, r.width, r.height);
    F.forEach((f, j) => {
      const x0 = (disp[j] / total) * r.width, x1 = ((j + 1 < F.length ? disp[j + 1] : total) / total) * r.width;
      const y = r.height / 2, h = f.state === "locked" ? 4 : 2;
      ctx.fillStyle = f.state === "locked" ? INK : f.state === "matching" ? FAINT : HAIR;
      ctx.fillRect(x0, y - h / 2, Math.max(x1 - x0 - 0.5, 0.5), h);
      if (f.model) { ctx.fillStyle = INK; ctx.fillRect(x0, y - 7, 1, 4); }
    });
  }
  $("timeline").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect(), t = ((e.clientX - r.left) / r.width) * total;
    let j = 0;
    while (j + 1 < F.length && disp[j + 1] <= t) j++;
    seek(j);
    clock = t;
  });

  // ── metrics ───────────────────────────────────────────────────────────────
  const kb = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(2)} MB`);
  function metrics() {
    const f = F[k];
    $("m0").textContent = `${f.cells.length} cells written · ${kb(f.vt)} VT`;
    $("m1").textContent = f.model ? `model · ${f.state}` : `template · ${f.state}`;
    $("m2").textContent = `${kb(f.a2ui)} · ${f.structural ? "layout + data" : "data only"}`;
    $("bytes").innerHTML = `VT <b>${kb(cum.vt)}</b> · A2UI <b>${kb(cum.inc)}</b> <span title="the same UI re-sent in full every keyframe">(full ${kb(cum.full)})</span>`;
    $("clock").textContent = `${Math.floor(f.t / 60)}:${String(Math.floor(f.t % 60)).padStart(2, "0")}`;
  }

  // ── controls ──────────────────────────────────────────────────────────────
  $("play").onclick = () => setPlaying(!playing);
  for (const b of $("speed").children) b.onclick = () => {
    speed = +b.dataset.v;
    [...$("speed").children].forEach((x) => x.classList.toggle("on", x === b));
  };
  for (const b of $("mode").children) b.onclick = () => {
    mode = b.dataset.v;
    [...$("mode").children].forEach((x) => x.classList.toggle("on", x === b));
    $("legend").hidden = mode !== "roles";
  };
  $("legend").innerHTML = ROLE_NAMES.slice(1).map((n, i) => {
    const bgc = i + 1 === RAW ? "repeating-linear-gradient(135deg,#cbc7be 0 1px,transparent 1px 3px)" : ROLE_TONE[i + 1];
    return `<span><i style="background:${bgc}"></i>${n}</span>`;
  }).join("");
  addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); } });

  // ── embed protocol ────────────────────────────────────────────────────────
  function post(msg) { if (embedded) parent.postMessage(msg, "*"); }
  addEventListener("message", (e) => {
    if (e.source !== parent || !e.data || typeof e.data.type !== "string") return;
    const type = e.data.type.replace(/^(netscraper|phosphene):/, "");
    if (type === "play" && !reduced) setPlaying(true);
    else if (type === "pause") setPlaying(false);
    else if (type === "seek" && typeof e.data.t === "number") {
      let j = 0;
      while (j + 1 < F.length && disp[j + 1] <= e.data.t) j++;
      seek(j);
    }
  });

  boot().catch((err) => { document.body.textContent = `phosphene: ${err}`; });
})();

"""gui_assets.py - the single-page front end for Phase 8's GUI mode.

Kept as a Python string constant rather than a separate .html file so the
package has no data files to install - `pip install llmadapt` ships one wheel
of .py modules and the GUI works, with no package_data configuration to get
wrong. gui.py serves this string directly.

Vanilla JS and SVG only. No framework, no CDN, no build step - the same
zero-dependency rule the rest of llmadapt follows, applied to the front end.
The page fetches everything it needs (preset names, palette colours, the
starting spec) from the local server at load, so this string contains no
generated data other than the session token substituted for %%TOKEN%%.
"""

# %%TOKEN%% is replaced by gui.py at serve time. Every fetch sends it back, so
# another process on the same machine (or a page in another tab) can't drive
# this server just by knowing the port.
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llmadapt - company builder</title>
<style>
  :root {
    --surface: #fcfcfb; --panel: #f4f3f0; --ink: #0b0b0b; --ink-2: #52514e;
    --line: #d8d6d0; --accent: #2a78d6; --danger: #b3261e; --ok: #008300;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
               color: var(--ink); background: var(--surface); font-size: 14px; }
  #app { display: grid; grid-template-columns: 290px 1fr; height: 100%; }
  #side { background: var(--panel); border-right: 1px solid var(--line); overflow-y: auto; padding: 14px; }
  #main { position: relative; overflow: hidden; }
  h1 { font-size: 15px; margin: 0 0 12px; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--ink-2);
       margin: 18px 0 8px; border-top: 1px solid var(--line); padding-top: 12px; }
  h2:first-of-type { border-top: 0; padding-top: 0; margin-top: 10px; }
  label { display: block; font-size: 12px; color: var(--ink-2); margin: 8px 0 3px; }
  input, select, textarea, button { font: inherit; width: 100%; padding: 6px 8px;
       border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink); }
  button { cursor: pointer; background: #fff; }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 600; }
  button.danger { color: var(--danger); }
  .row { display: flex; gap: 6px; margin-top: 8px; }
  .row > * { flex: 1; }
  .hint { font-size: 11px; color: var(--ink-2); line-height: 1.45; margin: 6px 0 0; }
  .checks { max-height: 132px; overflow-y: auto; border: 1px solid var(--line);
            border-radius: 6px; background: #fff; padding: 6px; }
  .checks label { display: flex; align-items: center; gap: 6px; margin: 2px 0; color: var(--ink); }
  .checks input { width: auto; }
  #toolbar { position: absolute; top: 10px; left: 10px; right: 10px; display: flex; gap: 8px;
             align-items: center; z-index: 5; }
  #toolbar button { width: auto; }
  #status { margin-left: auto; font-size: 12px; color: var(--ink-2); max-width: 45%; text-align: right; }
  #canvas { width: 100%; height: 100%; display: block; background: var(--surface); }
  .node rect { fill: #fff; stroke-width: 2; cursor: grab; }
  .node.sel rect { stroke-dasharray: 5 3; }
  .node text { pointer-events: none; }
  .node .nm { font-size: 13px; font-weight: 600; }
  .node .rk { font-size: 11px; fill: var(--ink-2); }
  .edge { fill: none; stroke: #b9b7b0; stroke-width: 1.6; }
  #menu { position: absolute; background: #fff; border: 1px solid var(--line); border-radius: 8px;
          box-shadow: 0 6px 20px rgba(0,0,0,.14); padding: 5px; display: none; z-index: 20; min-width: 214px; }
  #menu button { text-align: left; border: 0; border-radius: 5px; padding: 7px 9px; background: none; }
  #menu button:hover { background: var(--panel); }
  #menu hr { border: 0; border-top: 1px solid var(--line); margin: 4px 2px; }
  #problems { position: absolute; bottom: 10px; left: 10px; right: 10px; max-height: 30%;
              overflow-y: auto; background: #fff; border: 1px solid var(--line); border-radius: 8px;
              padding: 10px 12px; display: none; z-index: 6; }
  #problems ul { margin: 6px 0 0; padding-left: 18px; }
  #problems li { margin: 3px 0; }
  .band { fill: rgba(42,120,214,.10); stroke: var(--accent); stroke-dasharray: 4 3; }
  .muted { color: var(--ink-2); }
</style>
</head>
<body>
<div id="app">
  <aside id="side">
    <h1>Company builder</h1>

    <label>Company name</label>
    <input id="coName" value="New Company">

    <div class="row">
      <div>
        <label>Palette</label>
        <select id="palette"></select>
      </div>
      <div>
        <label>Token budget</label>
        <input id="budget" type="number" min="0" step="1000" value="0">
      </div>
    </div>
    <p class="hint">Budget 0 means no ceiling. A ceiling is the only thing that
      stops a runaway company spending; setting one is strongly advised.</p>

    <h2>Start from a template</h2>
    <select id="template"></select>
    <div class="row">
      <select id="size"></select>
      <button id="applyTemplate">Load</button>
    </div>
    <p class="hint" id="templateHint"></p>

    <h2>Selected employee</h2>
    <div id="editor"><p class="hint">Select a node to edit it. Shift-click for several.</p></div>

    <h2>Model routing</h2>
    <div class="row">
      <div>
        <label>Default mode</label>
        <select id="policyMode"></select>
      </div>
    </div>
    <p class="hint">What every employee inherits unless their own panel says otherwise.
      "local" keeps the whole company off paid APIs.</p>

    <h2>Review</h2>
    <div class="row">
      <div>
        <label>Mode</label>
        <select id="reviewMode"></select>
      </div>
      <div>
        <label>Max rounds</label>
        <input id="reviewRounds" type="number" min="0" max="5" value="1">
      </div>
    </div>

    <h2>Finish</h2>
    <button class="primary" id="build">Build this company</button>
    <div class="row">
      <button id="check">Check</button>
      <button id="download">Save JSON</button>
    </div>
    <p class="hint">"Build" hands the design back to the Python process that
      opened this page and closes the server. Nothing is run and nothing is spent -
      you start tasks yourself afterwards.</p>
  </aside>

  <main id="main">
    <div id="toolbar">
      <button id="addBtn">+ Employee</button>
      <button id="connectBtn">Connect</button>
      <button id="autoBtn">Tidy layout</button>
      <span id="status" class="muted"></span>
    </div>
    <svg id="canvas"></svg>
    <div id="menu"></div>
    <div id="problems"></div>
  </main>
</div>

<script>
"use strict";
const TOKEN = "%%TOKEN%%";
const SVG_NS = "http://www.w3.org/2000/svg";
const NODE_W = 168, NODE_H = 52;

const state = {
  options: { ranks: [], skills: [], personalities: [], palettes: [], org_templates: [],
             sizes: [], review_modes: [], policy_modes: [], effort_levels: [], templates_detail: {} },
  palettes: {},
  spec: { name: "New Company", template: null, size: "small", employees: [], palette: "dataviz",
          total_token_budget: null, review_mode: "critique", max_review_rounds: 1,
          default_policy_mode: null, layout: {} },
  selected: new Set(),
  connectFrom: null,
  connecting: false,
  drag: null,
  band: null,
};

const $ = (id) => document.getElementById(id);
const canvas = $("canvas"), menu = $("menu"), problems = $("problems");

function api(path, body) {
  const opts = { method: body === undefined ? "GET" : "POST",
                 headers: { "X-Token": TOKEN, "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  return fetch(path, opts).then((r) => r.json());
}

function status(text, kind) {
  const el = $("status");
  el.textContent = text || "";
  el.style.color = kind === "bad" ? "var(--danger)" : kind === "ok" ? "var(--ok)" : "var(--ink-2)";
}

function uniqueName(base) {
  const taken = new Set(state.spec.employees.map((e) => e.name));
  if (!taken.has(base)) return base;
  let n = 2;
  while (taken.has(base + " " + n)) n++;
  return base + " " + n;
}

function byName(name) { return state.spec.employees.find((e) => e.name === name) || null; }

function layoutOf(name) {
  if (!state.spec.layout[name]) state.spec.layout[name] = [80 + Math.random() * 320, 80 + Math.random() * 260];
  return state.spec.layout[name];
}

function rankColor(rank) {
  const pal = state.palettes[$("palette").value] || {};
  const ranks = pal.ranks_light || [];
  const idx = state.options.ranks.indexOf(rank);
  if (!ranks.length) return "#2a78d6";
  return ranks[(idx < 0 ? ranks.length - 1 : idx) % ranks.length];
}

/* ---------------- rendering ---------------- */

function render() {
  while (canvas.firstChild) canvas.removeChild(canvas.firstChild);
  const edges = document.createElementNS(SVG_NS, "g");
  const nodes = document.createElementNS(SVG_NS, "g");
  canvas.appendChild(edges);
  canvas.appendChild(nodes);

  state.spec.employees.forEach((emp) => {
    if (!emp.reports_to || !byName(emp.reports_to)) return;
    const a = layoutOf(emp.reports_to), b = layoutOf(emp.name);
    const x1 = a[0] + NODE_W / 2, y1 = a[1] + NODE_H;
    const x2 = b[0] + NODE_W / 2, y2 = b[1];
    const mid = (y1 + y2) / 2;
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("class", "edge");
    path.setAttribute("d", `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`);
    edges.appendChild(path);
  });

  state.spec.employees.forEach((emp) => {
    const [x, y] = layoutOf(emp.name);
    const g = document.createElementNS(SVG_NS, "g");
    g.setAttribute("class", "node" + (state.selected.has(emp.name) ? " sel" : ""));
    g.setAttribute("transform", `translate(${x},${y})`);
    g.dataset.name = emp.name;

    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("width", NODE_W);
    rect.setAttribute("height", NODE_H);
    rect.setAttribute("rx", 8);
    rect.setAttribute("stroke", rankColor(emp.rank));
    g.appendChild(rect);

    const nm = document.createElementNS(SVG_NS, "text");
    nm.setAttribute("class", "nm");
    nm.setAttribute("x", NODE_W / 2); nm.setAttribute("y", 21);
    nm.setAttribute("text-anchor", "middle");
    nm.textContent = emp.name;
    g.appendChild(nm);

    const rk = document.createElementNS(SVG_NS, "text");
    rk.setAttribute("class", "rk");
    rk.setAttribute("x", NODE_W / 2); rk.setAttribute("y", 38);
    rk.setAttribute("text-anchor", "middle");
    const bits = [emp.rank];
    if (emp.skills && emp.skills.length) bits.push(emp.skills.join("/"));
    rk.textContent = bits.join(" - ").slice(0, 30);
    g.appendChild(rk);

    nodes.appendChild(g);
  });

  if (state.band) {
    const r = document.createElementNS(SVG_NS, "rect");
    r.setAttribute("class", "band");
    r.setAttribute("x", Math.min(state.band.x0, state.band.x1));
    r.setAttribute("y", Math.min(state.band.y0, state.band.y1));
    r.setAttribute("width", Math.abs(state.band.x1 - state.band.x0));
    r.setAttribute("height", Math.abs(state.band.y1 - state.band.y0));
    canvas.appendChild(r);
  }
  renderEditor();
}

/* ---------------- left panel editor ---------------- */

function renderEditor() {
  const box = $("editor");
  const names = [...state.selected];
  if (!names.length) {
    box.innerHTML = '<p class="hint">Select a node to edit it. Shift-click for several.</p>';
    return;
  }
  if (names.length > 1) {
    box.innerHTML = `<p class="hint">${names.length} selected. Right-click the canvas for bulk
      actions, or set a rank for all of them:</p>`;
    const sel = document.createElement("select");
    sel.innerHTML = '<option value="">Set rank for all...</option>' +
      state.options.ranks.map((r) => `<option>${r}</option>`).join("");
    sel.onchange = () => {
      if (!sel.value) return;
      names.forEach((n) => { byName(n).rank = sel.value; });
      render();
    };
    box.appendChild(sel);
    return;
  }

  const emp = byName(names[0]);
  if (!emp) { state.selected.clear(); return renderEditor(); }
  box.innerHTML = "";

  const field = (labelText, node) => {
    const l = document.createElement("label");
    l.textContent = labelText;
    box.appendChild(l);
    box.appendChild(node);
    return node;
  };

  const nameInput = document.createElement("input");
  nameInput.value = emp.name;
  nameInput.onchange = () => {
    const wanted = nameInput.value.trim();
    if (!wanted) { nameInput.value = emp.name; return; }
    if (wanted !== emp.name && byName(wanted)) {
      status("A employee called " + wanted + " already exists", "bad");
      nameInput.value = emp.name;
      return;
    }
    const old = emp.name;
    state.spec.employees.forEach((e) => { if (e.reports_to === old) e.reports_to = wanted; });
    state.spec.layout[wanted] = state.spec.layout[old];
    delete state.spec.layout[old];
    emp.name = wanted;
    state.selected = new Set([wanted]);
    render();
  };
  field("Name", nameInput);

  const rankSel = document.createElement("select");
  rankSel.innerHTML = state.options.ranks.map(
    (r) => `<option${r === emp.rank ? " selected" : ""}>${r}</option>`).join("");
  rankSel.onchange = () => { emp.rank = rankSel.value; render(); };
  field("Rank", rankSel);

  const mgrSel = document.createElement("select");
  mgrSel.innerHTML = '<option value="">(nobody - top of the org)</option>' +
    state.spec.employees.filter((e) => e.name !== emp.name)
      .map((e) => `<option${e.name === emp.reports_to ? " selected" : ""}>${e.name}</option>`).join("");
  mgrSel.onchange = () => { emp.reports_to = mgrSel.value || null; render(); };
  field("Reports to", mgrSel);

  const skillBox = document.createElement("div");
  skillBox.className = "checks";
  const descs = (state.options.descriptions || {});
  state.options.skills.forEach((skill) => {
    const l = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = (emp.skills || []).includes(skill);
    cb.onchange = () => {
      emp.skills = emp.skills || [];
      if (cb.checked) emp.skills.push(skill);
      else emp.skills = emp.skills.filter((s) => s !== skill);
      render();
    };
    // What the skill actually does, on hover - a checkbox list of bare names
    // is not something anyone can pick from confidently.
    const about = (descs.skills || {})[skill];
    if (about) l.title = skill + " - " + about;
    l.appendChild(cb);
    l.appendChild(document.createTextNode(skill));
    skillBox.appendChild(l);
  });
  field("Skills", skillBox);

  const persSel = document.createElement("select");
  persSel.innerHTML = '<option value="">(none)</option>' + state.options.personalities.map(
    (p) => `<option${p === emp.personality ? " selected" : ""} title="${descs.personalities ? (descs.personalities[p] || "") : ""}">${p}</option>`).join("");
  // A one-line "what this means" under the dropdown, updated as it changes -
  // an <option> title is not shown by every browser, and the description is
  // most wanted at the moment of choosing.
  const persAbout = document.createElement("div");
  persAbout.className = "hint";
  const showPers = () => {
    persAbout.textContent = (descs.personalities || {})[emp.personality] || "";
  };
  showPers();
  persSel.onchange = () => { emp.personality = persSel.value || null; showPers(); };
  const persWrap = document.createElement("div");
  persWrap.appendChild(persSel);
  persWrap.appendChild(persAbout);
  field("Personality", persWrap);

  const effortSel = document.createElement("select");
  effortSel.innerHTML = [""].concat(state.options.effort_levels).map(
    (e) => `<option value="${e}"${e === (emp.effort || "") ? " selected" : ""}>${e || "(rank default)"}</option>`
  ).join("");
  effortSel.onchange = () => { emp.effort = effortSel.value || null; };
  field("Effort hint", effortSel);

  const modeSel = document.createElement("select");
  modeSel.innerHTML = [""].concat(state.options.policy_modes).map(
    (m) => `<option value="${m}"${m === (emp.mode || "") ? " selected" : ""}>${m || "(use model_map)"}</option>`
  ).join("");
  modeSel.onchange = () => { emp.mode = modeSel.value || null; };
  field("Model policy mode", modeSel);

  const provInput = document.createElement("input");
  provInput.value = emp.provider || "";
  provInput.placeholder = "e.g. anthropic, ollama";
  provInput.onchange = () => { emp.provider = provInput.value.trim() || null; };
  field("Provider override", provInput);

  const modelInput = document.createElement("input");
  modelInput.value = emp.model || "";
  modelInput.placeholder = "e.g. claude-sonnet-5-5";
  modelInput.onchange = () => { emp.model = modelInput.value.trim() || null; };
  field("Model override", modelInput);

  const impInput = document.createElement("input");
  impInput.type = "number"; impInput.min = "0"; impInput.max = "1"; impInput.step = "0.05";
  impInput.value = emp.importance === undefined ? 0.5 : emp.importance;
  impInput.onchange = () => { emp.importance = parseFloat(impInput.value) || 0; };
  field("Budget importance (0-1)", impInput);

  const note = document.createElement("p");
  note.className = "hint";
  note.textContent = "API keys are deliberately not part of a saved design - they come from " +
    "the model_map you pass to build_company(), so a design file can be shared safely.";
  box.appendChild(note);

  const del = document.createElement("button");
  del.className = "danger";
  del.textContent = "Delete this employee";
  del.style.marginTop = "10px";
  del.onclick = () => { removeEmployees([emp.name]); };
  box.appendChild(del);
}

/* ---------------- editing ---------------- */

function addEmployee(x, y) {
  const name = uniqueName("Employee");
  state.spec.employees.push({ name, rank: state.options.ranks[3] || "SENIOR", reports_to: null,
                              skills: [], personality: null, effort: null, specialty: null,
                              importance: 0.5, provider: null, model: null, mode: null });
  state.spec.layout[name] = [x === undefined ? 120 : x, y === undefined ? 120 : y];
  state.selected = new Set([name]);
  render();
}

function removeEmployees(names) {
  const gone = new Set(names);
  state.spec.employees = state.spec.employees.filter((e) => !gone.has(e.name));
  state.spec.employees.forEach((e) => { if (gone.has(e.reports_to)) e.reports_to = null; });
  names.forEach((n) => { delete state.spec.layout[n]; state.selected.delete(n); });
  render();
}

function wouldCycle(childName, managerName) {
  let cursor = managerName;
  const seen = new Set();
  while (cursor) {
    if (cursor === childName) return true;
    if (seen.has(cursor)) return false;
    seen.add(cursor);
    const node = byName(cursor);
    cursor = node ? node.reports_to : null;
  }
  return false;
}

function connect(childName, managerName) {
  if (childName === managerName) return status("An employee cannot report to themselves", "bad");
  if (wouldCycle(childName, managerName)) return status("That would create a reporting cycle", "bad");
  byName(childName).reports_to = managerName;
  status(childName + " now reports to " + managerName, "ok");
  render();
}

function tidyLayout() {
  const depth = (emp) => {
    let d = 0, cursor = emp.reports_to, seen = new Set();
    while (cursor && !seen.has(cursor)) { seen.add(cursor); d++; cursor = (byName(cursor) || {}).reports_to; }
    return d;
  };
  const rows = {};
  state.spec.employees.forEach((e) => {
    const d = depth(e);
    (rows[d] = rows[d] || []).push(e);
  });
  Object.keys(rows).forEach((d) => {
    rows[d].forEach((emp, i) => {
      state.spec.layout[emp.name] = [40 + i * (NODE_W + 30), 60 + Number(d) * (NODE_H + 70)];
    });
  });
  render();
}

/* ---------------- canvas interaction ---------------- */

function pointAt(evt) {
  const r = canvas.getBoundingClientRect();
  return [evt.clientX - r.left, evt.clientY - r.top];
}

function nodeAt(evt) {
  const g = evt.target.closest ? evt.target.closest("g.node") : null;
  return g ? g.dataset.name : null;
}

canvas.addEventListener("pointerdown", (evt) => {
  hideMenu();
  if (evt.button === 2) return;
  const name = nodeAt(evt);
  const [px, py] = pointAt(evt);

  if (state.connecting) {
    if (!name) return;
    if (!state.connectFrom) {
      state.connectFrom = name;
      status("Now click this employee's manager (Esc to cancel)");
    } else {
      connect(state.connectFrom, name);
      state.connectFrom = null;
      state.connecting = false;
      $("connectBtn").classList.remove("primary");
    }
    return;
  }

  if (!name) {
    if (!evt.shiftKey) state.selected.clear();
    state.band = { x0: px, y0: py, x1: px, y1: py };
    canvas.setPointerCapture(evt.pointerId);
    render();
    return;
  }

  if (evt.shiftKey) {
    if (state.selected.has(name)) state.selected.delete(name);
    else state.selected.add(name);
  } else if (!state.selected.has(name)) {
    state.selected = new Set([name]);
  }
  const origins = {};
  state.selected.forEach((n) => { origins[n] = layoutOf(n).slice(); });
  state.drag = { x: px, y: py, origins };
  canvas.setPointerCapture(evt.pointerId);
  render();
});

canvas.addEventListener("pointermove", (evt) => {
  const [px, py] = pointAt(evt);
  if (state.drag) {
    const dx = px - state.drag.x, dy = py - state.drag.y;
    Object.keys(state.drag.origins).forEach((n) => {
      const o = state.drag.origins[n];
      state.spec.layout[n] = [o[0] + dx, o[1] + dy];
    });
    render();
  } else if (state.band) {
    state.band.x1 = px; state.band.y1 = py;
    render();
  }
});

canvas.addEventListener("pointerup", (evt) => {
  if (state.band) {
    const x0 = Math.min(state.band.x0, state.band.x1), x1 = Math.max(state.band.x0, state.band.x1);
    const y0 = Math.min(state.band.y0, state.band.y1), y1 = Math.max(state.band.y0, state.band.y1);
    if (x1 - x0 > 4 || y1 - y0 > 4) {
      state.spec.employees.forEach((emp) => {
        const [x, y] = layoutOf(emp.name);
        if (x + NODE_W > x0 && x < x1 && y + NODE_H > y0 && y < y1) state.selected.add(emp.name);
      });
    }
    state.band = null;
  }
  state.drag = null;
  render();
});

canvas.addEventListener("dblclick", (evt) => {
  if (nodeAt(evt)) return;
  const [x, y] = pointAt(evt);
  addEmployee(x - NODE_W / 2, y - NODE_H / 2);
});

canvas.addEventListener("contextmenu", (evt) => {
  evt.preventDefault();
  const name = nodeAt(evt);
  if (name && !state.selected.has(name)) state.selected = new Set([name]);
  render();
  showMenu(evt);
});

document.addEventListener("keydown", (evt) => {
  if (evt.key === "Escape") {
    state.connecting = false; state.connectFrom = null;
    $("connectBtn").classList.remove("primary");
    hideMenu(); status("");
  }
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes((document.activeElement || {}).tagName);
  if ((evt.key === "Delete" || evt.key === "Backspace") && state.selected.size && !typing) {
    evt.preventDefault();
    removeEmployees([...state.selected]);
  }
});

/* ---------------- right-click menu (bulk actions) ---------------- */

function hideMenu() { menu.style.display = "none"; }

function showMenu(evt) {
  const names = [...state.selected];
  menu.innerHTML = "";
  const item = (label, fn, disabled) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.disabled = !!disabled;
    if (disabled) b.style.opacity = ".45";
    b.onclick = () => { hideMenu(); fn(); };
    menu.appendChild(b);
  };
  const rule = () => menu.appendChild(document.createElement("hr"));

  item("Add employee here", () => {
    const [x, y] = pointAt(evt);
    addEmployee(x - NODE_W / 2, y - NODE_H / 2);
  });
  if (names.length) {
    rule();
    item(`Connect all ${names.length} to one manager...`, () => {
      const manager = window.prompt(
        "Name of the manager they should all report to:\n\n" +
        state.spec.employees.map((e) => e.name).join(", "));
      if (!manager) return;
      if (!byName(manager)) return status("No employee called " + manager, "bad");
      let done = 0;
      names.forEach((n) => {
        if (n !== manager && !wouldCycle(n, manager)) { byName(n).reports_to = manager; done++; }
      });
      status(`Connected ${done} employee(s) to ${manager}`, "ok");
      render();
    }, names.length === 0);
    item("Clear their manager", () => {
      names.forEach((n) => { byName(n).reports_to = null; });
      render();
    });
    item("Add a skill to all...", () => {
      const skill = window.prompt("Skill to add to all selected:\n\n" + state.options.skills.join(", "));
      if (!skill) return;
      if (!state.options.skills.includes(skill)) return status("Unknown skill " + skill, "bad");
      names.forEach((n) => {
        const emp = byName(n);
        emp.skills = emp.skills || [];
        if (!emp.skills.includes(skill)) emp.skills.push(skill);
      });
      render();
    });
    item("Duplicate selected", () => {
      const copies = [];
      names.forEach((n) => {
        const src = byName(n);
        const copy = JSON.parse(JSON.stringify(src));
        copy.name = uniqueName(src.name);
        const [x, y] = layoutOf(n);
        state.spec.layout[copy.name] = [x + 26, y + 26];
        state.spec.employees.push(copy);
        copies.push(copy.name);
      });
      state.selected = new Set(copies);
      render();
    });
    rule();
    item(`Delete ${names.length} selected`, () => removeEmployees(names));
  }
  rule();
  item("Select all", () => {
    state.selected = new Set(state.spec.employees.map((e) => e.name));
    render();
  });
  item("Tidy layout", tidyLayout);

  const r = canvas.getBoundingClientRect();
  menu.style.display = "block";
  menu.style.left = Math.min(evt.clientX - r.left, r.width - 224) + "px";
  menu.style.top = Math.min(evt.clientY - r.top, r.height - menu.offsetHeight - 10) + "px";
}

document.addEventListener("pointerdown", (evt) => {
  if (!menu.contains(evt.target)) hideMenu();
}, true);

/* ---------------- spec <-> form ---------------- */

function collect() {
  state.spec.name = $("coName").value.trim() || "New Company";
  state.spec.palette = $("palette").value;
  const budget = parseInt($("budget").value, 10);
  state.spec.total_token_budget = budget > 0 ? budget : null;
  state.spec.review_mode = $("reviewMode").value;
  /* "" is the "(inherit / none)" option - it has to become null, not "", or the
     spec would claim a mode it does not have and validate() would reject it. */
  state.spec.default_policy_mode = $("policyMode").value || null;
  state.spec.max_review_rounds = parseInt($("reviewRounds").value, 10) || 0;
  state.spec.size = $("size").value;
  return state.spec;
}

function showProblems(list, warnings) {
  if (!list.length && !(warnings || []).length) { problems.style.display = "none"; return; }
  problems.style.display = "block";
  const bits = [];
  if (list.length) {
    bits.push("<strong>Problems</strong><ul>" +
      list.map((p) => `<li>${escapeHtml(p)}</li>`).join("") + "</ul>");
  }
  if ((warnings || []).length) {
    bits.push('<strong class="muted">Warnings</strong><ul class="muted">' +
      warnings.map((p) => `<li>${escapeHtml(p)}</li>`).join("") + "</ul>");
  }
  problems.innerHTML = bits.join("");
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* ---------------- buttons ---------------- */

$("addBtn").onclick = () => addEmployee();
$("autoBtn").onclick = tidyLayout;
$("connectBtn").onclick = () => {
  state.connecting = !state.connecting;
  state.connectFrom = null;
  $("connectBtn").classList.toggle("primary", state.connecting);
  status(state.connecting ? "Click the employee who reports, then their manager" : "");
};
$("palette").onchange = render;

$("applyTemplate").onclick = () => {
  const name = $("template").value;
  if (!name) return;
  api("/api/template", { template: name, size: $("size").value }).then((data) => {
    if (data.error) return status(data.error, "bad");
    state.spec.employees = data.employees;
    state.spec.layout = {};
    state.spec.template = null;  // the roles are now concrete employees, not a template reference
    state.selected.clear();
    tidyLayout();
    status(`Loaded ${data.employees.length} employees from ${name}`, "ok");
  });
};

$("check").onclick = () => {
  api("/api/validate", collect()).then((data) => {
    showProblems(data.problems || [], data.warnings || []);
    status(data.problems && data.problems.length
      ? data.problems.length + " problem(s)" : "Looks valid",
      data.problems && data.problems.length ? "bad" : "ok");
  });
};

$("download").onclick = () => {
  const spec = collect();
  api("/api/save", spec).then((data) => {
    if (data.problems && data.problems.length) {
      showProblems(data.problems, data.warnings || []);
      return status("Not saved - fix the problems first", "bad");
    }
    showProblems([], data.warnings || []);
    // /api/save only writes a file when the Python process that opened this
    // page passed save_path= to launch_gui()/set_company_up(mode="gui")/
    // make_company_via_gui() - most callers don't, since it's meant for "the
    // caller wants a specific file on disk", not "the button should do
    // something." Without that fallback, this button used to report
    // "Saved to the launching process" while writing nothing anywhere the
    // user could find it - which reads as broken, because from the user's
    // side, nothing happened. It now always triggers an actual browser
    // download of the validated design, so the button does something
    // real regardless of how Python launched it; the server-side write
    // still happens too, and is reported alongside it, when save_path was set.
    downloadJson(spec, safeFilename(spec.name) + ".json");
    status(
      data.path
        ? "Downloaded, and also written to " + data.path + " by the launching process"
        : "Downloaded to your browser's downloads folder",
      "ok"
    );
  });
};

function safeFilename(name) {
  const cleaned = String(name || "company").trim().replace(/[^a-zA-Z0-9._-]+/g, "-");
  return cleaned || "company";
}

function downloadJson(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

$("build").onclick = () => {
  api("/api/build", collect()).then((data) => {
    if (data.problems && data.problems.length) {
      showProblems(data.problems, data.warnings || []);
      return status("Not built - fix the problems first", "bad");
    }
    showProblems([], data.warnings || []);
    document.body.innerHTML =
      '<div style="padding:44px;font-family:system-ui,sans-serif;max-width:680px">' +
      '<h1 style="font-size:20px">Company handed back to Python.</h1>' +
      '<p>' + escapeHtml(data.headcount) + ' employees. You can close this tab - ' +
      'the builder has returned your design to the process that opened it. ' +
      'Nothing has been run and nothing has been spent.</p>' +
      '<pre style="background:#f4f3f0;padding:14px;border-radius:8px;overflow:auto">' +
      escapeHtml(data.org_chart || "") + '</pre></div>';
  });
};

/* ---------------- boot ---------------- */

function fillSelect(el, values, current) {
  el.innerHTML = values.map((v) => `<option${v === current ? " selected" : ""}>${v}</option>`).join("");
}

Promise.all([api("/api/options"), api("/api/spec")]).then(([options, spec]) => {
  state.options = options;
  state.palettes = options.palette_colors || {};
  if (spec && spec.name) {
    state.spec = Object.assign(state.spec, spec);
    state.spec.layout = spec.layout || {};
  }
  $("coName").value = state.spec.name;
  $("budget").value = state.spec.total_token_budget || 0;
  $("reviewRounds").value = state.spec.max_review_rounds;
  fillSelect($("palette"), options.palettes, state.spec.palette);
  fillSelect($("size"), options.sizes, state.spec.size);
  fillSelect($("reviewMode"), options.review_modes, state.spec.review_mode);
  $("policyMode").innerHTML = '<option value="">(none)</option>' +
    options.policy_modes.map((m) => `<option>${m}</option>`).join("");
  $("policyMode").value = state.spec.default_policy_mode || "";
  $("template").innerHTML = '<option value="">(none)</option>' +
    options.org_templates.map((t) => `<option>${t}</option>`).join("");
  $("template").onchange = () => {
    const detail = options.templates_detail[$("template").value];
    $("templateHint").textContent = detail ? detail.description : "";
  };
  if (!state.spec.employees.length && !Object.keys(state.spec.layout).length) {
    status("Double-click the canvas to add an employee, or load a template.");
  }
  tidyLayout();
});

window.addEventListener("resize", render);
</script>
</body>
</html>
"""

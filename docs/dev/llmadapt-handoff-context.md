# llmadapt — handoff context (post Phase 0-8)

Companion to `company-roadmap.md`. The roadmap is the authoritative record of
*what was built and why*, per phase. This document covers everything around
that: conventions, decisions that shouldn't be re-opened, the file map, and how
to pick the work up next.

**This replaces the previous handoff doc.** Phases 0-8 are done; Phase 9 (the
TODO pass) is done on top of them. Every `# TODO` decision point in the library
is now closed - implemented, or answered in the docstring where it stood.

## 1. Project identity

- Library: **llmadapt** — a personal, zero-third-party-dependency,
  provider-agnostic Python LLM agent library (Anthropic/OpenAI/Gemini/Ollama/
  custom providers, tool-calling loop, sync + async), now with a full
  multi-agent company layer on top.
- GitHub: `Shyam-Parikh-2025/llmadapt`. Local path (Windows machine
  "shyam-pc"): `C:\Users\HP\OneDrive\Coding Projects\Basics Genai\llmadapt`.
- Package root: `src/llmadapt/`. Tests: `tests/` (flat, no subpackage).
- Version 0.3.0.

## 2. Status: Phases 0-8 complete

| Phase | Module(s) | What it does |
|---|---|---|
| 0-1 | `company.py`, `router.py` | Employee/Team/Company, delegation as a tool call, escalation up the reporting chain |
| 2 | `observability.py` | `EventLog`, org-chart renderers (ascii/mermaid/svg) |
| 3 | `budget.py` | Percentage token allocation, hard ceiling, ask-your-manager reallocation |
| 4 | `policy.py` | Local-vs-API routing driven by an effort hint |
| 5 | `presets.py` | One registry for skills, personalities, palettes, org templates |
| 6 | `delegation.py` | Stub-and-fill + plan-then-execute |
| 7 | `compaction.py` | "Broad concepts, drop specifics" log compaction |
| 8 | `builder.py`, `gui.py`, `gui_assets.py` | `set_company_up(mode="text"\|"gui")` |
| 9 | `company/`, `suggest.py`, `archive.py` | TODO pass: package split, cost-weighted budget, run archive, typo suggestions, company-wide policy mode |
| 10 | `env.py`, `presets/`, `compressor.py` | Pinned context, preset package + bigger catalogs, .env loading and layered key resolution |

Full suite: **17 files, 403 checks**, all passing.

## 3. Engineering conventions — still load-bearing

- **Zero third-party dependencies.** Optional deps (numpy, torch) are used
  opportunistically only when already installed, always with a fallback — see
  `benchmark.py`. This now extends to the front end: the Phase 8 GUI is vanilla
  JS + SVG with no framework, no CDN and no build step. Enforced rather than
  merely stated: Phase 10 found an `import dotenv` that had crept into
  `core.py`'s demo block and replaced it with the library's own
  `env.load_env()`. Worth grepping for stray imports occasionally.
- **Test style**: plain `assert` + `print("PASS: ...")`, no pytest, no
  framework. Offline `FakeResponder` standing in for an `Agent`'s HTTP layer
  via `agent._send_request = FakeResponder([...])`. Every new module gets its
  own `tests/test_<module>.py` in that style.
- **Run the whole suite** with `PYTHONPATH=src python3 tests/run_all.py` before
  calling anything done. `run_all.py` skips `test_trial_own.py` (an interactive
  REPL, not a test) and `browser_smoke.py` (needs playwright, deliberately not
  a dependency — run it by hand after touching `gui_assets.py`).
- **`company-roadmap.md` is the single source of truth across sessions.** Each
  phase entry captures *why* something is shaped the way it is, not just what
  landed. Keep that standard.
- **Flag simplifications honestly, in the code and in the roadmap.** Every
  phase has at least one: Phase 3's gate is pre-flight not mid-generation;
  Phase 4's price table is hand-entered and will go stale; Phase 5's approval
  detection is a convention not a protocol; Phase 6 checks generated code for
  syntax only. None of these are hidden.

## 4. Decisions already made — do not re-litigate

Everything in the previous handoff's list still stands (callback-based
escalation, 0-means-always-ask reserves, fixed rank budget shares, per-employee
`hire()` overrides). Added during Phases 4-8:

- **Effort does not feed `BudgetLedger.importance`** (roadmap open question #1,
  resolved *no*). `policy.suggested_importance()` is the explicit opt-in bridge.
- **The effort hint is the local/API threshold**, not a knob beside it —
  `cheap` takes any feasible local model, `balanced` requires `gpu_resident`,
  `effort` goes to the API.
- **One `PresetRegistry`, four instances.** Palettes are not a bespoke system
  in `observability.py`; org templates are not one in `company.py`.
- **`Team.reviewer` reviews via `review_mode="critique"`** (open question #3).
  `append` and `off` remain as modes. `max_review_rounds` counts *send-backs*
  and there is always one more review than revision.
- **Plan-then-execute lives with stub-and-fill in `delegation.py`** (open
  question #4) — one decomposition module, not two competing ones.
- **Compaction may never touch the spend/authority audit trail**
  (`ALWAYS_KEEP_KINDS`) or tool-call entries carrying an error.
- **Building a company never runs it**, in either Phase 8 mode. With no
  `model_map` everything defaults to local Ollama; with no `on_escalation` the
  default handler declines.
- **Runtime concerns stay out of `CompanySpec`** — no provider, no API key, no
  callable serializes into a design file.

## 5. File map (current)

```
src/llmadapt/
  __init__.py       - public API (58 exports)
  core.py           - Agent/Conversation/ToolRegistry
  compressor.py     - CompressionPolicy / ContextCompressor / HistoryCompactionPolicy
  hardware.py       - ResourceQuota, HardwareProfiler
  benchmark.py      - HardwareBenchmark (real CPU/GPU/PCIe numbers, disk-cached)
  local_agent.py    - find_local_model()
  selector.py       - ModelCatalog + LocalModelSelector (local fit-tier ranking)
  router.py         - ModelRouter, RoleRank
  suggest.py        - did_you_mean / closest_name              (Phase 9)
  env.py            - .env loading + layered key resolution     (Phase 10)
  archive.py        - RunArchive                               (Phase 9)
  company/          - the hierarchy, split out of company.py   (Phase 9)
    __init__.py     - re-exports every pre-split import path
    escalation.py   - EscalationEvent/Decision/Unresolved
    employee.py     - Employee
    team.py         - Team + the review-loop prompts
    company.py      - Company
  budget.py         - BudgetLedger                          (Phase 3)
  observability.py  - EventLog, org-chart renderers         (Phase 2)
  policy.py         - ModelPolicy, ApiModelCatalog          (Phase 4)
  presets/          - the four registries, split up            (Phase 10)
    __init__.py     - re-exports every pre-split import path
    registry.py     - Preset + PresetRegistry (the mechanism, alone)
    skills.py       - Skill + 15 built-ins
    personalities.py- Personality + 10 built-ins
    palettes.py     - Palette + 8 built-ins
    org_templates.py- RoleSpec/OrgTemplate + 11 built-ins
    bundle.py       - PresetBundle / default_bundle
    compose.py      - compose_system_instruction / skill_hints
  delegation.py     - stub_and_fill, plan_then_execute      (Phase 6)
  compaction.py     - LogCompactionPolicy                   (Phase 7)
  builder.py        - CompanySpec, set_company_up(text)     (Phase 8)
  gui.py            - local server for the node-graph editor (Phase 8)
  gui_assets.py     - the editor's single HTML page          (Phase 8)

tests/
  run_all.py           - the suite runner
  browser_smoke.py     - optional, needs playwright, NOT in the suite
  test_agent.py, test_full.py, test gemini ids.py, test_benchmark.py,
  test_selector.py, test_company.py, test_observability.py, test_budget.py,
  test_policy.py, test_presets.py, test_delegation.py, test_compaction.py,
  test_builder.py, test_gui.py
  test_archive.py      - RunArchive, history compaction, pins       (Phase 9-10)
  test_env.py          - .env parsing and key resolution             (Phase 10)
  test_import_paths.py - the guard on the company/ split             (Phase 9)
  test_trial_own.py    - an interactive REPL scratchpad, skipped by the runner
```

## 6. Delivery workflow — read this before an unattended run

The Phase 4-8 run was told to `git clone` and push each phase to GitHub. **It
could not.** The cloud sandbox routes git through a proxy that refuses to
inject credentials for any repository outside the session's authorized set:

```
remote: access denied by the git proxy: Shyam-Parikh-2025/llmadapt is not in
this session's authorized repository set
```

The push fails with 403 *before* the token is used, so no token, credential
helper or remote-URL trick gets around it. Cloning (read) works; pushing does
not. **Assume this is still true next time** unless the repo has been added to
the session's sources.

What worked instead, and what to plan for:

- Keep full git history in the cloud clone, committing per phase as normal.
- Sync each finished phase to the machine with `SendUserFile` +
  `mcp__remote-devices__device_commit_files`. That bridge needs the Claude
  desktop app open and connected; it stayed up for the whole overnight run.
- At the end, `git bundle create <file> --all` and deliver the bundle. It
  restores the entire history with `git clone <bundle> <dir>`, so nothing is
  lost — see `PUSH_TO_GITHUB.md`.

Also worth knowing: **the GitHub repo was several sessions behind the working
folder** when that run started (no `company.py`, `budget.py`,
`observability.py`, `selector.py`, `benchmark.py`). Check that first.

## 7. Where to pick up

`company-roadmap.md`'s "Not built yet" section is now the deferred list rather
than the plan. In rough order of value:

1. **The `company/` package split.** `company.py` has carried a note since
   Phase 0 saying it should become a package once enough neighbours landed;
   eight modules now orbit it. It was left alone during the overnight run
   because moving files mid-phase makes every phase's diff unreadable — that
   reason has expired. Do it as one standalone commit, keep the old import
   paths working via re-exports, and let the suite be the guard.
2. **Verify generated code** (Phase 6). Assembly is `compile()`-checked for
   syntax only; a confidently wrong body assembles cleanly. Actually running
   the implementers' code — even just importing it in a subprocess — is the
   increment that would make stub-and-fill trustworthy rather than useful.
3. **Budget governance v2** — the global water-filling optimizer Phase 3
   deferred. It has survived five further phases unchanged, which is evidence
   the simple version is load-bearing enough to build on.
4. **Mid-generation budget enforcement**, which needs streaming responses.
5. **Dynamic decomposition depth** (Phase 6's stretch goal) — still no clean
   formula, still research-adjacent.

## 8. Model / effort recommendation for the next run

Unchanged and still correct: for long unattended runs with real architectural
judgment in them, prefer **Opus** with extended thinking on and effort as high
as the interface allows. The Phase 4-8 run bore this out — Phase 5's registry
shape and Phase 8's GUI architecture were both decisions that everything after
them was built on top of, with nobody awake to catch a wrong turn.

For mechanical follow-ups (the package split, adding a test file), Sonnet at
high effort is a perfectly reasonable and cheaper choice.

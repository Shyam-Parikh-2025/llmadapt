# company.py roadmap — status and forward-looking design notes

Living reference for the multi-agent hierarchy work in `llmadapt`. Update this
alongside the code as phases land, so a future session (or a different AI
picking this up) doesn't have to reconstruct the plan from chat scrollback.

## Done

**Phase 0-1 — skeleton, delegation, escalation** (`company.py`, `router.py`)
`RoleRank` extended with `GENERAL_MANAGER`/`VOLUNTEER` + `ORDER` (seniority,
not capability). `Employee`/`Team`/`Company` classes. `Company.hire(name,
rank, reports_to=...)` builds an `Agent` from `model_map[rank]` and — if
`reports_to` is given — registers a `delegate_to_<name>` tool directly on the
manager's `Agent`, so delegation is just the normal tool-calling loop, no new
plumbing. Tool-iteration overflow escalates up the `reports_to` chain to
`Company._handle_iteration_escalation`, which tries the
`emergency_iteration_reserve` first (measured in *tool iterations*, unrelated
to real token cost) and only then calls the `on_escalation(event) ->
EscalationDecision` callback. `0` emergency budget means always ask first —
the deliberate cost-control default. (Renamed from `emergency_token_budget`/
`_emergency_remaining` in Phase 3 once a *real* token-denominated reserve
existed and needed the less ambiguous name.)

**Phase 2 — observability** (`observability.py`)
`EventLog` — live queryable wrapper (`by_employee`, `by_kind`, `since`) over
`Company.activity_log` / `Company.tool_call_log`, exposed via
`Company.activity()` / `Company.tool_calls()`. `Company.render_org_chart(fmt=
"ascii"|"mermaid"|"svg", theme="light"|"dark")` — the SVG renderer is a pure
Python tree layout (no graphviz/matplotlib dependency), colored with the
dataviz skill's fixed-order categorical palette (one hue per rank tier,
validated for colorblind-safe adjacent contrast; text always in ink tokens,
never the rank color itself).

**Phase 3 — budget governance** (`budget.py`, `company.py`, `core.py`)
`core.py`'s `Agent` now captures real per-provider usage on every response
(`_extract_usage`/`_record_usage`, called from `process_response` so both
sync and async paths get it) — `total_tokens_used`/`usage_log`, not a proxy
unit anymore. `budget.py`'s `BudgetLedger` does fixed percentage-of-total
splits by rank (`DEFAULT_RANK_BUDGET_SHARES`), scaled 0.5x-1.5x by a
per-employee `importance` (0.0-1.0, now a real `hire(..., importance=...)`
param). `Company._budget_gate()` runs before every `Employee.run()` — a
pre-flight check, not a mid-generation cutoff (llmadapt has no way to
interrupt a request already in flight with a real provider), checked in this
order: (1) the company-wide `total_token_budget` hard ceiling — once total
spend reaches it, *every* further request needs `on_escalation`, no internal
reallocation can cross it; (2) the employee's own rank share — if exhausted,
`_try_reallocate()` walks the `reports_to` chain for a manager with slack to
lend (a simplified, non-water-filling protocol — it grants a bonus rather
than literally debiting the lender, which is safe precisely because the hard
ceiling above is the real backstop) before escalating. `budget_exhausted`
escalations reuse the same `EscalationEvent`/`on_escalation` machinery as
`tool_iteration_limit` ones, with their own `emergency_budget_tokens` reserve
(skipped entirely once the hard ceiling itself is hit, same 0-means-always-ask
default as the iteration reserve) and `EscalationDecision.extra_token_budget`
for human-approved recovery — approving a *hard-ceiling* escalation actually
raises `total_token_budget` by the granted amount, since that's the only way
forward once the whole company is out of room. `Company.total_tokens_spent()`
and `Company.budget_report()` expose the live numbers. Per-employee
`provider`/`model`/`api_key` overrides landed on `hire()` at the same time
(open question #1 below — resolved yes, do it now).

Deliberately still simple, per the original v1 scope: fixed rank shares +
recursive ask-your-manager reallocation, not a global water-filling optimizer
— that's real scheduling, worth targeting once this simpler version has run
for a while.

**Phase 4 — auto mode model policy (local vs. API)** (`policy.py`, `company.py`)
`policy.py` adds the layer above `selector.py`/`router.allocate_local_auto`:
those rank among *local* candidates, this decides whether to be local at all.

`ApiModelSpec`/`ApiModelCatalog` are the user-supplied cost/specialty table
(per-1k input+output price, a coarse hand-set `capability` 0-1, free-form
`specialties` tags, optional `context_window`). `DEFAULT_API_CATALOG` seeds
five common models so a bare `ModelPolicy()` works, but **every seed entry
carries a `note` saying the numbers are hand-entered and will go stale** —
there is deliberately no live-pricing fetch (the flagged scope trap). Prices
are collapsed to one comparable number via `blended_cost_per_1k(output_ratio=
0.75)`; that ratio is an assumption about agent traffic (short prompts, long
completions), stated as such — it's a ranking aid, not a spend forecast.

**The effort hint is an input everywhere, from the start** —
`ModelPolicy.decide(effort=)`, `Company.hire(effort=)`, `Company.run(effort=)`.
Three levels only (`cheap`/`balanced`/`effort`) with a small fixed alias table
("needs effort", "keep it cheap", "high", …); three because a user cannot
calibrate seven levels against a catalog they wrote by hand. An unrecognized
string falls back to the rank default rather than raising, because in Phase 8
this value arrives from an LLM filling in a tool schema and a typo shouldn't
take a running company down. `DEFAULT_RANK_EFFORT` supplies the fallback per
rank, mirroring `router.allocate_compression_policy`'s rank table.

**The effort hint is what moves the local/API threshold** — that's the actual
mechanism, not a separate knob bolted next to it:
- `cheap` — any feasible local model wins, offloaded or not (free beats fast);
  API only if nothing local fits.
- `balanced` — local wins *only* at the `gpu_resident` tier (fits entirely in
  VRAM). A heavily offloaded model is slow enough that a cheap API call is the
  better trade, so it routes to the API and says so in the decision's `reason`.
- `effort` — straight to the API catalog without scoring local at all; local
  only if the API catalog is empty.

Within the API catalog the same hint picks the ranking rule: cheapest blended
cost / best capability-per-dollar / highest capability. An unmatched specialty
tag widens to the whole catalog with an explicit note rather than failing.

`PolicyDecision` is returned for every outcome including failure
(`kind="unavailable"`) rather than raising — same philosophy as
`SelectionResult.needs_install`. It carries the full `reason` string and the
raw `SelectionResult` it was based on, and is stored on `Employee.
model_decision` + logged to `activity_log` as `kind="model_policy"`, so "why
is this employee on this model?" stays answerable months later.

Failure modes are deliberately asymmetric: `mode="local"` **fails closed**
(`unavailable` + install hint) because an explicit local request should never
silently start spending money; `allow_api_fallback=True` opts in. `mode="auto"`
falls back to the API, since that's the whole point of auto. A policy that
can't route anything at `hire()` time is **logged and ignored** (the literal
`model_map` values are used) rather than fatal — a routing miss shouldn't
abort building an org chart halfway; it resurfaces as a real provider error on
first request.

`DEFAULT_LOCAL_BINDINGS` bridges selector.py's provider names (which describe
*where weights live*) to `Agent.change_api`'s transports: ollama → the Ollama
endpoint; lm-studio → `openai` + localhost:1234; hf/vllm → `openai` +
localhost:8000, with the decision's reason explicitly warning that raw HF
weights on disk are not an endpoint.

Per-task (not just per-employee) effort is real: `Company.run(task, effort=)`
calls `Company.reassign_model()`, which re-decides and swaps the endpoint via
`Agent.change_api()` — keeping conversation history, registered tools
(including every `delegate_to_*`), and usage counters. The hint applies to the
entry point only; delegates keep their own standing hints, since a manager
finding a task hard doesn't mean every intern touching it needs a frontier
model. Flagged honestly in the docstring: mid-conversation provider swaps on a
long tool-heavy history aren't hardened, the intended use is a fresh task.

**Resolved: open question #1 — effort does NOT feed `BudgetLedger.importance`.**
They answer different questions (how capable a model vs. how large a token
slice), and coupling them automatically would quietly redefine what
`importance` means in Phase 3's ledger and make budget reports unreadable.
Instead `policy.suggested_importance(effort)` is an explicit opt-in bridge the
caller passes into `hire(importance=...)` themselves. Explicit beats implicit
where the ledger is the thing standing between this library and a surprise
bill.

Also fixed while getting the suite green: `core.py`'s
`process_openai_custom_response` raised a raw `IndexError` on an empty
`choices` list instead of the clear `RuntimeError` the Gemini path already
produced (`test_full.py` was asserting the latter). Added `tests/run_all.py`,
a dependency-free whole-suite runner (skips `test_trial_own.py`, an
interactive REPL rather than a test).

**Phase 5 — one named-preset registry** (`presets.py`, `company.py`, `observability.py`)
`PresetRegistry` is the mechanism, instantiated four times — `SKILLS`,
`PERSONALITIES`, `PALETTES`, `ORG_TEMPLATES`. Same class, same API
(`get`/`register`/`resolve`/`names`/`copy`) for all four. That was the explicit
correction to honor: palettes are picked by name through the *same* mechanism
as skills, not a bespoke colour system inside `observability.py`, and org
templates are not a bespoke system inside `company.py`.

What committing to one mechanism actually bought:
- `resolve()` takes a **name or the object**, so `hire(skills=["python"])` and
  `hire(skills=[Skill(...)])` both work and a one-off preset needs no
  registration (passing an object deliberately does *not* register it).
- `names()` is where Phase 8's schema enums come from — one method, four kinds,
  via `PresetBundle.names()`/`describe()`.
- `PresetBundle.fork()` gives a Company private membership, so registering a
  custom skill can't leak into another Company in the same process (or another
  test). Presets are treated as immutable value objects, so a fork isolates
  membership only.
- `get()` raises listing what *is* available and never returns None — a typo'd
  skill name would otherwise silently produce an employee with no skills.
  `register()` refuses to shadow an existing name without `overwrite=True`.

`compose_system_instruction()` templates base → role line → personality →
skills → constraints, in a **fixed order**, because a system instruction that
reshuffles between runs makes prompt caching useless and behavior changes
impossible to attribute. Constraints are pooled across skills and deduplicated.
With no presets it returns `""`, so a pre-Phase-5 hire is byte-identical.

Skills carry a `specialty`/`effort` hint that feeds **Phase 4's policy** — but
only as a fallback; an explicit `hire(specialty=…)` always wins. That is what
makes the two phases one system rather than two.

Org templates (`OrgTemplate`/`RoleSpec`) are declarative — roles refer to each
other by key, and `validate()` catches dangling reporting lines, cycles, and
duplicate leads before anything is half-built. **Scaling is opt-in**:
`build_from_template(name, size=…)` takes the size as an argument and never
infers it from the task text, because inferring it means guessing at spend on
the user's behalf — the thing Phase 3 exists to prevent. Roles declare
`count_by_size`, so "large" multiplies worker tiers while leaving oversight
roles alone, and a role whose manager doesn't exist at a smaller size reports
to that manager's manager instead (so one template works at every size without
a rewritten small-only copy). Built-ins: `solo`, `small-coding-team` (the
roadmap's reference shape), `research-pod`, `writing-desk`.

Palettes: `dataviz` (the pre-Phase-5 colours, still the default, so every
existing `render_org_chart()` call renders identically), plus `grayscale`,
`ocean`, `ember`. **Only `dataviz` claims the colourblind-safe validation** —
the others are stylistic and say so, rather than implying validation by sitting
in the same list.

**Resolves open question #3 — `Team.reviewer` now actually reviews.** The
decision (it needed one, not just an implementation): `review_mode="critique"`
is the default — the lead drafts, the reviewer either signs off or lists what
must change, and a requested change goes back to the lead. Chosen because it is
the only option where the reviewer's opinion can change the delivered output,
which is what "reviewer" means outside this codebase. The alternatives are kept
as modes rather than discarded: `"append"` (reviewer independently answers, both
returned labelled — a second opinion, not a review) and `"off"` (exactly the
pre-Phase-5 lead-only behavior).

Two deliberate bounds on that loop, both cost-driven:
- `max_review_rounds` defaults to **1**, counting *send-backs*. Two agents can
  disagree forever and each round costs real tokens — an unbounded critique
  loop is precisely the hazard Phase 3 exists to prevent.
- There is always **one more review than revision**, so the final revision is
  looked at before it ships. Without that pass a revised draft would go out
  carrying a critique written about the previous version.
- If an objection still stands when the rounds run out, the work ships **with
  the outstanding critique appended** rather than the objection being silently
  dropped or an exception raised. The caller gets both and decides.

Flagged simplification: approval detection is a *convention* — the reviewer is
asked to reply exactly `APPROVED` and `_looks_approved()` checks the opening of
the reply (rejecting obvious negations). A structured tool call would be
stricter but would force every reviewer onto a tool-calling-capable model,
which the cheap local tiers this library targets often are not. A reviewer that
buries "APPROVED" mid-critique will be misread.

**Phase 6 — task decomposition: stub-and-fill + plan-then-execute** (`delegation.py`)
Both strategies in one module, deliberately. They are the same idea at
different granularities — a more capable agent decides the *shape*, cheaper
agents fill it in — and splitting them would create two competing
decomposition systems, which is what the roadmap warned about when it folded
"plan-then-execute" (open question #4) into this phase.

*stub-and-fill*: the architect emits a module of signatures + docstrings +
type hints with `...` bodies; implementers fill one body each; results are
reassembled. **Reuses `ContextCompressor.code_to_stub()`** rather than writing
a second AST layer — it already produces exactly the signature+docstring+`...`
shape, so it normalizes the architect's output into the interface implementers
see (a chatty architect that half-implements something still hands over a clean
interface).

*plan-then-execute*: the planner emits a numbered plan, steps run in order, and
each step sees a **compressed digest** of earlier results
(`compress_tool_output`, the same machinery tool results already use) rather
than their full text — feeding every prior step's output into every later step
is how a five-step plan becomes a context-window failure. `synthesize=True`
gives the planner a final pass; set it False when the last step is already the
deliverable.

**Fixed by rank in v1**, as decided: `DEFAULT_ARCHITECT_RANKS` (most senior
first — architecting is the leverage point, a bad interface makes every filled
body wrong) and `DEFAULT_IMPLEMENTER_RANKS` (cheapest first — bodies are the
high-volume, low-leverage part, which is the economic point). Implementers get
stubs round-robin, deliberately *not* "best implementer per stub", because
ranking stubs by difficulty needs exactly the dynamic-decomposition judgment
this phase decided not to fake.

Failure handling is the bulk of the design, since both strategies are driven by
model output that will sometimes be malformed. Nothing raises for a
model-quality failure — everything lands on the result object and in the
activity log:
- unparseable architect module → `parse_error`, and **no implementer is
  called** (nobody pays for an unusable plan)
- implementer returns another stub, or renames the function → that stub is
  reported `unfilled` rather than silently accepted; a surviving `...` would
  produce a syntactically valid module that does nothing, the worst failure
  mode available here
- unfilled stubs stay **visible in the assembled module** under an
  `# UNFILLED (reason):` comment rather than being dropped, so the artifact
  says where it is incomplete
- architect returns no stubs → its answer is returned verbatim rather than
  inventing stubs to justify the strategy
- a plan that doesn't parse → the whole task runs as a single step with
  `degraded=True`, rather than failing
- a step that raises → recorded on that step; the remaining steps still run

`MAX_STUBS` (20) and `MAX_PLAN_STEPS` (12) are runaway guards in the same
spirit as Phase 3's budget ceiling and Phase 5's `max_review_rounds`: each stub
and each step is a real model call, and a model asked to decompose a vague task
will happily produce forty.

`Company.run_structured(task, strategy=…)` is the entry point; the structured
strategies return result *objects*, not strings, because "did every step
succeed?" and "which stubs came back unfilled?" are the questions a caller
actually has. `effort`/`specialty` work as in `run()`, re-routing the
architect/planner through the Phase 4 policy.

Stated limits: nothing here executes or tests the generated code —
assembly is checked with `compile()` for **syntax only**, so a confidently
wrong body assembles cleanly. An implementer that keeps the function name but
changes the parameter list is accepted as-is. Plan-step parsing is a
numbered-list regex over prose; models are good at numbered lists and bad at
guarantees, which is why the no-match path degrades instead of erroring.

**Phase 7 — log compaction** (`compaction.py`, `company.py`)
The "keep broad concepts, drop specifics" pass over `activity_log` /
`tool_call_log`, run automatically at the end of every completed top-level task
(`run()` / `run_structured()`) — a task boundary is exactly when the specifics
of *how* it got done stop mattering and the shape is what you keep.

**Reuses `compressor.py` twice over**, as instructed — not a second compression
system:
- *the code*: `compress_tool_output` shrinks result previews,
  `_dedupe_repeated_lines` collapses repetition before summarizing,
  `token_estimate` measures what was saved (so the numbers are comparable with
  what `CompressionPolicy` budgets against elsewhere).
- *the pattern*: `LogCompactionPolicy` is deliberately shaped like
  `HistoryCompactionPolicy` — same three modes (`off`/`algorithmic`/`agent`),
  same untouched `keep_recent` tail, same trigger threshold, same
  summarizer-optional contract, same "a typo'd mode degrades to the free/safe
  path", same "agent mode falls back to algorithmic if the summarizer fails, so
  choosing it can never break a run". Anyone who has configured history
  compaction already knows how to configure this.

Algorithmic mode collapses old activity events into one rollup carrying counts
per kind, the employees involved, and the time span — every individual task
string dropped. Tool calls group by `(employee, tool_name)` into one row with
call count, total/mean duration and one compressed example preview, which is
the useful residue of a thousand identical `delegate_to_worker` calls.

**The one genuinely new decision: what compaction may not touch.** Compaction
is lossy and irreversible, and these logs are the only record of where the
money went. `ALWAYS_KEEP_KINDS` protects escalations, escalation decisions,
emergency-reserve draws and budget reallocations at *every* mode regardless of
age; tool-call entries that recorded an **error** are likewise never collapsed,
since an error is the specific thing you go back to a log to find. The rule for
what belongs on that list: if losing the event would make "who approved this
spend?" or "what went wrong?" unanswerable after the fact, it stays.

Compaction assigns **in place** (`log[:] = …`) rather than rebinding, because
Phase 2's `EventLog` deliberately wraps the company's list live — rebinding
would leave any `EventLog` a caller already holds pointed at the stale list.
The policy's own methods never mutate the list passed to them; only
`compact_company()` writes.

**Phase 8 (text/schema mode) — `set_company_up(mode="text")`** (`builder.py`)
The AI-callable company builder. The design constraint driving everything: the
*calling model* must be able to see its options and fill them in correctly.

The schema is **generated by `core.ToolRegistry`** from `set_up_company`'s type
hints and docstring — not a hand-written parallel format, as instructed. The
payoff is concrete: the tool an agent calls and the schema it reads come from
the same function, so they cannot drift (a test asserts they match). Because
`ToolRegistry`'s generator maps flat scalars, `set_up_company` takes **flat
scalar parameters** plus one `employees_json` escape hatch for org shapes no
template covers — a nested-object schema would have meant writing a second
schema generator, the exact thing this phase was told not to do.

`company_options()` pulls the valid names for every preset kind straight from
Phase 5's `PresetBundle.names()`, and `company_setup_schema()` injects them as
`enum`s and into the parameter descriptions. **This is why text mode had to
wait for Phase 5**: a schema saying `template: str` is useless to a model,
while one carrying the actual template, skill and personality names is fillable
in one shot. It reads the registries live, so a forked bundle's custom presets
show up too.

`CompanySpec`/`EmployeeSpec` are the serializable format (the same one `gui.py`
saves and loads, so either half of Phase 8 can consume the other's output).
`validate()` **returns** a list of problems rather than raising on the first —
the caller is often a model filling in a schema, and "here are the four things
wrong" beats a traceback. Every message lists what *is* allowed. It catches
unknown fields, unknown ranks/skills/personalities/templates/palettes,
duplicate names, dangling `reports_to` (including against names a template will
create, since a spec may extend a template), and reporting cycles.

Three safety decisions, all in the same direction — an AI must not be able to
spend money by accident:
- Building **runs nothing**. `set_up_company` returns the org, never a result.
  A tool that could build *and* start a company is one hallucinated argument
  away from a bill.
- With no `model_map`, every rank defaults to **local Ollama** — the only
  provider needing no API key — so an unconfigured build cannot spend.
- With no `on_escalation`, the default handler **declines**, matching the
  0-means-always-ask reserves from Phases 0-3: "no human attached" must mean
  stop, never approve-yourself.

Runtime concerns (`model_map`, `on_escalation`, `model_policy`, `presets`,
`log_compaction`) are arguments to `build_company`, deliberately *not* spec
fields: they're callables and policy objects, they don't serialize, and putting
them in the spec would let an AI-authored JSON blob name a provider and a key.
The spec describes the org; the caller supplies the runtime. Every default
taken is reported in `BuildResult.warnings` rather than applied silently.

**Phase 8 (GUI mode) — the node-graph company editor** (`gui.py`, `gui_assets.py`)
The half the roadmap called "meaningfully bigger than everything built so far
combined", and it is: a small local web app, not a Python function.
`http.server` serves one self-contained page plus a small JSON API, opened with
`webbrowser.open()`. Vanilla JS and SVG — no framework, no CDN, no build step,
so the zero-dependency rule holds on both sides of the wire.

**The division of labour is the design.** The browser owns *interaction*
(dragging, rubber-band and shift multi-select, the right-click bulk actions);
Python owns *truth* (which preset names exist, whether a design is valid, what
a template expands to). The page hardcodes no rank, skill, template or palette
— it fetches them from Phase 5's registries at load, so adding a skill in
`presets.py` makes it appear in the editor with no front-end change. A test
asserts the JS contains none of those names.

Implemented: click-to-connect (click the report, then the manager), shift-click
and rubber-band multi-select, right-click bulk actions including "connect all N
to one manager", "add a skill to all", duplicate and delete, a left panel for
per-employee rank / manager / skills / personality / effort / policy mode /
provider / model / importance, a left menu for templates, palettes and review
settings, live validation against the server, auto-layout, and drag-to-move
with positions persisted in `CompanySpec.layout` (which `build_company()`
ignores entirely — it exists so reopening a design doesn't scramble the
picture).

Guards, none of them incidental:
- The server binds **127.0.0.1 only** and every request must carry a random
  per-session token issued at launch, so another local process — or a page in
  another tab — can't drive the editor by guessing the port. Not an auth
  system; the minimum that keeps an unauthenticated local port from being a
  foot-gun.
- **API keys are not editable and never enter a saved design**, and the page
  says so where a user would look for the field. They come from the
  `model_map` passed to `build_company()`, so a design file can be shared or
  committed safely — the same rule text mode set.
- Loading a template **expands it into concrete employees** rather than storing
  a template reference: once nodes are dragged around, "this is the
  small-coding-team template" stops being true, and a spec claiming a template
  it no longer matches is worse than one that just lists its employees.
- A build that fails validation **leaves the editor open** instead of returning
  a broken company; only a successful build releases the waiting caller.
- The GUI builds an org and **runs nothing** — identical to text mode, for
  identical reasons.

Testing, honestly split: `tests/test_gui.py` (30 checks, in the suite) drives
every route over real HTTP and statically checks the page — one script block,
no external loads, balanced structure, no hardcoded vocabulary, every fetch
carrying the token, every called route actually served, user text escaped
before insertion. `tests/browser_smoke.py` (**not** in the suite, and its
filename deliberately doesn't match the runner's pattern) drives a headless
Chromium through the DOM behaviour that genuinely can't be covered otherwise —
template load, add, rename, drag, connect mode, the right-click menu, the
cycle guard, check and build — failing on any console error. It needs
playwright, which llmadapt does not depend on and shouldn't acquire for one
module. It was run during this phase and passes with no console errors.

**Phase 9 — the TODO pass: package split, cost-weighted budget, run archive**
(`company/`, `suggest.py`, `archive.py`, `budget.py`, `policy.py`, `delegation.py`,
`presets.py`, `builder.py`, `gui.py`, `core.py`, `compressor.py`, `compaction.py`)

Not a new feature phase. This was a read of the whole codebase against the
`# TODO` decision points left in it, plus the working-machine edits made after
the Phase 4-8 run. Every TODO in the library is now closed — either
implemented, or answered in the docstring where it stood so the next reader
gets the reasoning rather than the question.

**The `company/` package split, finally.** The note `company.py` had carried
since Phase 0 came due at 1118 lines. Only its own classes moved —
`escalation.py`, `employee.py`, `team.py`, `company.py` under `company/` — and
the eight orbiting modules stayed exactly where they were, so
`from llmadapt.policy import ModelPolicy` is untouched and the split fixes one
oversized file rather than reorganising the library. `company/__init__.py`
re-exports everything the old module did. `tests/test_import_paths.py` is the
guard: all 54 public exports resolve from the package root, the re-exported
classes are the *same objects* (a shim that rebuilt a class would break
`isinstance` and `except EscalationUnresolved` across module boundaries), and
no sibling module has been pulled under `company/`.

**Decorators were silently disappearing** (`delegation.py`). The architect
prompt had been edited to ask for decorators, which exposed a real defect:
`ast.get_source_segment()` starts at the `def` line, and since Python 3.8 a
decorated node's `lineno` points at the keyword rather than the first
decorator. So decorators fell outside the slice in both `extract_stubs()` and
`_implementation_for()`, and — belonging to the same node — were not emitted as
preamble either. An `@property` or an `@functools.cache` simply vanished, and
the assembled module compiled cleanly without it. `StubPlan.interface()` made
it worse: it goes through `code_to_stub()`, which unparses the AST and so
*kept* the decorator the assembler was about to drop, meaning the implementer
saw one thing and the artifact got another. This is precisely the failure mode
Phase 6 says it exists to prevent — a syntactically valid module that is
quietly wrong — one line above where it was already being prevented.
`_source_of()` widens the range; `Stub` records the normalized decorator
expressions; an implementer that drops or swaps one is reported `unfilled`,
the same rule as one that renames the function.

**The architect's one-line shortcut, kept but bounded.** The prompt now lets
the architect implement a function outright when it fits on one line and needs
no docstring. Worth keeping — a trivial helper is not worth an implementer
round-trip at the cheap tiers this library targets — but it is the only code in
the assembled module that no second agent ever looks at. So: allowed one-liners
are recorded on `StubPlan.architect_implemented`, surfaced on the result and
logged; anything past `MAX_ARCHITECT_BODY_LINES` (1) is stripped back to a stub
and put in the pool with a `demoted_reason`; and a function the architect wrote
*with* a docstring is still a deliberate helper kept verbatim in the preamble,
exactly as before. The rule the architect is given is narrow, so the check is
narrow to match.

**Not all tokens are equal** (`budget.py`) — the one that had stopped being
theoretical at Phase 4. An intern on local Ollama and a C-suite on a frontier
API model both spend "tokens", and the ledger charged them identically, so a
half-local company had a ceiling that did not mean what it looked like.
`CostModel` reads prices that **already exist** rather than adding a second
table to hand-maintain (the exact staleness Phase 4's own docstring warns
about): `hire(cost_weight=)`, then `PolicyDecision.estimated_cost_per_1k`, then
`model_map[rank]["cost_per_1k"]` (so cost weighting works with no ModelPolicy
attached at all), then a catalog match, then local detection — including any
`localhost` base_url, since policy.py routes LM Studio through
`provider="openai"` at a local port and the provider name alone would read that
as paid traffic. Unknown falls to 1.0, deliberately not 0.0: the errors are
asymmetric, and guessing free makes the budget unenforceable.

Weight 1.0 means "a typical model in your catalog" — the median blended cost,
computed once and **frozen**, because a baseline that drifted as employees were
hired would make the ledger non-monotonic and stop an hour-old budget report
from reconciling. Opt-in via `cost_weighted_budget=True`; off, every number is
byte-identical. `budget_report()` carries `raw_tokens`, `cost_weight` and a
`cost_basis` string beside the charged figure — the weighted number explains a
budget decision, the raw one is the only figure that reconciles against a
provider's usage page. `BudgetLedger.charged_spend()` is the single place the
ledger's unit is decided.

**Silent fallbacks made visible** (`suggest.py`, `policy.py`, `presets.py`). A
misspelled *skill* always raised a KeyError listing every valid name. A
misspelled *effort* did not: `resolve_effort("balnced", "JUNIOR")` quietly
returned `"cheap"` — the lowest tier — so one character silently downgraded an
employee's model. Phase 4's decision to fall back rather than raise stands (the
value comes from an LLM filling a schema), but the fallback now warns, suggests
the nearest level, and lands in `PolicyDecision.reason`. `suggest.py` holds the
one `difflib` helper both halves use — its own module because `presets.py` and
`policy.py` are deliberately independent of each other. Plain strings stay the
wire format everywhere: they have to survive a JSON spec and a tool call, which
an enum cannot.

**`default_policy_mode`** (`company/`, `builder.py`, `gui.py`). `mode="local"`,
`mode="api"` and `allow_local=False` always existed per hire; what was missing
was saying it once. "This whole company stays local" previously meant passing
`mode="local"` to every `hire()` call forever, and one omission meant that
employee silently started spending. Precedence is narrowest-wins:
`hire(mode=)` > `model_map[rank]["mode"]` > the company default. It is a
`CompanySpec` field (it names no vendor and carries no credential, so the
no-runtime-in-the-spec rule holds) with a control in the editor. Two things
worth noting: `reassign_model()` defaulted to `mode="auto"` outright, which
broke the guarantee on the per-task path — a local-pinned company re-routed to
the API the moment `run(task, effort=...)` reassigned the entry point — and a
bad `default_policy_mode` **raises at construction**, the one place in the
routing path that does, because unlike a per-hire mode it is set once by hand
and silently ignoring it means a company the caller believes is local paying
for its entire life.

**`total_token_budget=0` still means unlimited, loudly** (`builder.py`). A magic
sentinel for "unlimited" is one more thing for an AI to fill in wrong, and
`validate()` rejecting a plausible `0` turns an omission into a failed build.
So the meaning is unchanged and a `NO SPEND CEILING` warning goes into
`BuildResult.warnings` instead, with the schema text saying so in the words a
model reads. Building still runs nothing, so nothing is spent either way.

**`archive.py` — the uncompacted record.** Compaction is lossy in two places,
and `ALWAYS_KEEP_KINDS` only guarantees "who approved this" and "what broke".
`RunArchive` is one JSONL file per run, written **through** — at log time, not
at compaction time, because archiving a view compaction has already touched
archives the wrong thing, and a run that dies before the first compaction would
otherwise leave nothing at all, which is exactly the long unattended run this
is for. One mechanism with two writers (the company log and the conversation
transcript) rather than two half-identical file sinks — the Phase 5
`PresetRegistry` move again. Off by default; any I/O failure disables it with a
warning rather than touching the run; JSONL so a crash-truncated file still
reads to the last complete line. `delete_on_clean_exit` covers "delete it if
everything went fine", gated on an explicit `close(clean=True)` and never on
`atexit` — which also fires on the paths where things went wrong, and silently
deleting a bad run's record is the opposite of the point. `Company.finish()`
decides what clean means and refuses to say so with an unresolved escalation
in the log. Its own label field is `stream`, not `kind`, because activity
events already carry a `kind` and an archive field by that name silently
overwrote it.

**`HistoryCompactionPolicy` was never wired to anything.** Written in Phase 3
with a full docstring, three modes and a demo — and no call site anywhere in
the library, not exported from `__init__.py`, and no test file. History
compaction was a class you had to reach into `compressor.py` and drive
yourself. `Agent(history_compaction=)` now runs it at the top of
`generate()`/`generate_async()` — there rather than inside the tool loop, so a
`mode="agent"` summarizer is not paid for several times in one turn. `None`
still means never touch history. It assigns in place, as `compact_company()`
does; a policy that raises is logged and skipped, the same contract as agent
mode falling back to algorithmic — choosing compaction can never be a way to
break a run.

**Smaller decisions closed in place.** `compose_system_instruction()` stays in
the system instruction rather than being said once in the first message: it is
already composed once at `hire()`, a stable system prefix is the part providers
actually cache, and moving it into a user turn would put the employee's
identity inside the region `HistoryCompactionPolicy` may summarize away —
silently becoming a different employee at turn 40 is a far worse failure than a
few repeated tokens. Commenting rules became their own `commenting` skill
rather than a paragraph inside `python`, because they are not about Python and
folding them in means copying them into the next language's skill (they were
also concatenated without spaces, producing "small enough to test.Include
docstrings" in every request). `compress_tool_output` with `budget <= 0` still
drops the output — raising would kill a run over a config slip in a truncation
utility, and returning the original would break the budget it exists to
enforce — but now warns, since silently discarding a tool result is expensive
to debug from the far end.

**Phase 10 — pinned context, preset catalogs, and keys from the environment**
(`env.py`, `presets/`, `core.py`, `compressor.py`, `company/`)

Four requests from the working machine, one of which turned out to rest on a
premise that doesn't hold and was rebuilt into the thing it was reaching for.

**Pinned messages — what compaction may never touch, conversation side.**
The request was to send skills and personality once, in the first message,
"so it does not need to be sent in every system instruction". The premise
fails: every provider here is **stateless**. Anthropic, OpenAI, Gemini and
Ollama keep no server-side memory, so the entire history is re-sent on every
request and a first user message is re-sent exactly as often as a system
instruction. There is no send-once option at the API level to take.

Moving it would also have cost real money and real correctness. `core.py` puts
the system instruction in `payload["system"]` (Anthropic) and
`payload["systemInstruction"]` (Gemini) — **top-level fields, not messages**,
which is precisely the stable prefix providers cache against. Relocating that
text into the message array takes it out of the cached prefix and gets it
re-billed at full rate every turn. And `HistoryCompactionPolicy` protects
`history[0]` *only when its role is "system"*, so the identity would have
become eligible for summarization — an employee silently ceasing to be the
employee you hired, around turn 40, with nothing reporting it.

What the idea was actually reaching for is that the *only* protected thing was
that one system message. A spec agreed at turn 3, an interface everything
downstream targets, a correction the model already got wrong once — all fair
game. So: `Conversation.pin()`/`pin_last()`, with `Agent.pin_context()`,
`Employee.pin_context()` and `Company.pin_context()` at each layer. This is the
conversation-side twin of Phase 7's `ALWAYS_KEEP_KINDS`, and the rule for what
earns a pin is the same one: if losing it would change the answers, it stays.

Two details that are the whole implementation:

- The marker is `PIN_KEY = "_pinned"`, defined in `compressor.py` (the import
  runs core → compressor, and the policy that honours it lives there). The
  leading underscore is load-bearing — `export_for()` already strips
  `_`-prefixed keys before a message goes over the wire, the convention
  `_native` established — so a pin is bookkeeping this library sees and no
  provider ever does. A test asserts that for all four providers.
- `compact()` walks the old groups and flushes runs of unpinned ones **around**
  each pin, rather than compacting everything and re-inserting the pins after.
  That is what preserves chronology. A pinned turn from step 2 re-emitted after
  a digest covering steps 1–9 reads to the model as though it happened later,
  and nothing in the transcript would say otherwise.

**`presets/` — a package, and catalogs worth picking from.** `presets.py` was
748 lines and about half of it was content rather than mechanism. Split into
`registry.py` (the mechanism, alone), `skills.py`, `personalities.py`,
`palettes.py`, `org_templates.py`, `bundle.py`, `compose.py`. Adding a built-in
is now an edit to one small catalog file, and the mechanism cannot grow a
per-kind special case by accident because it no longer sits beside any one
kind's data. Same contract as the `company/` split: every import path
preserved, `tests/test_import_paths.py` the guard.

Catalogs roughly tripled — skills 6→15, personalities 4→10, palettes 4→8, org
templates 4→11 — each entry following the rules the existing ones set: one
skill does one job, constraints are real "don't" rules rather than restated
instructions, and an `effort` hint is declared only where high effort is
genuinely warranted, since it routes every employee holding that skill to an
expensive model. `high-contrast` states in its own description that it is a
*luminance* choice and not the colourblind validation, because only `dataviz`
carries that and a palette in the same list would otherwise imply it.
`security-review-board` is the one template whose worker tier is `SENIOR`
rather than `JUNIOR`, and says why: a security finding is expensive when wrong
in either direction.

The new tests are the invariants that rot silently as a catalog grows — a
template naming a renamed skill, a palette missing a rank, a skill claiming an
effort level that doesn't exist — plus a real build of all 11 templates at all
3 sizes. The Phase 8 schema grew to 3.4KB with the bigger enums, still one-shot
readable, which is why `company_options()` reads the registries live instead of
hardcoding names.

**`env.py` — keys from the environment, and from `.env`.** Confirmed first that
per-rank model defaults already work exactly as asked: `model_map` is that
feature, and `hire()` overrides it per employee. Key resolution was the half
that was broken, and broken invisibly.

`core.py` resolved a missing key with `os.getenv(PROVIDER_API_KEY)`, which
reads the *process* environment only. A `.env` at the project root — where this
project's own `GEMINI_API_KEY` actually lives — was never opened by anything in
`src/`. The only `load_dotenv()` calls were in a test scratchpad and in
`core.py`'s own demo block, and `dotenv` is a third-party package, which has no
business being imported by a library whose first stated convention is zero
third-party dependencies. The key was present, the code looked like it should
find it, and it didn't. That import is gone.

The parser handles the forms that actually appear, including `KEY = value` with
spaces around the `=` — which is how this project's `.env` is written, and
which a naive split reads as a key named `"GEMINI_API_KEY "` that matches
nothing. Loading **never writes into `os.environ`**: the file is a fallback
consulted only for names the real environment lacks, so an exported variable
always wins and importing llmadapt cannot change the environment of the program
that imported it. The upward search is bounded at six levels and stops at a
filesystem root, so a script run from an odd directory cannot wander into a home
directory and pick up another project's credentials.

One key per provider is right for the big three and wrong the moment two
endpoints share a provider *protocol* but not an account — OpenRouter,
Together, Groq and a local vLLM are all `provider="openai"` to this transport.
Resolution is layered, most specific first: explicit `api_key` →
`model_map["key_env"]` → `<ALIAS>_API_KEY` → `<PROVIDER>_API_KEY`. The last is
the original behaviour, so every existing config is untouched.

A missing key **warns rather than raises** — the endpoint may be a local one
behind a `base_url` — and the warning names every variable that was tried,
because "no key" alone doesn't tell you which name you were supposed to set.
`Company.key_sources()` answers the same question for a whole org and returns
provenance only, never a key.

## Not built yet

Nothing from the original Phase 0-8 plan remains. What's left is the
deliberately-deferred work called out in the entries above:

- **Budget governance v2** — a real global water-filling optimizer instead of
  Phase 3's fixed rank shares + ask-your-manager reallocation. Flagged since
  Phase 3 as worth targeting once the simpler version has proven the escalation
  path end to end. It has now run through six more phases without needing to
  change, which is some evidence the simple version is load-bearing enough.
  Phase 9's cost weighting changed the *unit* the shares are denominated in,
  not the allocation algorithm, so this is still open exactly as stated.
- **Dynamic decomposition depth** (Phase 6's stretch goal) — deciding how far
  to break a task down from the task itself, rather than fixing architect and
  implementer tiers by rank. Still no clean formula; still research-adjacent.
- **Verifying generated code** — Phase 6 checks assembled modules with
  `compile()` for syntax only. Actually running or testing what the
  implementers wrote is the obvious next increment, and the one that would
  make stub-and-fill trustworthy rather than merely useful.
- **Mid-generation budget enforcement** — Phase 3's gate is pre-flight, because
  llmadapt cannot interrupt a request already in flight. Streaming responses
  would make a real mid-response cutoff possible.
- ~~**A `company/` package split**~~ — **done in Phase 9**, as one standalone
  commit with `tests/test_import_paths.py` as the guard.

- **Pinning by policy rather than by hand** — Phase 10's pins are placed
  explicitly. A policy that pinned automatically (every escalation decision,
  every `APPROVED` from a reviewer) would be the natural next step, and is the
  same question `ALWAYS_KEEP_KINDS` answered for the log side. Left alone
  because an automatic pin that fires too often silently makes compaction
  useless, and nothing would report that.

- **Mid-generation history compaction cost** — Phase 9 wired
  `HistoryCompactionPolicy` into `Agent.generate()`, which runs it once per
  turn. A very long tool loop inside a single turn can still outgrow the
  context window without compaction getting another chance; running it
  per-iteration would fix that and would also pay a `mode="agent"` summarizer
  several times per turn. Needs a real trigger, not a call site.

- **Cost weighting from live usage rather than a per-1k estimate** — Phase 9's
  `CostModel` weighs an employee by their model's blended price, which assumes
  the `output_ratio` Phase 4 documents. `Agent.usage_log` already holds the
  real input/output split per call, so charging the actual mix is possible and
  would remove the assumption. Not done because it makes an employee's weight
  vary per call, which complicates the reallocation walk.

## Open questions

1. ~~Should the effort hint feed `BudgetLedger.importance`?~~ **Resolved in
   Phase 4: no.** Kept as separate knobs, with `policy.suggested_importance()`
   as an explicit opt-in bridge. Reasoning in the Phase 4 entry above.
2. Ready-made color palettes + custom palette support for `render_org_chart`
   (deferred until Phase 5's named-preset registry exists) — addressed by
   Phase 5's palette registry.
3. ~~**`Team.reviewer` is structural only**~~ **Resolved in Phase 5** — see the
   Phase 5 entry for the decision and its bounds. Original note: — `Team.__init__` stores a reviewer
   and the class comment promises "at least one reviewer by default", but
   `Team.run()` just calls `self.lead.run(...)` and the reviewer is never
   invoked. Needs a *decision* on what review means (reviewer also gets the
   task and their output is appended? reviewer critiques the lead's output and
   can send it back?), not just an implementation. Folded into Phase 5, which
   owns team templates.
4. ~~**"Plan-then-execute for complex tasks"**~~ **Resolved in Phase 6** — built
   alongside stub-and-fill in `delegation.py`. Original note: was in the original spec alongside
   stub-and-fill and skills, but had no home in Phases 4-8. Folded into Phase 6
   — stub-and-fill is itself a form of planning, and doing both in one module
   avoids two competing task-decomposition systems.

5. ~~**Should compaction keep an uncompacted copy of everything?**~~
   **Resolved in Phase 9: yes, as an opt-in file.** `archive.RunArchive`, off
   by default, written through at log time. See the Phase 9 entry for why a
   file rather than a second in-memory list, and why deletion is gated on an
   explicit clean finish rather than `atexit`.
6. ~~**Should misspelled names raise or fall back?**~~ **Resolved in Phase 9:
   both, unchanged — but neither silently.** Presets keep raising, efforts keep
   falling back (an LLM fills that field), and both now suggest the nearest
   real name. The asymmetry was never the problem; the silence was.

7. ~~**Can the presets be sent once, in the first message, instead of in every
   system instruction?**~~ **Resolved in Phase 10: no, and the premise doesn't
   hold.** Every provider here is stateless, so the full history — first user
   message included — is re-sent every turn regardless. On Anthropic and Gemini
   the system instruction is a top-level payload field, i.e. the cached prefix,
   so moving it would cost more rather than less, and it would put the
   employee's identity inside the region compaction may summarize away. Pins
   were built instead, which is what the idea was reaching for.
8. ~~**Should API keys come from the environment per model?**~~ **Resolved in
   Phase 10: per endpoint, layered.** Per-provider was already there and stays
   as the fallback; an alias or an explicit `key_env` handles the case that
   actually needs it, which is two same-protocol endpoints on different
   accounts. Per-*model* variables were rejected: model IDs contain dots,
   colons and dashes, so the name-mangling has to be guessed identically by the
   user and the code, and a mismatch fails silently as "no key".

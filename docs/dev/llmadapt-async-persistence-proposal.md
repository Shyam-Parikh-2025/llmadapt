# llmadapt — asyncio + persistence: audit findings and outcome

**Status:** all five phases (A–E) implemented and shipped, plus the preset expansion.
**Tests:** started at **17 files, 433 checks**. Now **22 files, 484 checks**, all passing, run three times over to check the timing-sensitive concurrency cases are not flaky.

---

### Decisions taken

| # | Decision | Chosen |
|---|---|---|
| 1 | Async delegation model (§2.2) | **Option B — async all the way down** |
| 2 | Sibling-failure semantics (§2.5) | **No cancellation**, except control-flow types — decided during Phase C |
| 3 | Paused-run API (§3.4) | **New opt-in `run_resumable()`**; `run()` unchanged |
| 4 | Sync `generate()` as an `asyncio.run` wrapper | **Yes — `_execute_with_thread` and the blocking `spinner()` deleted** |
| 5 | API keys in the snapshot | **Never** — checked on the way out, not trusted |
| 6 | Scope | Phase A first, then B–E completed in sequence |

### What shipped

**Phase A — shared dispatch + the §0 bug fix.** `core.ToolControlFlow`; `ToolRegistry.execute()` re-raises it instead of stringifying; `EscalationUnresolved` subclasses it. `Agent._run_tool_calls` is the single place invoke-and-compress is written, with all four `process_*_response` methods normalizing to `{"id","name","args"}`.

**Phase B — async core.** `generate_async` is the implementation; `chat`/`generate` are wrappers over it via `core.run_coroutine_blocking`, which detects a running loop, warns loudly about the block it cannot remove, and runs the coroutine on a helper thread. `ToolRegistry.execute_async` awaits `async def` tools directly and pushes sync ones to a worker thread — which is also what keeps a nested `Agent.chat()` inside a sync tool from tripping the fallback. `_maybe_compact_history` moved off the loop.

**Phase C — async company, real parallelism.** `Employee.run_async`, an `async def` `delegate_tool`, `Company.run_async`/`run_structured_async`, async twins for `Team.run` and both delegation strategies. Tool calls now fan out through `asyncio.gather(return_exceptions=True)`. Guards: a company-wide budget lock held across the *decision only* (holding it across an escalation would deadlock, since an approved retry can itself delegate), a per-employee lock so one employee's conversation cannot interleave two tasks, and `on_escalation` accepted as sync or async — a blocking handler runs off the loop so other employees keep working. `stub_and_fill` fans out (stubs are independent); `plan_then_execute` deliberately does not (each step feeds the next).

**Phase D — state snapshot.** `state.py`: a versioned `CompanyState` document, `Company.save_state()` / `restore_state()`. Rehydrate-into-a-live-company, since a company holds callables no file can carry; roster, ranks, reporting lines and tool names are all verified, and a mismatch raises listing every difference. A credential-shaped-field check runs on the way out.

**Phase E — pause/resume.** `ESCALATION_PENDING` as a distinct falsey sentinel (not a flag on `EscalationDecision`, where it would read as a decline), `RunPaused`, `run_resumable()` / `resume()`, and `PausedRun`. The mechanism: a pause rolls the interrupted turn out of history and stores it on the Agent as `_pending_turn` — the raw provider response plus the outputs of the tool calls that already finished, keyed by position. Resuming replays those instead of re-running them, at any delegation depth, and the whole thing round-trips through `CompanyState`, so a run can resume in a different process.

**Presets.** 15→27 skills, 10→22 personalities, 8→20 palettes, 11→20 org templates. Palette ramps were generated rather than hand-typed so every hex is valid by construction; a new check asserts each registry keeps at least 20 entries with unique, well-formed names.

### Behavioural notes worth knowing

- **OpenAI path:** tool arguments are decoded for the whole turn up front, so malformed provider JSON fails before any tool runs rather than part-way through.
- **Simultaneous pauses:** two delegates can ask for a human at the same instant. One pause is surfaced; the other is *not* marked finished, so on resume it runs and asks again. One extra question beats one unanswered question treated as answered.
- **Budget overshoot** remains possible and is documented where it lives: the gate checks before the model answers and the charge lands after, which was equally true before anything ran concurrently. Closing it properly needs a reservation, not a lock.
- **`plan_then_execute`** still catches `Exception` per step, so an unresolved escalation inside a plan step lands in `step.error` with `result.ok == False` rather than aborting the plan. Visible and honest, unlike the swallow Phase A fixed — but a policy call if you want it changed.
- **Tool-name inconsistency** (pre-existing): the registered delegation tool is lower-cased (`delegate_to_junior`) while `tool_call_log` records `delegate_to_<Employee.name>`. Harmless; annoying if you filter logs by tool name.

---

## 1. Headline: the two features share one refactor

The async work and the pause/resume work both bottom out in the same place: **the tool-dispatch step inside the four `process_*_response` methods**.

- Concurrency needs that loop to become `await asyncio.gather(...)`.
- Pause/resume needs that loop to be *checkpointable* — "which of this turn's tool calls already produced results, and which one were we inside when we paused."

Right now that loop is written out **four times** (`core.py:836` gemini, `:879` anthropic, `:912` openai/custom, `:938` ollama). Four copies is already the `REVIEW_MODES` situation this codebase treats as a bug class; making an async twin of each would make it eight.

**Do the extraction once, first.** Everything downstream gets cheaper, and it ships with zero behaviour change and zero API change.

---

## 2. Concurrency audit

### 2.1 What's confirmed

- All four processors run tool calls in a plain sequential `for` loop. A manager delegating three independent subtasks pays the sum, not the max. Confirmed.
- `generate` (`core.py:719`) and `generate_async` (`core.py:1038`) are already ~90% identical — same payload build, same iteration cap, same `process_response`. **Only the transport differs**: sync uses `_execute_with_thread` + the blocking `spinner()`; async uses `asyncio.to_thread` + `_animate_spinner`. Making async the one true implementation is a genuinely small diff here, not a rewrite.
- `Agent.chat_async` / `generate_async` are unreferenced anywhere else in the package. Nothing above `Agent` has an async counterpart.

### 2.2 The decision the handoff didn't surface: async only pays off if delegation is async

`delegate_to_X` is a plain sync callable. If tool dispatch becomes `gather` + `asyncio.to_thread`, then each parallel delegate runs `Employee.run` → `Agent.chat` → **its own `asyncio.run()`** inside a worker OS thread. That "works," but:

| | Option A: delegates stay sync, dispatched via `to_thread` | Option B: async all the way down |
|---|---|---|
| Diff size | small | medium |
| Concurrency model | N OS threads | one event loop, one thread |
| Correct lock | `threading.Lock` | `asyncio.Lock` |
| `activity_log` / `tool_call_log` appends | **racy** — need locks | safe (`list.append` between awaits is atomic) |
| Nested event loops | one per delegate | none |

**Option A reintroduces exactly the OS-thread races that choosing asyncio was meant to avoid, and makes `asyncio.Lock` the wrong primitive.** I recommend **Option B**: `Employee.run_async`, `delegate_tool` produces a coroutine function, `ToolRegistry` learns to await coroutine tools directly and only `to_thread`s genuinely-sync user tools.

### 2.3 Real races (Option B, one event loop)

| Site | Hazard | Fix |
|---|---|---|
| `Company._budget_gate` (`company.py:799`) | Reads `total_tokens_spent()` and `remaining()`, returns `None`, and the caller *then* awaits `chat()`. Two delegates can both pass a gate that only one budget's worth of room supports. | `asyncio.Lock` spanning gate → charge. Needs a real design decision: the "charge" only materialises after the model responds, so the lock has to cover a *reservation*, not just the read. |
| `_try_reallocate` (`company.py:845`) | Read-then-`grant_bonus` on the same ledger. | Same lock. |
| `_handle_budget_escalation` (`company.py:758`) | `self._emergency_budget_spent += grant` and `budget.total_token_budget = ... + ...` are read-modify-write. Currently safe (no await between), but `on_escalation` becoming awaitable puts an await in the middle. | Same lock, held across the whole handler. |
| `Company._log` / `record_tool_call` | **Not a race under Option B** — bare `list.append`, no await inside. Don't add locks here. Under Option A they *are* racy. | Leave alone (Option B). |
| `compact_logs()` at task end | Replaces the log lists while other coroutines may be mid-run. | Only run it when no task is in flight, or take the same lock. |

### 2.4 Two blocking calls that would freeze the event loop

Both are in the current `generate_async` path already:

1. **`_maybe_compact_history()` (`core.py:583`)**, called at the top of `generate_async` *synchronously*. In `mode="agent"` this makes a full blocking summarizer LLM round-trip. That stalls every other concurrently-running employee. Needs `await asyncio.to_thread(...)` or an async summarizer path.
2. **`on_escalation`** — a sync callable that may block on `input()`. Needs `inspect.iscoroutinefunction` detection: await if async, `to_thread` if sync.

### 2.5 `gather` semantics

`ToolRegistry.execute` already catches everything and returns a string, so `gather` will rarely see an exception — *except* for whatever control-flow types we carve out in §0. Use `return_exceptions=True` and convert anything unexpected into the same `"Tool Execution Failure (...)"` shape, so a sibling failure never silently cancels the others. `gather` preserves input order, so writing results back in order stays correct.

**Open question (§7 Q2):** should one failing/declined delegate cancel its siblings? My recommendation: no — collect all results — *except* the escalation control-flow types, which should propagate and cancel.

### 2.6 Nested-event-loop fallback

Detection: `asyncio.get_running_loop()` succeeding inside the sync wrapper. Recovery: run `asyncio.run(...)` on a fresh `threading.Thread` (legal — new thread, no loop). stdlib only, no `nest_asyncio`.

Warning text must state plainly that this **does not** unblock the caller's loop unless the caller writes `await` at the call site, and must point at `run_async()` as the actual fix. Draft:

> `Company.run()` was called from inside a running event loop. Recovered by running it on a helper thread, and the result is correct — but your event loop stayed blocked for the whole call, because only an `await` at your call site can yield control back to it. Use `await company.run_async(task)` instead.

Loud (`warnings.warn`, not `logger.debug`), and it must never silently become an auto-approve path for anything.

---

## 3. Persistence audit

### 3.1 What can and cannot be serialized

**Serializable** (a `save_state()` can capture all of this):
company name, `model_map` (minus keys), budget config + `_bonus_allocation`, `_emergency_iteration_remaining`, `_emergency_budget_spent`, `activity_log`, `tool_call_log`, teams; per employee: name/rank/`reports_to` name/importance/effort/specialty/skills/personality/`cost_weight`, and per Agent: `conversation.history`, `system_instruction`, `total_tokens_used`, `usage_log`, `max_tool_iterations`, `max_context_tokens`.

**Not serializable** — this is the constraint that shapes the whole API:
`on_escalation` (a callable), **every user-registered tool in `ToolRegistry.functions_maps`** (arbitrary Python callables), `model_policy`, `presets` (has runtime `.register()`), `cost_model`, the compaction policies, the `archive` handle, API keys.

**Therefore `resume()` cannot be `Company.load(path)`.** It has to be *rehydrate into a live company*: the caller reconstructs the Company the way they built it (code, or a `CompanySpec`), then calls `company.restore_state(snapshot)`, which re-applies histories and counters by matching employees **by name** and validating that the org chart and the registered tool-name set match what was saved. Any mismatch → loud error, never a silent partial restore. That is the same conservative posture as the rest of the library.

### 3.2 Snapshot format: dedicated document, not `RunArchive` replay

Recommend a versioned JSON `CompanyState` document mirroring the `CompanySpec.to_dict()/from_dict()` pattern (`builder.py:112`). Against replaying `RunArchive`:

- The archive is **off by default** (`archive=None`), so replay would only work for runs that opted in.
- It is a *report* log, not an event-sourcing log. `_bonus_allocation`, `_emergency_budget_spent`, and `_emergency_iteration_remaining` are never written to it in a form you could reconstruct exactly — `_log("emergency_budget_used", ...)` records the grant, not the invariant.
- JSONL-with-a-truncated-last-line is a virtue for an audit trail and a liability for a state snapshot.

Keep them separate and say so: `RunArchive` answers "what happened," `CompanyState` answers "where are we." Same event may appear in both; neither derives from the other.

### 3.3 Pause/resume: it's more tractable than the handoff feared

The handoff's worry — "Python cannot serialize a live paused call stack" — is true but, I think, over-scoped. You don't need the stack. You need, at each level of the delegation chain, two things:

1. the agent's `conversation.history` (already fully serializable, already carries the assistant's `tool_calls` message for the turn we're inside), **and**
2. *which tool calls in that turn already produced results.*

(2) is the only thing missing today, and it's missing precisely because tool dispatch is an inline `for` loop with no record of its own progress. Extract that loop (§1) and add a `pending_tool_calls` / `completed_results` checkpoint to it, and a pause becomes:

```
snapshot = {
  "version": 1,
  "company": {...},
  "employees": {name: {agent_state...}},
  "call_stack": [                       # outermost first
    {"employee": "Boss",   "tool_call_id": "c1",
     "completed": {"c0": "...result..."}, "pending": ["c1", "c2"]},
    {"employee": "Junior", "tool_call_id": None, "completed": {}, "pending": []}
  ],
  "paused_event": {EscalationEvent as dict},
  "resume_token": "..."
}
```

On resume: rebuild the live Company, `restore_state`, then walk the `call_stack` outermost-in, re-entering each level's tool loop with the completed results replayed from the snapshot (**not** re-executed — that's the whole point) and the pending ones resumed, starting with the one the human just decided.

**This works at any delegation depth** and does not require rewriting the run loop into a full state machine. It requires making one step — tool dispatch — checkpointable. That is the same extraction §1 already calls for.

Caveat worth naming: replaying completed results means side-effecting tools are not re-run, which is correct, but any tool whose result was *not* deterministic-and-recorded (streaming, a handle, an open file) can't be replayed. For v1 I'd document that tool results must be plain strings — which `_stringify_tool_output` already enforces.

### 3.4 API shape for a paused run

`Company.run()` returning `str | PausedRun` breaks every caller doing `result.strip()`. Two options:

- **(a)** `run()` raises `RunPaused(token)`. Non-breaking on the success path, but control flow via exception.
- **(b)** `run()` stays exactly as today (a `PENDING` return from `on_escalation` under plain `run()` is an error with a clear message), and a **new** `run_resumable()` returns `RunResult | PausedRun`. Fully opt-in, zero break, and the sync-simple surface (`quick_company(...)`, `company.run("do the thing")`) that you've said is a selling point stays untouched.

**Recommend (b).**

`EscalationDecision` needs a way to say "pending." Note the footgun: `approve=False` currently means *decline → raise*, so a `pending=True` field would have to be checked before `approve` at three call sites (`company.py:748`, `:773`, and the reserve paths). Cleaner: a distinct module-level sentinel, `ESCALATION_PENDING`, returned *in place of* an `EscalationDecision`. Impossible to confuse with a decline. And `default_on_escalation` keeps declining — "nobody's watching" must never become "pend forever."

---

## 4. Proposed phasing

Each phase is independently shippable and independently testable.

**Phase A — extraction + the §0 bug fix.** No async, no persistence, no API change.
- Normalize all four processors to `[{"id","name","args"}]` + a small write-back closure; one shared `_dispatch_tool_calls`.
- Add a control-flow exception carve-out in `ToolRegistry.execute` so `EscalationUnresolved` (and later the pause signal) propagates instead of stringifying.
- Tests: dispatch parity per provider; the §0 repro as a regression test.

**Phase B — async core.** `generate_async` becomes the implementation; `generate` = thin wrapper + nested-loop fallback + the §2.6 warning. `ToolRegistry` gains an async dispatch path (await coroutine tools, `to_thread` sync ones). `gather(return_exceptions=True)`. `_maybe_compact_history` off the loop.

**Phase C — async company.** `Employee.run_async`, async `delegate_tool`, `Company.run_async` / `run_structured_async`, `asyncio.Lock` on the budget gate, sync-or-async `on_escalation`. **This is where the actual latency win lands.** Async `plan_then_execute` / `stub_and_fill` are natural here too — `stub_and_fill`'s stub loop (`delegation.py:673`) is embarrassingly parallel; `plan_then_execute`'s step loop is *not* (each step feeds the next), which is worth stating rather than parallelizing by reflex.

**Phase D — snapshot.** `CompanyState`, `save_state()`, `restore_state()`. No pause yet. Useful on its own: survives a crash *between* tasks.

**Phase E — pause/resume.** `ESCALATION_PENDING`, `run_resumable()`, `PausedRun`, the `call_stack` checkpoint, `resume(token, decision)`.

Phases A/B/D are mechanical enough for a lighter model once the shape is agreed. C and E are the concurrency-safety and state-machine pieces — worth high effort.

---

## 5. Conventions I'll hold to

- stdlib only. No `nest_asyncio`, no `pytest`, no async test helper. Tests stay plain assert/print with a `FakeResponder`; async tests drive coroutines through `asyncio.run` in the test file itself.
- Every new public export added to `tests/test_import_paths.py` (`llmadapt.core` entry is at line 92).
- No sync/async duplication: async is the implementation, sync is a wrapper.
- No new automatic-recovery path becomes an auto-approve. The nested-loop fallback warns loudly and changes nothing about escalation.
- New fixed vocabularies (e.g. a `PauseState`) get the lowercase-alias treatment scoped to their own submodule, matching `role`/`mode`/`review`; nothing new re-exported bare from `llmadapt`.
- 433 checks is the floor; every phase adds to it.

---

## 6. Risks I'm watching

- **Budget reservation vs. charge.** The gate checks before spend and the charge lands after the model responds. Under concurrency, a lock around the *check* isn't enough; there has to be a reservation. This is the subtlest correctness issue in the whole plan.
- **Deadlock in the thread-hop fallback.** A sync `run()` called from inside a loop, which itself delegates and hits another sync entry point. Needs a re-entrancy guard.
- **Replay fidelity in Phase E.** Restoring "completed" tool results without re-running them is correct only if results are pure strings. Documented constraint, plus a check at pause time.
- **Silent scope creep in Phase C.** Making `Employee.run` async touches `Team.run`, both delegation strategies, and `gui.py`. I'd rather ship B, prove the win on a real three-way delegation, then do C.

---

## 7. The decisions this plan turned on

*(All settled — recorded in the table at the top. Kept here for the reasoning behind each.)*

1. **Option A vs Option B (§2.2)** — sync delegates via `to_thread`, or async all the way down? *Recommend B.*
2. **Sibling failure semantics (§2.5)** — does one failing/declined delegate cancel its siblings? *Recommend: no, except escalation control-flow types.*
3. **Paused-run API (§3.4)** — `run()` raises `RunPaused`, or a new opt-in `run_resumable()`? *Recommend `run_resumable()`.*
4. **Sync `generate()` as an `asyncio.run` wrapper?** This makes `_execute_with_thread` and the blocking `spinner()` dead code (neither is referenced anywhere outside `core.py`, and neither is in any test). Delete them, or keep them? *Recommend: wrapper, delete both, note it in the changelog.*
5. **Does the snapshot ever contain API keys?** *Recommend never — resume re-resolves from env via `env.resolve_api_key`, so a snapshot is safe to commit or ship to another machine.*
6. **How much to green-light now.** A alone (small, fixes a real bug, unblocks everything)? A+B? Or the whole A–E arc?

All six are answered and all five phases are shipped.

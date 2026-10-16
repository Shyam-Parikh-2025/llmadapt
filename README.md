# llmadapt

[![tests](https://github.com/Shyam-Parikh-2025/llmadapt/actions/workflows/tests.yml/badge.svg)](https://github.com/Shyam-Parikh-2025/llmadapt/actions/workflows/tests.yml)

A small, provider-agnostic toolkit for building with LLMs. One `Agent` talks to
**Anthropic, OpenAI, Gemini and Ollama** (plus any OpenAI-compatible endpoint),
calls your Python functions as tools, and comes with ready-made skills,
toolkits, a multi-agent "company" layer, a command-line tool and an MCP server.

**Zero dependencies** - everything, HTTP included, is the Python standard
library. Python 3.9+. Windows, macOS and Linux.

Version 0.4.0 - see [what's new](#whats-new-in-040).

## Install

```bash
pip install "git+https://github.com/Shyam-Parikh-2025/llmadapt.git"
```

or from a clone, editable:

```bash
git clone https://github.com/Shyam-Parikh-2025/llmadapt.git
cd llmadapt
pip install -e .
```

(PyPI currently has an old 0.1.0 under this name; `pip install llmadapt` will
get 0.4.0 once it is published there.)

Then check what is set up:

```bash
llmadapt doctor
```

It reports which API keys it can see (never their values), whether Ollama is
running and which models it has, your hardware, and which provider `llmadapt
ask` would use.

## Configure a provider

Pick at least one:

| provider    | how to enable it                                                     | default model       |
|-------------|----------------------------------------------------------------------|---------------------|
| `anthropic` | set `ANTHROPIC_API_KEY`                                              | `claude-sonnet-5-5` |
| `openai`    | set `OPENAI_API_KEY`                                                 | `gpt-5.4-mini`      |
| `gemini`    | set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)                           | `gemini-3.6-flash`  |
| `ollama`    | install [Ollama](https://ollama.com), then `ollama pull llama3.1:8b` | `llama3.1:8b`       |
| `custom`    | any OpenAI-compatible server: `Agent("openai", base_url=..., key_alias=...)` | -          |

Keys can live in your environment or in a `.env` file in your project (found
automatically, up to six folders up; the real environment always wins). Other
settings, all optional:

| variable | effect |
|---|---|
| `LLMADAPT_<PROVIDER>_MODEL` | change a provider's default model, e.g. `LLMADAPT_ANTHROPIC_MODEL=claude-haiku-5-5` |
| `LLMADAPT_PROVIDER` | which provider the CLI uses when you don't say |
| `LLMADAPT_VERBOSE=0` | turn off the step-by-step status lines |
| `OLLAMA_HOST` | where Ollama runs, if not `localhost:11434` |

Model names change often. Passing `model=` explicitly is always the most
reliable choice; the defaults above are a convenience.

## Quick start

```python
from llmadapt import Agent

agent = Agent("anthropic")                  # or "openai", "gemini", "ollama"
print(agent.chat("What's 12 * 7?"))
print(agent.chat("Now divide that by 4."))  # the conversation carries over
```

From a terminal:

```bash
llmadapt ask "Explain Python's GIL in two sentences"
git diff | llmadapt ask "Review this diff" --skills code-review
llmadapt chat --provider ollama
```

More in [`examples/`](examples/): a quick start, a code assistant over a
folder, structured JSON output, and a multi-agent team.

## Tools: your own functions

```python
from typing import Literal

def get_weather(city: str, unit: Literal["F", "C"] = "F") -> dict:
    """Look up the current weather for a city.

    Args:
        city: City name, e.g. "Paris".
        unit: Temperature unit.
    """
    return {"city": city, "temp": 72, "unit": unit}

agent.add_tool(get_weather)          # or decorate it with @agent.tool
print(agent.chat("What's the weather in NYC?"))
```

The JSON schema the model sees is generated from the function: the docstring
becomes the description, type hints become JSON types (`str`, `int`, `float`,
`bool`, `list[...]`, `dict`, `Optional[...]`, `Literal[...]` and `Enum` are
understood), and an `Args:` section (or `:param x:` lines) describes each
parameter. Return values are JSON-serialized for the model; plain strings pass
through. If a tool raises, the model sees the error and can try again instead
of the run crashing. Tools can be `async def`, and several tool calls in one
turn run concurrently.

## Ready-made toolkits

Common tools, already written, safe by default:

```python
from llmadapt import Agent, toolkit

agent = Agent("ollama", tools=toolkit("fs", "code", root="./my_project"))
print(agent.chat("Which modules make network calls?"))
```

| toolkit | tools | notes |
|---|---|---|
| `fs` | `read_file`, `list_files`, `search_files` | read-only; confined to `root` (symlinks included) |
| `fs-write` | `fs` + `write_file`, `edit_file` | opt-in; `edit_file` needs a unique exact match |
| `code` | `outline_file`, `stub_file` | a file's classes and functions without the bodies |
| `shell` | `run_command` | one program per call, from an allowlist; runs in `root` |
| `python` | `run_python` | runs a snippet in a fresh interpreter |
| `web` | `fetch_url` | http(s) only, HTML reduced to text |

`shell` and `python` run real processes on your machine. Their allowlist stops
mistakes, not a determined model, so only give them to a model you would let
type in your terminal. The individual factories (`filesystem_tools(root,
allow_write=True)`, `shell_tools(root, allowed_commands=[...])`, ...) take
more options.

## Skills and personalities

Skills are reusable instructions with hard constraints: 39 built in, including
`python`, `testing`, `code-review`, `debugging`, `security-review`, `sql`,
`agentic-coding`, `test-driven-development` and `technical-docs`. Personalities
set tone, for example `concise`, `mentor`, `skeptical` or `socratic`.

```python
agent = Agent("openai", skills=["python", "testing"], personality="concise")
```

`llmadapt list skills` (or `llmadapt.SKILLS.names()`) shows them all. Typos
are caught (`unknown skill 'pyhton' (did you mean 'python'?)`), and you can
register your own (see [Presets](#multi-agent-companies) below).

## Structured output

```python
bugs = agent.chat_json("Find the bugs in: def avg(xs): return sum(xs)/len(xs)-1",
                       schema_hint=[{"line": "int", "problem": "str"}])
```

This returns parsed JSON (a dict or list). Code fences and chatter around the
JSON are tolerated. If the reply doesn't parse, the model is shown the error
and asked once more. It works with every provider.

## Using llmadapt from an AI coding assistant

llmadapt is useful to assistants like Claude Code, Cursor or Copilot as well
as to people:

- **Read less.** `llmadapt outline big_file.py` lists every class and function
  with line numbers. `llmadapt stub file.py` shows signatures without bodies.
  `pytest 2>&1 | llmadapt compress --max-chars 4000` folds repeated lines and
  keeps the head and tail. None of these call an LLM.
- **Delegate.** Hand self-contained work (first drafts of tests or docstrings,
  summaries, bulk edits, a second-opinion review) to a local or cheaper model
  with `llmadapt ask ... --file x.py --skills testing`. To fan out,
  `llmadapt map "Summarize this file in one line" src/*.py --jsonl` runs one
  agent per file, several at a time. stdout is only the answer, `--usage`
  reports tokens and estimated cost on stderr, and the exit code says whether
  it worked.
- **As MCP tools.** `llmadapt mcp` is a Model Context Protocol server
  exposing `ask_model`, `review_code`, `outline_code`, `compress_file` and
  `list_skills`:

  ```bash
  claude mcp add llmadapt -- llmadapt mcp --provider ollama --root .
  ```

  Any other MCP client works with `{"command": "llmadapt", "args": ["mcp", "--provider", "ollama", "--root", "."]}`.
- **As a skill.** [`.claude/skills/llmadapt/SKILL.md`](.claude/skills/llmadapt/SKILL.md)
  teaches an assistant when and how to use all of the above. Claude Code
  picks it up inside this repo; copy it to `~/.claude/skills/llmadapt/` to
  have it everywhere.

## Reliability and control

```python
agent = Agent(
    "anthropic",
    request_timeout=120,          # seconds per HTTP request (default 600; None = wait forever)
    max_retries=3,                # retries 429/5xx/529/dropped connections, honouring Retry-After
    max_tool_iterations=10,       # tool-calling rounds before giving up (default 6)
    max_tokens=2000,              # cap on each reply
    extra_params={"temperature": 0.2},   # any extra request-body fields the provider accepts
    verbose=False,                # no status lines (or set LLMADAPT_VERBOSE=0)
    prompt_cache=True,            # Anthropic: cache system prompt + tools + history prefix
)
agent.switch_api("ollama", model="qwen2.5-coder:7b")   # same conversation, different model

agent.total_tokens_used          # real token counts reported by the provider
agent.cost_estimate()            # rough USD from the built-in price table (None if unknown, 0.0 if local)
```

`prompt_cache=True` matters most for agents with long system prompts (skills)
or many tool rounds. Every request after the first re-reads the cached prefix
at about a tenth of the normal input price. OpenAI and Gemini cache
automatically.

If the model keeps calling tools without ever answering, `chat()` raises a
`RuntimeError` instead of looping forever.

## Compressing tool output

Long tool results (file dumps, logs, search results) eat into the context
window fast, especially for smaller/local models. `CompressionPolicy` lets
an agent opt in to automatic truncation and dedup of oversized tool output
before it's added to conversation history:

```python
from llmadapt import Agent, CompressionPolicy

agent = Agent(
    provider="ollama",
    model="llama3.1:8b",
    compression_policy=CompressionPolicy(enabled=True, max_chars=1500, min_chars_to_bother=300),
)

# or change it later on an existing agent
agent.set_compression_policy(CompressionPolicy(enabled=True, max_tokens=800))
```

Compression is **off by default** — a plain `Agent()` with no policy behaves
exactly as before. When enabled, it:

- collapses repeated consecutive lines (logs/stack traces love to repeat one
  line dozens of times),
- truncates from the middle, snapped to line boundaries, keeping head and
  tail context,
- optionally accepts a `summarizer` callable (`(text, budget) -> str`, e.g. a
  closure around another agent) instead of plain truncation.

`CompressionPolicy` is a thin, stateless config wrapper around
`ContextCompressor`, which is also usable directly for other jobs:

```python
from llmadapt.compressor import ContextCompressor

# strip function/class bodies down to signatures + docstrings, for feeding
# code into a model without spending tokens on implementation details
stub = ContextCompressor.code_to_stub(source_code)

# water-fill one shared character budget across several tool outputs
trimmed = ContextCompressor.compress_batch(outputs, total_budget=4000)
```

## Tracking token usage

`Agent` can estimate how many tokens the current conversation would cost on
the next request, and how many are left before a budget you set is hit. This
uses the same no-external-tokenizer heuristic as `CompressionPolicy`
(`ContextCompressor.token_estimate`), so treat it as an estimate rather than
an exact provider count:

```python
agent = Agent(provider="anthropic", model="claude-sonnet-5-5",
               max_context_tokens=100_000)

agent.chat("Summarize this document...")

agent.tokens_used()   # -> estimated tokens system instruction + history + tools would cost
agent.tokens_left()   # -> max_context_tokens - tokens_used(), floored at 0
```

`max_context_tokens` is a separate concept from `max_tokens`: `max_tokens` is
the per-request *output* cap sent to the provider (Anthropic's `max_tokens`,
Gemini's `maxOutputTokens`, ...), while `max_context_tokens` is a budget you
track your whole conversation against. If you don't set `max_context_tokens`
(default `None`), `tokens_left()` returns `None` since there's no budget to
count down from. Change it later with `agent.set_max_context_tokens(200_000)`.

## Rank-based routing

`ModelRouter` maps an organizational "rank" (intern up through C-suite) to a
model choice and a matching compression policy, so a multi-agent setup can
pick both in one place instead of hardcoding them per agent:

```python
from llmadapt import Agent, ModelRouter, RoleRank

model = ModelRouter.allocate_model(
    rank=RoleRank.JUNIOR,
    policy={"cost_priority": "high"},
    model_map={"local_default": "ollama/llama3.1:8b", "frontier_default": "claude-sonnet-5-5"},
)
compression = ModelRouter.allocate_compression_policy(RoleRank.JUNIOR)

agent = Agent(provider="ollama", model="llama3.1:8b", compression_policy=compression)
```

`ModelRouter.check_hardware_safety(...)` also guards against routing a rank
to a local model binary too large for the current machine's RAM, falling
back to a configured alternative when it would exceed the allowed threshold.

## Hardware awareness (for local models)

`hardware.py` provides standard-library-only utilities for reasoning about
the machine an agent is running on, mainly aimed at running local model
binaries (e.g. via Ollama) safely alongside everything else on the box:

```python
from llmadapt import HardwareProfiler, ResourceQuota, MoELayerOffloader, LocalModelSingleton

specs = HardwareProfiler.inspect()
# {"os": "Windows", "cpu_cores": 16, "system_ram_gb": 32.0, "gpu_vram_gb": 12.0}

quota = ResourceQuota(mode="auto", ttl_seconds=120)
plan = MoELayerOffloader.calculate_offload(
    total_layers=32, num_experts=8, model_size_gb=14.0, quota=quota,
)
# how many layers fit on GPU vs. spill to system RAM

# Ensures only one local model binary is loaded at a time, and unloads it
# automatically after `quota.ttl_seconds` of inactivity to free VRAM/RAM.
process = LocalModelSingleton.acquire_model("llama3.1:8b", launch_func=my_launcher, quota=quota)
```

## Local models

Pass `is_local=True` when constructing an `Agent` pointed at a local
provider (e.g. Ollama) to flag it as a local-model agent for the rest of
your routing/hardware logic:

```python
agent = Agent(provider="ollama", model="llama3.1:8b", is_local=True)
```

## Multi-agent companies

Beyond a single `Agent`, llmadapt can run a hierarchy of them — a `Company` of
`Employee`s with reporting lines, delegation, budgets and oversight. Delegation
is just the ordinary tool-calling loop: hiring someone with `reports_to=` puts a
`delegate_to_<name>` tool on their manager's agent.

**The one-call path.** For a ready-made org shape, `quick_company()` is a
template name in and a running-ready `Company` out — no `model_map` or
`on_escalation` to spell out for a first try:

```python
from llmadapt import quick_company

company = quick_company("small-coding-team")
print(company.run("Write a CSV parser"))
```

This is sugar over `Company(...)` + `Company.build_from_template(...)` below,
and keeps the same safe defaults rather than a framework's usual "just trust
me": no `model_map` still means every rank stays on local Ollama (a company
you didn't configure cannot spend money), no `on_escalation` still means
decline, and no `total_token_budget` still means unlimited. Pass any of the
three explicitly — plus anything `build_from_template()` accepts, e.g.
`review_mode="append"` — the moment you're past a first try:

```python
company = quick_company(
    "small-coding-team", name="Acme", size="medium",
    model_map={"C_SUITE": {"provider": "anthropic", "model": "claude-sonnet-5-5"}},
    total_token_budget=200_000,
)
```

**Building an org by hand** gives full control over ranks, reporting lines and
per-employee overrides — `hire()` instead of a template:

```python
from llmadapt import Company, EscalationDecision
from llmadapt.router import role

def on_escalation(event):
    print(f"{event.employee_name} needs help: {event.message}")
    return EscalationDecision(approve=True, extra_token_budget=5000)

company = Company(
    name="Acme",
    model_map={
        role.C_SUITE: {"provider": "anthropic", "model": "claude-sonnet-5-5"},
        role.JUNIOR:  {"provider": "ollama",    "model": "llama3.1:8b"},
    },
    on_escalation=on_escalation,
    total_token_budget=200_000,
)

boss = company.hire("Ada", role.C_SUITE)
dev  = company.hire("Grace", role.JUNIOR, reports_to=boss, skills=["python"])

print(company.run("Write a CSV parser"))
print(company.render_org_chart(fmt="ascii"))
print(company.budget_report())
```

`role` is `RoleRank` under a second, lowercase name — the exact same object,
not a copy — so a rank reads as `role.SENIOR` instead of `RoleRank.SENIOR`.
Same idea as `logging.INFO` or `socket.AF_INET` in the standard library: a
short, lowercase, module-like handle in front of the actual constants. An
earlier version of this bound `C_SUITE`/`MANAGER`/`SENIOR`/`JUNIOR`/`INTERN`/
`VOLUNTEER` as bare top-level names instead — shorter still, but four of those
are common enough English words that `from llmadapt import *` could shadow
one you already had. `role` cuts that risk down to a single name, and — since
it's imported from `llmadapt.router` rather than the bare `from llmadapt
import ...` everything else here uses — it only shows up when you deliberately
reach for it. `RoleRank.SENIOR` remains available at the top level, unchanged,
as the zero-ambiguity spelling that needs no second import at all. (`role` is
also a name this codebase's own templates use constantly as a loop variable —
`for role in roles: ...` — so a local `role = ...` will shadow the import
within that function, same as any other Python name would.)

`PolicyMode`/`mode` (`from llmadapt.company import mode`) and `ReviewMode`/
`review` (same import) are the same pattern applied to the two other small
fixed vocabularies below — `mode.AUTO`/`.LOCAL`/`.API` for `ModelPolicy`
routing, `review.CRITIQUE`/`.APPEND`/`.OFF` for `Team.review_mode`. All three
(`role`, `mode`, `review`) are deliberately plain classes of string constants,
not real `Enum`s: every value already has to behave like an ordinary string
everywhere it travels (dict keys, JSON, f-string-embedded messages), and a
real Enum only gets that for free if you remember to override `__str__` -
`f"{x}"` on an un-fixed `class X(str, Enum)` prints `"X.MEMBER"`, not the
value, which is exactly the kind of thing that goes unnoticed until it shows
up wrong in a log or a system instruction. See `router.py`'s comment next to
`role`'s definition for the full reasoning, confirmed against a live
interpreter rather than assumed.

`on_escalation` is optional on `Company(...)` too — leaving it out falls back
to declining every escalation (`llmadapt.default_on_escalation` /
`always_decline`, the same function), so a throwaway example doesn't need a
callback just to run. For local-only prototyping where auto-approving is
genuinely harmless (no API key, nothing to spend), `always_approve(...)` is a
ready-made handler that approves everything with fixed grants:

```python
from llmadapt import Company, always_approve

company = Company(name="Sandbox", model_map=local_only_map,
                  on_escalation=always_approve(extra_token_budget=5000))
```

Reach for a real callback (like the one above) the moment there's an actual
human, or actual spend, on the other end.

**Budgets.** `total_token_budget` is a hard company-wide ceiling; nothing
crosses it without `on_escalation` approving. Below it, each rank gets a
percentage share scaled by a per-employee `importance`, and an employee who
runs dry asks up the reporting chain before escalating. Emergency reserves
default to `0`, meaning "always ask a human first". Note this is a *pre-flight*
gate, not a mid-generation cutoff — llmadapt can't interrupt a request already
in flight.

**Local vs. API routing.** Attach a `ModelPolicy` and an effort hint decides
where work runs: `cheap` keeps it on any local model that fits, `balanced`
takes local only when it fits entirely in VRAM, `effort` goes to the API. The
API model table is *your* config — the shipped defaults are hand-entered and
will go stale.

```python
from llmadapt import ModelPolicy, ApiModelCatalog, ApiModelSpec
from llmadapt.router import role
from llmadapt.company import mode

policy = ModelPolicy(api_catalog=ApiModelCatalog([
    ApiModelSpec(name="gpt-5.4-mini", provider="openai",
                 cost_per_1k_input=0.00075, cost_per_1k_output=0.0045,
                 capability=0.74, specialties=("code",)),
], include_defaults=False))

company = Company(..., model_policy=policy)
company.hire("Ada", role.SENIOR, mode=mode.AUTO, effort="needs effort")
company.run("something hard", effort="needs effort")   # per-task, too
```

`effort`/`specialty` stay plain strings on purpose, unlike `mode` — they
already accept natural-language aliases (`"needs effort"`, `"keep it
cheap"`, `specialty="code"` matched freely against whatever an
`ApiModelCatalog` tags), which a fixed set of named constants would fight
rather than help. `policy.EFFORT_CHEAP`/`EFFORT_BALANCED`/`EFFORT_EFFORT` (and
`EFFORT_LEVELS`, the tuple of all three) already exist as the one canonical
source those aliases resolve to, if you want the constant instead of the
string — nothing new was needed there.

**Presets.** Skills, personalities, org templates and org-chart palettes all
come from one registry, so they work the same way and can all be extended.
The built-in catalog ships 39 skills, 22 personalities, 20 palettes and 20 org
templates — enough that picking one is usually faster than writing one:

```python
from llmadapt import SKILLS, ORG_TEMPLATES, Skill

SKILLS.register(Skill(name="sql", instructions="Write portable ANSI SQL.",
                      constraints=("Never use vendor-specific syntax.",)))

team = company.build_from_template("small-coding-team", size="medium")
print(company.render_org_chart(fmt="svg", palette="ocean"))
```

A `Team` with a `reviewer` actually reviews: the lead drafts, the reviewer
signs off or sends it back (bounded by `max_review_rounds`, default 1), and an
objection that survives the last round ships appended to the work rather than
being dropped. `review_mode` takes `"critique"` (default) / `"append"` /
`"off"` as plain strings, or the same values as `review.CRITIQUE` /
`review.APPEND` / `review.OFF` (`from llmadapt.company import review`) - the
`mode`/`role` pattern applied a third time, for the one other small fixed
vocabulary in this library that wasn't already a plain string with nowhere
else it needed to be spelled out consistently.

**Task decomposition.** `run_structured()` offers two strategies beyond plain
delegation:

```python
result = company.run_structured("build a JSON toolkit", strategy="stub_fill")
print(result.code, result.unfilled)      # senior tier wrote the signatures,
                                          # cheap tiers wrote the bodies
result = company.run_structured("write a report", strategy="plan")
print(result.answer, [s.instruction for s in result.steps])
```

Neither raises on a model-quality failure — an unfilled stub or a failed step
comes back on the result object. Assembled code is checked with `compile()` for
**syntax only**; nothing runs or tests it.

**Log compaction.** `Company(log_compaction=LogCompactionPolicy(mode="algorithmic"))`
collapses old activity and tool-call entries after each task, keeping counts
and dropping specifics. Escalations, budget decisions and errors are never
compacted away.

**Not sure what to pass?** `company_help()` returns the whole picture in one
call — what the pieces are, every valid template/skill/personality/palette name
*with a description of what it does*, the safety defaults written out, how to
run work afterwards, and worked examples. It builds nothing and spends nothing.

```python
from llmadapt import company_help

print(company_help(as_text=True))          # markdown, for a human
context = company_help()                   # a dict, for handing to a model
```

Same thing via `set_company_up(mode="help")`, and a model calling the setup
tool can get it with `set_up_company(help_only=True)` before it commits to any
arguments.

**Building a company.** Either from code, from an AI tool call, or by hand:

```python
from llmadapt import register_company_builder, make_company_via_gui

# for an AI caller - the schema is generated from the function's own hints
register_company_builder(my_agent)

# for a human - opens an interactive node-graph editor in the browser and
# blocks until "Build" is clicked
result = make_company_via_gui(model_map=my_model_map)
company = result.company
```

`make_company_via_gui(**kwargs)` is `set_company_up(mode="gui", **kwargs)`
under a name that says what it does - both spellings work and take the same
keywords (`model_map` / `on_escalation` / `model_policy` / `presets` /
`log_compaction`, plus `launch_gui`'s own `port` / `save_path` /
`open_browser` / `block` / `timeout`). Every mode builds the org and **runs
nothing** — starting work stays your explicit step. With no `model_map`,
everything defaults to local Ollama so an unconfigured build cannot spend
money; with no `on_escalation`, the default handler declines.

**Running work in parallel.** When a manager's turn asks for several
delegations at once, they run concurrently — three independent subtasks cost
the slowest one, not the sum. Nothing about the sync API changes to get this:
`company.run(...)` still takes a string and returns a string, with no event
loop in sight.

```python
company.run("split this three ways")          # sync, as always
await company.run_async("split this three ways")   # from async code
```

Every synchronous entry point (`Agent.chat`, `Employee.run`, `Company.run`,
`run_structured`, both delegation strategies) is a thin wrapper over its
`_async` twin, so there is one implementation of each rather than two that
drift. Call a sync method from inside a running event loop and it still
returns the right answer, on a helper thread, with a loud warning explaining
that your loop stayed blocked anyway and that `await company.run_async(...)`
is the actual fix.

Tools may be `async def` (awaited directly) or ordinary functions (run off the
loop so they cannot stall it) — you do not have to pick, and existing tools
keep working untouched.

How many run at once is capped, and the cap is a number rather than an
accident: `Agent.max_parallel_tools` defaults to 8. A model can ask for twenty
tools in one turn, and with delegation wired as an ordinary tool that means
twenty employees starting at once. Set it higher for I/O-bound work, or to 1
for strictly sequential tool calls with the async machinery still in place.

```python
agent.set_max_parallel_tools(4)
```

"Unlimited" is not really an option worth taking: synchronous tools queue on
asyncio's default thread pool, which caps at `min(32, cpu_count + 4)`, so
removing this limit just borrows one that varies by machine.

**Saving and resuming.** `save_state()` captures where a company *is* —
conversations, budget ledger, counters, logs — as plain JSON. It never
contains an API key, so it is safe to commit or move between machines.

```python
company.save_state(path="run.json")

# in another process: build the company the same way you built it the first
# time, then restore onto it
company = build_my_company()
company.restore_state("run.json")
```

A snapshot cannot rebuild a company on its own — a company holds callables (the
escalation handler, every registered tool) that no data file can carry. Every
structural fact that *can* be checked is: roster, ranks, reporting lines,
registered tool names. A mismatch raises and lists all of them rather than
restoring the parts that happen to line up.

This is a different thing from `RunArchive`, and does not replace it: the
archive says what happened, the snapshot says where things stand.

**Escalations that wait for a human.** An `on_escalation` handler has a third
answer besides approve and decline: `ESCALATION_PENDING` means "hold on,
someone is looking at it". The run stops where it is and hands back everything
needed to pick it up later — possibly hours later, in a different process.

```python
from llmadapt import ESCALATION_PENDING, EscalationDecision, PausedRun

def on_escalation(event):
    queue_for_review(event)
    return ESCALATION_PENDING

result = company.run_resumable("do the work")
if isinstance(result, PausedRun):
    company.save_state(path="paused.json", paused=result)
    ...                                    # hours later, maybe elsewhere
    answer = company.resume(result, EscalationDecision(approve=True,
                                                       extra_token_budget=50_000))
```

Work that already finished is not done again. If a manager delegated three ways
and two reports answered before the third asked for a human, resuming replays
those two answers rather than re-running them — they already had whatever
effects they had. This holds at any delegation depth: each agent in the chain
keeps the turn it was in the middle of.

`run_resumable()` is separate from `run()` on purpose. `run()` returning
`str | PausedRun` would put an isinstance check in front of every caller who is
never going to pause anything, so pausing is opt-in — and a `run()` that does
receive a pause raises with a message pointing here, rather than swallowing it.
Declining at resume time raises exactly as declining in the moment would:
waiting and then saying no is still saying no.

## Supported providers

| provider    | env var for API key                     | notes                          |
|-------------|------------------------------------------|--------------------------------|
| `anthropic` | `ANTHROPIC_API_KEY`                      | Messages API                   |
| `openai`    | `OPENAI_API_KEY`                         | Chat Completions API           |
| `gemini`    | `GEMINI_API_KEY` or `GOOGLE_API_KEY`     | generateContent endpoint       |
| `ollama`    | *(none - local)*                         | `OLLAMA_HOST`, default `localhost:11434` |
| `custom`    | *(none - pass directly)*                 | requires `base_url` + a `custom_format_func` you supply to `chat()` |

OpenAI-compatible services (OpenRouter, Groq, Together, vLLM, LM Studio, ...)
work as `provider="openai"` with a `base_url`. Give each one its own key with
`key_alias="openrouter"` (which reads `OPENROUTER_API_KEY`) or
`key_env="MY_VAR"`.

Some newer OpenAI models (the GPT-6 family) only support tool calling through
the Responses API, which llmadapt does not speak yet. Use them for plain chat,
or use `gpt-5.4-mini` / `gpt-5-mini` when you need tools.

## What's new in 0.4.0

- **Ready to use:** `llmadapt doctor`, current default models (overridable
  with environment variables), `GOOGLE_API_KEY` and `OLLAMA_HOST` support, and
  `switch_api()`, which this README had always documented but which didn't
  exist until now.
- **Sturdier:** HTTP timeouts and automatic retries for rate limits and
  overloads. Status output can't crash a run on a Windows pipe, the spinner
  only shows on a real terminal, and Anthropic thinking blocks survive a model
  switch.
- **Ready-made toolkits:** `fs`, `fs-write`, `code`, `shell`, `python`, `web`.
- **12 new coding skills** (39 total), plus `skills=`, `personality=` and
  `tools=` on a plain `Agent`.
- **Better tool schemas** from type hints and docstrings, and `chat_json()`.
- **CLI and MCP server** for use from terminals and AI coding assistants,
  including `llmadapt map` to fan one prompt out over many files.
- **Cost control:** opt-in Anthropic prompt caching, `cost_estimate()`, a
  refreshed price table, and a larger Anthropic `max_tokens` default (16000),
  because thinking counts toward it. A reply that runs out of tokens before
  answering now raises an error instead of returning an empty string.
- **Tests:** two suites that were broken on Windows are fixed, the full suite
  runs on Python 3.9-3.13, and CI covers Linux, macOS and Windows.

## Running tests

```bash
python tests/run_all.py          # the whole suite (sets PYTHONPATH=src itself)
PYTHONPATH=src python tests/test_agent.py   # or one file
```

Tests run entirely offline against scripted fake responses, with no API key
or network needed. They use plain `assert` + `print`, with no test framework.

`tests/browser_smoke.py` is deliberately outside the suite. It drives the
GUI through a real headless browser and needs `playwright`, which llmadapt
does not depend on. Run it by hand after changing `gui_assets.py`.

## Contributing

See [CLAUDE.md](CLAUDE.md) for the conventions (zero dependencies, test style,
Python 3.9 floor). The design history of the multi-agent layer is in
[`docs/dev/company-roadmap.md`](docs/dev/company-roadmap.md).

## License

MIT

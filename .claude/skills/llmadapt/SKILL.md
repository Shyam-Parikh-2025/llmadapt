---
name: llmadapt
description: Use the llmadapt CLI/library to delegate sub-tasks to local Ollama or cheaper API models, get a second-opinion code review from a different model, or read big files and logs compactly (outline, stub, compress). Use when a task is self-contained and bulky, when an independent review would help, or when output is too large to read in full.
---

# llmadapt for coding assistants

llmadapt is a zero-dependency Python library + CLI that talks to Anthropic,
OpenAI, Gemini and Ollama through one interface. For an AI coding assistant it
is useful in three ways.

## 1. Read less (no LLM call, always safe, instant)

| Need | Command |
|---|---|
| Map of a big source file (classes/functions + line numbers + 1-line docs) | `llmadapt outline path/to/file.py [more files...]` |
| Python file without bodies (signatures, types, docstrings) | `llmadapt stub path/to/file.py` |
| Huge log / test output squeezed to a budget (repeats collapsed, head+tail kept) | `pytest 2>&1 \| llmadapt compress --max-chars 4000` |

Use `outline` before reading a file over ~300 lines, then read only the line
ranges you need. Use `compress` on any command whose output may exceed a few
thousand characters.

## 2. Delegate (costs the user's tokens or local compute)

```bash
llmadapt doctor                      # first: which providers/keys/models are available
llmadapt ask "<complete instructions>" --file a.py --file b.py --skills python,testing
git diff | llmadapt ask "Review this diff for bugs" --skills code-review
llmadapt ask "Return the public functions as JSON: [{name, args}]" --file mod.py --json
llmadapt map "Write a Google-style docstring for each public function" src/*.py --jsonl --usage
```

- stdout is only the answer; status/errors go to stderr. Exit codes: 0 ok,
  1 provider/network failure, 2 usage error.
- `map` runs one fresh agent per file (`-c` sets concurrency, default 4) and
  prints results in input order; `--jsonl` gives `{"file", "answer"|"error"}`.
- `--usage` prints tokens and an estimated cost to stderr - mention it to the
  user when delegating to a paid API.
- Provider is auto-picked (running Ollama first, then any API key found). Pin
  it with `--provider ollama|anthropic|openai|gemini` and `--model NAME`.
- `--skills` takes built-in skill names; see `llmadapt list skills` (e.g.
  `python`, `testing`, `code-review`, `debugging`, `technical-docs`,
  `agentic-coding`, `test-driven-development`).
- `--tools fs,code --root DIR` lets the delegated model read files itself
  (read-only, confined to DIR). `fs-write`, `shell` and `python` give it real
  write/execute power - only with the user's go-ahead.

Good things to delegate: first drafts of docstrings/tests/boilerplate,
summarising long files or logs, bulk mechanical rewrites, classification, a
second-opinion review. Keep for yourself: anything needing the whole
conversation's context, architecture decisions, and final verification.

The delegated model is usually weaker than you. Give it complete,
self-contained instructions and **always check its output** (run the tests,
read the diff) before relying on it. Tell the user when you used it and with
which provider/model, since it may cost them money.

## 3. As MCP tools (no shell needed)

If the user wants it wired into the assistant directly:

```bash
claude mcp add llmadapt -- llmadapt mcp --provider ollama --root .
```

This exposes `ask_model`, `review_code`, `outline_code`, `compress_file` and
`list_skills`. Add `--tools fs,code` to let `ask_model` read files under the root.

## In Python

```python
from llmadapt import Agent, toolkit
agent = Agent("ollama", skills=["python", "testing"], tools=toolkit("fs", "code", root="."),
              verbose=False)
print(agent.chat("Write pytest tests for utils/slugify.py"))
data = agent.chat_json("List the TODOs in utils/ as [{file, line, text}]")
```

## Setup / troubleshooting

- Install: `pip install llmadapt` (or `pip install -e .` from a clone).
- Keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (or
  `GOOGLE_API_KEY`), from the environment or a `.env` file in the project.
- Defaults: `LLMADAPT_PROVIDER`, `LLMADAPT_<PROVIDER>_MODEL`, `OLLAMA_HOST`.
- `llmadapt doctor` explains what is missing. Never print or log key values.

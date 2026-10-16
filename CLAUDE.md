# Working on llmadapt

Notes for AI coding assistants (and humans) changing this repository.

## Ground rules

- **Zero third-party dependencies** in `src/`. Standard library only, including
  the GUI (vanilla JS, no CDN) and the MCP server (hand-rolled JSON-RPC).
  Optional packages (numpy, torch) may be used only opportunistically with a
  fallback - see `benchmark.py`. Grep for stray imports before finishing.
- **Python 3.9+.** No `match`, no `X | Y` in runtime-evaluated annotations, no
  `zip(strict=)`, no `dataclass(slots=/kw_only=)`.
- **Run the whole suite before calling anything done:**
  `python tests/run_all.py` (it sets `PYTHONPATH=src` itself). It is offline:
  providers are faked by assigning `agent._send_request = FakeResponder([...])`.
- **Test style:** plain `assert` + `print("PASS: ...")`, no pytest. Every new
  module gets `tests/test_<module>.py` in that style. The runner counts
  `PASS` lines, so a file that exits early still "passes" - keep imports of
  modules with side effects guarded (see `__main__.py`).
- **stdout discipline:** the CLI prints only answers on stdout; the MCP server
  owns stdout for JSON-RPC. Status output goes through `Agent.status_stream`.
- **Never log or print an API key value** - only where it came from
  (`env.resolve_api_key` returns the source for exactly this reason).
- **Sync wraps async.** Every sync entry point (`Agent.chat`, `Company.run`,
  ...) is a thin wrapper over its `_async` twin; change the async one.

## Map

- `core.py` - `Agent`, `Conversation`, `ToolRegistry` (schema generation,
  tool dispatch, HTTP with retries). Start here.
- `toolkits.py` - ready-made tools (fs, fs-write, code, shell, python, web).
- `presets/` - skills, personalities, palettes, org templates (one registry
  class, four instances). New built-in skills go in `presets/skills.py`.
- `cli.py`, `mcp_server.py`, `__main__.py` - the `llmadapt` command.
- `company/`, `budget.py`, `policy.py`, `delegation.py`, `compaction.py`,
  `builder.py`, `gui*.py`, `state.py`, `archive.py` - the multi-agent layer.
- `docs/dev/` - design history: `company-roadmap.md` is the record of *why*
  things are shaped the way they are. Read the relevant phase before
  reworking a subsystem.

## Releasing

Version lives only in `src/llmadapt/__init__.py` (`__version__`). Then:

```bash
python -m pip install build twine
python -m build && python -m twine check dist/*
python -m twine upload dist/*
```

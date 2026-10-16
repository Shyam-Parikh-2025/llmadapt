"""cli.py - the `llmadapt` command.

    llmadapt doctor                       check keys, .env, Ollama and hardware
    llmadapt ask "question" [options]     one-shot answer from any provider
    llmadapt chat [options]               interactive conversation
    llmadapt outline FILE...              classes/functions with line numbers
    llmadapt stub FILE                    a Python file without its bodies
    llmadapt compress [FILE]              shrink a log/tool output (stdin ok)
    llmadapt list skills|personalities|templates|toolkits|palettes
    llmadapt mcp [options]                run as an MCP server over stdio

`ask` reads piped stdin as extra context, so it composes in a shell:

    git diff | llmadapt ask "Review this diff" --skills code-review
    pytest 2>&1 | llmadapt compress --max-chars 4000

Answers go to stdout; status and errors go to stderr, so a calling program can
capture just the answer. Also reachable as `python -m llmadapt`.
"""

import argparse
import json
import os
import platform
import sys
import urllib.request
from typing import List, Optional, Sequence

from . import __version__

API_PROVIDERS = ("anthropic", "openai", "gemini")
ALL_PROVIDERS = API_PROVIDERS + ("ollama", "custom")


def _err(*parts: object) -> None:
    print(*parts, file=sys.stderr, flush=True)


def _out(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)


# ---------------------------------------------------------------------------
# Provider discovery
# ---------------------------------------------------------------------------


def ollama_status(timeout: float = 1.0) -> dict:
    """{"reachable": bool, "url": str, "models": [...], "error": str|None}."""
    from .core import ollama_base_url

    url = ollama_base_url()
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name", "?") for m in data.get("models", [])]
        return {"reachable": True, "url": url, "models": models, "error": None}
    except Exception as e:  # noqa: BLE001 - any failure just means "not reachable"
        return {"reachable": False, "url": url, "models": [], "error": f"{e.__class__.__name__}: {e}"}


def pick_provider(explicit: Optional[str] = None) -> tuple:
    """(provider, why). Order: --provider, LLMADAPT_PROVIDER, a running
    Ollama (free and local), then the first API provider with a key, then
    Ollama anyway (so the error message is about Ollama not running)."""
    from .env import env_value, resolve_api_key

    if explicit:
        return explicit.lower(), "--provider"
    configured = env_value("LLMADAPT_PROVIDER")
    if configured:
        return configured.lower(), "LLMADAPT_PROVIDER"
    if ollama_status(timeout=0.5)["reachable"]:
        return "ollama", "Ollama is running locally"
    for provider in API_PROVIDERS:
        key, _ = resolve_api_key(provider=provider)
        if key:
            return provider, f"an API key for {provider} was found"
    return "ollama", "nothing else configured"


def _pick_ollama_model(requested: Optional[str]) -> Optional[str]:
    """If no model was named and the default Ollama model is not installed,
    use the first installed one rather than failing with a 404."""
    if requested:
        return requested
    from .core import Agent

    default = Agent.default_model_for("ollama")
    status = ollama_status(timeout=0.5)
    installed = status["models"]
    if not installed:
        return default
    if default in installed or f"{default}:latest" in installed:
        return default
    return installed[0]


# ---------------------------------------------------------------------------
# Agent construction shared by ask/chat/mcp
# ---------------------------------------------------------------------------


def _split(values: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    for value in values or ():
        out.extend(v.strip() for v in value.split(",") if v.strip())
    return out


def build_agent(args: argparse.Namespace, quiet_default: bool = True):
    from .core import Agent
    from .toolkits import toolkit

    provider, why = pick_provider(getattr(args, "provider", None))
    model = getattr(args, "model", None)
    if provider == "ollama":
        model = _pick_ollama_model(model)
    tools = toolkit(*_split(args.tools), root=args.root) if getattr(args, "tools", None) else []
    extra = {}
    if getattr(args, "temperature", None) is not None:
        extra["temperature"] = args.temperature
    agent = Agent(
        provider=provider,
        model=model,
        base_url=getattr(args, "base_url", None),
        system_instruction=getattr(args, "system", None) or "",
        skills=_split(getattr(args, "skills", None)) or None,
        personality=getattr(args, "personality", None),
        tools=tools,
        max_tokens=getattr(args, "max_tokens", None),
        max_tool_iterations=getattr(args, "max_steps", None) or 12,
        verbose=bool(getattr(args, "verbose", False)) if quiet_default else None,
        extra_params=extra or None,
    )
    agent.status_stream = sys.stderr  # stdout is for the answer only
    return agent, why


def _add_agent_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("-p", "--provider", choices=ALL_PROVIDERS,
                   help="anthropic | openai | gemini | ollama | custom (default: auto - see `llmadapt doctor`)")
    p.add_argument("-m", "--model", help="model name (default: the provider's default, or LLMADAPT_<PROVIDER>_MODEL)")
    p.add_argument("--base-url", help="custom endpoint, e.g. an OpenAI-compatible server")
    p.add_argument("-s", "--system", help="system instruction")
    p.add_argument("--skills", action="append",
                   help="built-in skills, comma-separated (see `llmadapt list skills`)")
    p.add_argument("--personality", help="a built-in personality (see `llmadapt list personalities`)")
    p.add_argument("-t", "--tools", action="append",
                   help="toolkits the model may use, comma-separated: fs, fs-write, code, shell, python, web")
    p.add_argument("--root", default=".", help="folder the file/shell toolkits are confined to (default: .)")
    p.add_argument("--max-tokens", type=int, help="cap on the length of each reply")
    p.add_argument("--max-steps", type=int, help="max tool-calling rounds per answer (default 12)")
    p.add_argument("--temperature", type=float, help="sampling temperature (not every model accepts it)")
    p.add_argument("-v", "--verbose", action="store_true", help="show each step (on stderr, so stdout stays just the answer)")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    from .core import Agent
    from .env import find_env_file, resolve_api_key
    from .hardware import HardwareProfiler

    _out(f"llmadapt {__version__}  (Python {platform.python_version()}, {platform.system()} {platform.machine()})")
    env_file = find_env_file()
    _out(f".env file: {env_file or 'none found (searched this folder and up to 6 parents)'}")
    _out("")
    _out("API keys (values are never shown):")
    for provider in API_PROVIDERS:
        key, source = resolve_api_key(provider=provider)
        mark = "OK " if key else "-- "
        _out(f"  {mark} {provider:<10} {source}")
    _out("")
    status = ollama_status()
    if status["reachable"]:
        models = ", ".join(status["models"][:12]) or "(no models pulled - try `ollama pull llama3.1:8b`)"
        _out(f"Ollama: running at {status['url']}")
        _out(f"  models: {models}")
    else:
        _out(f"Ollama: not reachable at {status['url']} ({status['error']})")
        _out("  Install from https://ollama.com, or ignore this if you only use API providers.")
    _out("")
    try:
        hw = HardwareProfiler.inspect()
        _out(f"Hardware: {hw['cpu_cores']} CPU cores, {hw['system_ram_gb']:.1f} GB RAM, "
             f"{hw['gpu_vram_gb']:.1f} GB GPU VRAM")
    except Exception as e:  # noqa: BLE001
        _out(f"Hardware: could not inspect ({e})")
    _out("")
    provider, why = pick_provider(None)
    model = _pick_ollama_model(None) if provider == "ollama" else Agent.default_model_for(provider)
    _out(f"`llmadapt ask` would use: {provider} / {model}  ({why})")
    _out("Override with --provider/--model, or set LLMADAPT_PROVIDER and LLMADAPT_<PROVIDER>_MODEL.")
    return 0


def _gather_context(args: argparse.Namespace) -> str:
    parts: List[str] = []
    for path in args.file or ():
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            parts.append(f"--- {path} ---\n{f.read()}")
    if not sys.stdin.isatty() and not getattr(args, "no_stdin", False):
        piped = sys.stdin.read()
        if piped.strip():
            parts.append(f"--- stdin ---\n{piped}")
    return "\n\n".join(parts)


def cmd_ask(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt).strip()
    context = _gather_context(args)
    if not prompt and not context:
        _err("llmadapt ask: give a prompt, e.g.  llmadapt ask \"explain this\" --file x.py")
        return 2
    if context:
        prompt = f"{prompt}\n\n{context}" if prompt else context
    agent, why = build_agent(args)
    if args.verbose:
        _err(f"[llmadapt] {agent.provider} / {agent.model} ({why})")
    try:
        if args.json:
            result = agent.chat_json(prompt)
            _out(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _out(agent.chat(prompt))
    except Exception as e:  # noqa: BLE001 - CLI boundary: report, don't traceback
        _err(f"llmadapt ask failed ({agent.provider} / {agent.model}): {str(e).strip()}")
        if agent.provider == "ollama":
            _err("Is Ollama running? Check with `llmadapt doctor`, or pick another --provider.")
        return 1
    if args.usage:
        _err(f"[llmadapt] tokens used: {agent.total_tokens_used} "
             f"(input {sum(u['input'] for u in agent.usage_log)}, output {sum(u['output'] for u in agent.usage_log)})")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    agent, why = build_agent(args)
    _err(f"llmadapt chat - {agent.provider} / {agent.model} ({why}). Type /exit to quit, /reset to clear.")
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            _err("")
            return 0
        text = line.strip()
        if not text:
            continue
        if text in ("/exit", "/quit", "exit", "quit"):
            return 0
        if text == "/reset":
            agent, _ = build_agent(args)
            _err("(conversation cleared)")
            continue
        try:
            _out(agent.chat(text))
        except Exception as e:  # noqa: BLE001
            _err(f"error: {str(e).strip()}")


def cmd_outline(args: argparse.Namespace) -> int:
    from .toolkits import outline_source

    for i, path in enumerate(args.files):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        if i:
            _out("")
        _out(f"{path}  ({len(source.splitlines())} lines)")
        _out(outline_source(source, filename=path))
    return 0


def cmd_stub(args: argparse.Namespace) -> int:
    from .compressor import ContextCompressor

    with open(args.file, "r", encoding="utf-8", errors="replace") as f:
        _out(ContextCompressor.code_to_stub(f.read()))
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    from .compressor import ContextCompressor

    if args.file and args.file != "-":
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    _out(ContextCompressor.compress_tool_output(text, max_chars=args.max_chars, max_tokens=args.max_tokens))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from .presets import ORG_TEMPLATES, PALETTES, PERSONALITIES, SKILLS
    from .toolkits import TOOLKIT_DESCRIPTIONS

    kind = args.kind
    if kind == "toolkits":
        rows = [{"name": n, "description": d} for n, d in TOOLKIT_DESCRIPTIONS.items()]
    else:
        registry = {"skills": SKILLS, "personalities": PERSONALITIES,
                    "templates": ORG_TEMPLATES, "palettes": PALETTES}[kind]
        rows = registry.describe()
    if args.json:
        _out(json.dumps(rows, indent=2))
        return 0
    width = max((len(r["name"]) for r in rows), default=4)
    for row in rows:
        _out(f"{row['name']:<{width}}  {row.get('description', '')}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import serve

    serve(args)
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    _out(__version__)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmadapt",
        description="Provider-agnostic LLM agent toolkit - Anthropic, OpenAI, Gemini, Ollama.",
        epilog="Run `llmadapt doctor` first to see what is configured.",
    )
    parser.add_argument("--version", action="version", version=f"llmadapt {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("doctor", help="check API keys, .env, Ollama and hardware")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("ask", help="one-shot question; piped stdin is added as context")
    p.add_argument("prompt", nargs="*", help="the question or instruction")
    p.add_argument("-f", "--file", action="append", help="include a file's contents as context (repeatable)")
    p.add_argument("--json", action="store_true", help="ask for JSON and print it parsed and pretty")
    p.add_argument("--usage", action="store_true", help="print token usage to stderr")
    p.add_argument("--no-stdin", action="store_true", help="do not read piped stdin")
    _add_agent_options(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive conversation")
    _add_agent_options(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("outline", help="list classes/functions with line numbers (any language; best for Python)")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_outline)

    p = sub.add_parser("stub", help="print a Python file with function bodies stripped")
    p.add_argument("file")
    p.set_defaults(func=cmd_stub)

    p = sub.add_parser("compress", help="shrink long output: collapse repeats, keep head and tail")
    p.add_argument("file", nargs="?", help="file to read (default: stdin)")
    p.add_argument("--max-chars", type=int, default=4000)
    p.add_argument("--max-tokens", type=int, help="budget in estimated tokens instead of characters")
    p.set_defaults(func=cmd_compress)

    p = sub.add_parser("list", help="list built-in presets and toolkits")
    p.add_argument("kind", choices=("skills", "personalities", "templates", "toolkits", "palettes"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("mcp", help="serve llmadapt's tools over MCP (stdio) for Claude Code, Claude Desktop, etc.")
    _add_agent_options(p)
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("version", help="print the version")
    p.set_defaults(func=cmd_version)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, KeyError, ValueError, PermissionError, NotADirectoryError) as e:
        _err(f"llmadapt {args.command}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

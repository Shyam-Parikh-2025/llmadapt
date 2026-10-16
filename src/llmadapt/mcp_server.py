"""mcp_server.py - llmadapt as a Model Context Protocol server (stdio).

Lets an AI coding assistant (Claude Code, Claude Desktop, Cursor, ...) use
llmadapt as a set of tools: hand a sub-task to a cheaper or local model, get a
second opinion on code from a different model, or read a big file's outline
instead of the whole thing.

Register it with Claude Code:

    claude mcp add llmadapt -- llmadapt mcp --provider ollama --root .

or in any MCP client's JSON config:

    {"mcpServers": {"llmadapt": {"command": "llmadapt",
                                  "args": ["mcp", "--provider", "ollama", "--root", "."]}}}

Tools exposed:

  ask_model      delegate a prompt (optionally with files, skills, toolkits)
  review_code    a second-opinion code review of a file or a diff
  outline_code   classes/functions with line numbers, no bodies
  compress_file  a long log/output file squeezed to a character budget
  list_skills    the built-in skills ask_model accepts

Implementation notes. The protocol is JSON-RPC 2.0, one message per line on
stdin/stdout - small enough to implement with the standard library, so this
adds no dependency. stdout belongs to the protocol: anything else that prints
(an Agent's status lines, a stray print in a tool) is redirected to stderr for
the life of the server, because one stray line on stdout corrupts the stream.
Each ask_model call builds a fresh Agent - calls are stateless.
"""

import argparse
import json
import sys
from typing import Any, Callable, Dict, List, Optional

from . import __version__

SERVER_NAME = "llmadapt"
DEFAULT_PROTOCOL_VERSION = "2025-06-18"

INSTRUCTIONS = (
    "llmadapt lets you delegate work to other LLMs (local Ollama models or other API providers) and "
    "read code compactly. Use ask_model for self-contained sub-tasks that do not need your full context "
    "(drafting docstrings or tests, summarizing a long file, bulk rewrites, classification). Use review_code "
    "for an independent second opinion. Use outline_code before reading a large file. The delegated model "
    "is usually weaker than you: give it complete instructions and check its output."
)


class ToolError(Exception):
    """A failure to report to the client as a tool result with isError=true."""


class LlmadaptMCPServer:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, root: str = ".", toolkits: Optional[List[str]] = None,
                 skills: Optional[List[str]] = None, max_tokens: Optional[int] = None,
                 max_steps: int = 12, agent_factory: Optional[Callable[..., Any]] = None):
        from .toolkits import _Root

        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.root = _Root(root)
        self.allowed_toolkits = list(toolkits or [])
        self.default_skills = list(skills or [])
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self._agent_factory = agent_factory  # tests inject a fake here
        self._resolved: Optional[tuple] = None

    # -- the delegated model ----------------------------------------------

    def _target(self) -> tuple:
        if self._resolved is None:
            from .cli import _pick_ollama_model, pick_provider

            provider, why = pick_provider(self.provider)
            model = _pick_ollama_model(self.model) if provider == "ollama" else self.model
            self._resolved = (provider, model, why)
        return self._resolved

    def _make_agent(self, system: str = "", skills: Optional[List[str]] = None,
                    toolkits: Optional[List[str]] = None, provider: Optional[str] = None,
                    model: Optional[str] = None):
        from .core import Agent
        from .toolkits import toolkit

        default_provider, default_model, _ = self._target()
        provider = provider or default_provider
        model = model or (default_model if provider == default_provider else None)
        requested = list(toolkits or [])
        not_allowed = [t for t in requested if t not in self.allowed_toolkits]
        if not_allowed:
            raise ToolError(
                f"toolkit(s) {', '.join(not_allowed)} not enabled for this server. Enabled: "
                f"{', '.join(self.allowed_toolkits) or '(none)'} - start the server with --tools to allow more.")
        tools = toolkit(*requested, root=self.root.path) if requested else []
        factory = self._agent_factory or Agent
        agent = factory(provider=provider, model=model, base_url=self.base_url if provider == default_provider else None,
                        system_instruction=system or "", skills=(self.default_skills + list(skills or [])) or None,
                        tools=tools, max_tokens=self.max_tokens, max_tool_iterations=self.max_steps,
                        verbose=False)
        agent.status_stream = sys.stderr
        return agent

    def _read(self, path: str) -> str:
        from .toolkits import _read_text

        try:
            return _read_text(self.root.resolve(path))
        except (OSError, ValueError, PermissionError) as e:
            raise ToolError(f"cannot read {path!r}: {e}") from None

    # -- tools --------------------------------------------------------------

    def tool_ask_model(self, prompt: str, system: str = "", skills: Optional[List[str]] = None,
                       files: Optional[List[str]] = None, toolkits: Optional[List[str]] = None,
                       json_output: bool = False, provider: Optional[str] = None,
                       model: Optional[str] = None) -> str:
        parts = [prompt]
        for path in files or ():
            parts.append(f"--- {path} ---\n{self._read(path)}")
        agent = self._make_agent(system=system, skills=skills, toolkits=toolkits, provider=provider, model=model)
        full_prompt = "\n\n".join(parts)
        if json_output:
            answer = json.dumps(agent.chat_json(full_prompt), indent=2, ensure_ascii=False)
        else:
            answer = agent.chat(full_prompt)
        return f"{answer}\n\n[llmadapt: {agent.provider}/{agent.model}, {agent.total_tokens_used} tokens]"

    def tool_review_code(self, path: Optional[str] = None, diff: Optional[str] = None,
                         focus: str = "", provider: Optional[str] = None,
                         model: Optional[str] = None) -> str:
        if not path and not diff:
            raise ToolError("give either path (a file under the server root) or diff (unified diff text)")
        material = f"--- {path} ---\n{self._read(path)}" if path else f"--- diff ---\n{diff}"
        ask = ("Review this code. List concrete problems ordered by severity - bugs and correctness first, "
               "then risk, then clarity. For each, cite the line or hunk and say what to change. If it is fine, "
               "say so briefly rather than inventing issues.")
        if focus:
            ask += f" Focus especially on: {focus}."
        agent = self._make_agent(skills=["code-review"], provider=provider, model=model)
        answer = agent.chat(f"{ask}\n\n{material}")
        return f"{answer}\n\n[llmadapt review: {agent.provider}/{agent.model}]"

    def tool_outline_code(self, path: str) -> str:
        from .toolkits import outline_source

        source = self._read(path)
        return f"{path} ({len(source.splitlines())} lines)\n" + outline_source(source, filename=path)

    def tool_compress_file(self, path: str, max_chars: int = 4000) -> str:
        from .compressor import ContextCompressor

        text = self._read(path)
        return ContextCompressor.compress_tool_output(text, max_chars=int(max_chars))

    def tool_list_skills(self) -> str:
        from .presets import SKILLS

        return "\n".join(f"{s.name}: {s.description}" for s in SKILLS.all())

    def tool_definitions(self) -> List[Dict[str, Any]]:
        from .presets import SKILLS

        skill_names = SKILLS.names()
        target_note = ""
        if self._resolved:
            target_note = f" Default target: {self._resolved[0]}/{self._resolved[1] or 'default model'}."
        provider_prop = {"type": "string", "enum": ["anthropic", "openai", "gemini", "ollama"],
                         "description": "Override which provider answers this call."}
        model_prop = {"type": "string", "description": "Override the model for this call."}
        return [
            {
                "name": "ask_model",
                "description": ("Send a self-contained prompt to another LLM (local Ollama or an API model) and "
                                "return its answer. Good for delegating drafting, summarizing, bulk edits or "
                                "classification. Include everything it needs - it cannot see your context."
                                + target_note),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Complete instructions for the model."},
                        "system": {"type": "string", "description": "Optional system instruction."},
                        "skills": {"type": "array", "items": {"type": "string", "enum": skill_names},
                                   "description": "Built-in skills to give the model, e.g. python, testing."},
                        "files": {"type": "array", "items": {"type": "string"},
                                  "description": "Files (relative to the server root) to append to the prompt."},
                        "toolkits": {"type": "array", "items": {"type": "string",
                                                               "enum": self.allowed_toolkits or ["(none enabled)"]},
                                     "description": "Toolkits the model may use while answering "
                                                    "(only those enabled when the server started)."},
                        "json_output": {"type": "boolean",
                                        "description": "Ask for a JSON answer and return it parsed/pretty."},
                        "provider": provider_prop,
                        "model": model_prop,
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "review_code",
                "description": "Get an independent code review from another model, for a file or a diff.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File to review, relative to the server root."},
                        "diff": {"type": "string", "description": "Unified diff text to review instead of a file."},
                        "focus": {"type": "string", "description": "What to look at hardest, e.g. 'concurrency'."},
                        "provider": provider_prop,
                        "model": model_prop,
                    },
                },
            },
            {
                "name": "outline_code",
                "description": ("List a source file's classes and functions with line numbers and one-line "
                                "docstrings - a cheap map before reading a large file. No LLM call."),
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File relative to the server root."}},
                    "required": ["path"],
                },
            },
            {
                "name": "compress_file",
                "description": ("Read a long text file (log, test output, dump) squeezed to a character budget: "
                                "repeated lines collapsed, middle trimmed, head and tail kept. No LLM call."),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File relative to the server root."},
                        "max_chars": {"type": "integer", "description": "Character budget (default 4000)."},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "list_skills",
                "description": "List the built-in skills ask_model accepts, with one-line descriptions.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    # -- JSON-RPC -----------------------------------------------------------

    def handle(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process one JSON-RPC message; return the response, or None for a
        notification."""
        method = message.get("method")
        msg_id = message.get("id")
        is_notification = "id" not in message
        params = message.get("params") or {}
        try:
            if method == "initialize":
                requested = str(params.get("protocolVersion") or "")
                version = requested if requested[:4].isdigit() else DEFAULT_PROTOCOL_VERSION
                result: Any = {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": __version__},
                    "instructions": INSTRUCTIONS,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.tool_definitions()}
            elif method == "tools/call":
                result = self._call_tool(params.get("name"), params.get("arguments") or {})
            elif method and method.startswith("notifications/"):
                return None
            else:
                if is_notification:
                    return None
                return {"jsonrpc": "2.0", "id": msg_id,
                        "error": {"code": -32601, "message": f"method not found: {method}"}}
        except Exception as e:  # noqa: BLE001 - a protocol-level failure, reported as such
            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32603, "message": str(e)}}
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _call_tool(self, name: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        handlers = {
            "ask_model": self.tool_ask_model,
            "review_code": self.tool_review_code,
            "outline_code": self.tool_outline_code,
            "compress_file": self.tool_compress_file,
            "list_skills": self.tool_list_skills,
        }
        handler = handlers.get(name or "")
        if handler is None:
            return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}
        try:
            text = handler(**arguments)
            return {"content": [{"type": "text", "text": text}], "isError": False}
        except ToolError as e:
            return {"content": [{"type": "text", "text": str(e)}], "isError": True}
        except TypeError as e:
            return {"content": [{"type": "text", "text": f"bad arguments for {name}: {e}"}], "isError": True}
        except Exception as e:  # noqa: BLE001 - tool failures go back to the model, not up the stack
            return {"content": [{"type": "text", "text": f"{name} failed: {e.__class__.__name__}: {str(e).strip()}"}],
                    "isError": True}

    def serve_forever(self, stdin=None, stdout=None) -> None:
        """Read newline-delimited JSON-RPC from stdin until EOF."""
        stdin = stdin or sys.stdin
        out = stdout or sys.stdout
        for raw in stdin:
            line = raw.strip() if isinstance(raw, str) else raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as e:
                self._write(out, {"jsonrpc": "2.0", "id": None,
                                  "error": {"code": -32700, "message": f"parse error: {e}"}})
                continue
            if isinstance(message, list):
                responses = [r for r in (self.handle(m) for m in message if isinstance(m, dict)) if r]
                if responses:
                    self._write(out, responses)
                continue
            response = self.handle(message) if isinstance(message, dict) else None
            if response is not None:
                self._write(out, response)

    @staticmethod
    def _write(out, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False) + "\n"
        buffer = getattr(out, "buffer", None)
        if buffer is not None:
            buffer.write(data.encode("utf-8"))  # bytes: no "\r\n" translation on Windows
            buffer.flush()
        else:
            out.write(data)
            out.flush()


def serve(args: argparse.Namespace) -> None:
    """Entry point for `llmadapt mcp`."""
    from .cli import _split

    server = LlmadaptMCPServer(
        provider=getattr(args, "provider", None), model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None), root=getattr(args, "root", ".") or ".",
        toolkits=_split(getattr(args, "tools", None)), skills=_split(getattr(args, "skills", None)),
        max_tokens=getattr(args, "max_tokens", None), max_steps=getattr(args, "max_steps", None) or 12,
    )
    protocol_out = sys.stdout
    sys.stdout = sys.stderr  # everything except protocol messages goes to stderr
    try:
        provider, model, why = server._target()
        print(f"llmadapt MCP server {__version__}: delegating to {provider}/{model or 'default'} ({why}); "
              f"root={server.root.path}; toolkits={','.join(server.allowed_toolkits) or 'none'}",
              file=sys.stderr, flush=True)
        stdin = getattr(sys.stdin, "buffer", sys.stdin)
        server.serve_forever(stdin=stdin, stdout=protocol_out)
    finally:
        sys.stdout = protocol_out


__all__ = ["LlmadaptMCPServer", "serve", "ToolError"]

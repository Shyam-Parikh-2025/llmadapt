import json
import inspect
import logging
import urllib.request
import urllib.error
import threading
import asyncio
import time
import warnings
from typing import get_type_hints, Any, Callable, Optional

# escalation decision now defaults on Company itself (see
# company/escalation.py's default_on_escalation) - Company(on_escalation=None)
# falls back to it instead of requiring every caller to pass a callback.
from .compressor import PIN_KEY, CompressionPolicy, ContextCompressor, HistoryCompactionPolicy
from .env import resolve_api_key

logger = logging.getLogger(__name__)


class Conversation:
    """ Conversation Class:
    This needs to manage chat memory and has to have information of where the information came from, 
    such as from a tool, model or the user.
    This then needs to be able to send such memory when asked for based on the syntax and structure 
    required depending on the AI model used.
    Key Providers: Gemini, OpenAI, Anthropic, Ollama (for local models).
    These providers should naturally work within the library.
    Custom Providers are possible to use but a function to specify the structure is required. """
    
    def __init__(self, system_instruction: str = "", archive=None):
        self.history = []
        self.system_instruction = system_instruction
        self.model_role = "assistant"
        # An optional archive.RunArchive. None (the default) means nothing is
        # written anywhere. Every message is archived as it is appended -
        # write-through rather than at compaction time - so the file is the
        # untouched transcript by construction, and a crash mid-run still
        # leaves everything up to that point on disk. See archive.py.
        self.archive = archive

        if self.system_instruction:
            self._append({"role": "system", "content": self.system_instruction})

    def _append(self, msg: dict):
        """The single place a message enters history, so the archive cannot
        miss one. Archiving first would risk recording a message that then
        failed to be stored; archiving after keeps history authoritative and
        the file a faithful shadow of it."""
        self.history.append(msg)
        if self.archive is not None:
            self.archive.append("message", dict(msg), position=len(self.history) - 1)

    def add_user_msg(self, text: str):
        self._append({"role": "user", "content": text})
    
    def add_model_msg(self, text: str = None, tool_calls: list = None, role: str = None,
                       native: dict = None, native_provider: str = None):
        if role is None:
            role = self.model_role

        msg = {"role": role} 
        if text:
            msg["content"] = text
        if tool_calls:
            normalized_calls = []
            for tc in tool_calls:
                func = tc.get("function") or {}
                args = func.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                normalized_calls.append({
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": func.get("name"),
                        "arguments": args
                    }
                })
            msg["tool_calls"] = normalized_calls

        if native is not None:
            msg["_native"] = native
            msg["_native_provider"] = native_provider
        self._append(msg)
    
    # The marker HistoryCompactionPolicy honours. Leading underscore on purpose:
    # export_for() already strips every "_"-prefixed key before a message goes
    # over the wire (the same convention _native/_native_provider use), so a pin
    # is bookkeeping this library can see and no provider ever will.
    PIN_KEY = PIN_KEY

    def pin(self, text: str, role: str = "user", reason: str = "") -> dict:
        """Append a message that history compaction may never touch.

        The conversation-side twin of Phase 7's ALWAYS_KEEP_KINDS. Until now the
        only protected thing was a leading system message, which covers the
        employee's identity and nothing else - so a spec, an API contract, a
        decision made at turn 3 that everything after it depends on, all became
        eligible to be summarized into "the user asked about the schema" once
        the conversation grew.

        Use it for facts whose loss changes the answers: constraints that must
        hold, an interface the code has to match, a correction the model got
        wrong once already. Not for bulk - a pinned message is by definition
        never compacted, so pinning a large document means carrying it in full
        for the rest of the run.
        """
        msg = {"role": role, "content": text, self.PIN_KEY: True}
        if reason:
            msg["_pin_reason"] = reason
        self._append(msg)
        return msg

    def pin_last(self, reason: str = "") -> Optional[dict]:
        """Pin the message already at the end of the history.

        For pinning something that arrived normally - a model's answer worth
        keeping verbatim, or a user turn that turned out to be load-bearing -
        rather than adding a new message to say it again.
        """
        if not self.history:
            return None
        msg = self.history[-1]
        msg[self.PIN_KEY] = True
        if reason:
            msg["_pin_reason"] = reason
        return msg

    def pinned(self) -> list:
        """Every pinned message, in order. The leading system instruction is
        included: it is protected by the same guarantee, just by a different
        mechanism, and a caller asking "what survives compaction?" wants one
        answer rather than two."""
        out = []
        if self.history and self.history[0].get("role") == "system":
            out.append(self.history[0])
        out.extend(m for m in self.history if m.get(self.PIN_KEY) and m not in out)
        return out

    def add_tool_response(self, function_name: str, output: str, tool_call_id: str = None):
        msg = {
            "role": "tool",
            "name": function_name,
            "content": output
        }
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        self._append(msg)

    def export_for(self, provider: str, special_format: Callable[[list], list] = None,
                   model: Optional[str] = None) -> list:
        """Translates local flat memory frames into target external API layouts.

        `model` is the model the request is about to go to. Only Anthropic
        uses it: thinking blocks are tied to the model that produced them, so
        replaying them to a different model (after switch_api) is rejected -
        they are dropped from that model's view instead."""
        if special_format and callable(special_format):
            return special_format(self.history)

        if provider in ["ollama", "openai", "custom"]:
            cleaned_history = []
            for msg in self.history:
                cleaned_msg = {k: v for k, v in msg.items() if not k.startswith("_")}
                
                if provider in ["openai", "custom"] and "tool_calls" in cleaned_msg:
                    stringified_calls = []
                    for tc in cleaned_msg["tool_calls"]:
                        func = tc.get("function") or {}
                        args = func.get("arguments") or {}
                        if not isinstance(args, str):
                            args = json.dumps(args)
                        
                        stringified_calls.append({
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": func.get("name"),
                                "arguments": args
                            }
                        })
                    cleaned_msg["tool_calls"] = stringified_calls
                
                cleaned_history.append(cleaned_msg)
            return cleaned_history
            
        elif provider == "anthropic":
            return self._export_anthropic(model=model)
            
        elif provider == "gemini":    
            return self._export_gemini()

    def change_system_instruction(self, new_instruction: str):
        self.system_instruction = new_instruction
        updated = False
        for msg in self.history:
            if msg["role"] == "system":
                msg["content"] = self.system_instruction
                updated = True
                break
        if not updated and self.system_instruction:
            self.history.insert(0, {"role": "system", "content": self.system_instruction})

    def change_model_role(self, new_role: str):
        self.model_role = new_role

    def _export_anthropic(self, model: Optional[str] = None) -> list:
        """Structures conversation blocks to comply with Anthropic context constraints."""
        out = []
        pending_tool_results = []
        emitted_native_ids = set() 

        def flush():
            if pending_tool_results:
                out.append({"role": "user", "content": pending_tool_results.copy()})
                pending_tool_results.clear()

        for msg in self.history:
            if msg["role"] == "system":
                continue

            if msg.get("_native_provider") == "anthropic" and "_native" in msg:
                native_hash = json.dumps(msg["_native"], sort_keys=True)
                if native_hash in emitted_native_ids:
                    continue
                emitted_native_ids.add(native_hash)
                flush()
                native = msg["_native"]
                produced_by = msg.get("_native_model")
                if model and produced_by and produced_by != model and isinstance(native.get("content"), list):
                    kept = [b for b in native["content"]
                            if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))]
                    native = dict(native, content=kept)
                out.append(native)
                continue

            if msg["role"] == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg["content"]
                })
                continue

            flush()

            if msg["role"] == "user":
                out.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tool_call in msg.get("tool_calls", []):
                    content.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": tool_call["function"]["arguments"]
                    })
                out.append({"role": "assistant", "content": content if content else (msg.get("content") or "")})

        flush()
        return out

    def _export_gemini(self) -> list:
        """Structures conversation blocks to comply with Gemini context constraints."""
        gemini_history = []
        for msg in self.history:
            if msg["role"] == "system": 
                continue 
            if msg.get("_native_provider") == "gemini" and "_native" in msg:
                gemini_history.append(msg["_native"])
                continue
            if msg["role"] == "user":
                gemini_history.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == self.model_role:
                parts = []
                if msg.get("content"):
                    parts.append({"text": msg["content"]})
                if "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        function_call_part = {
                            "name": tool_call["function"]["name"],
                            "args": tool_call["function"]["arguments"]
                        }
                        if tool_call.get("id"):
                            function_call_part["id"] = tool_call["id"]
                        parts.append({"functionCall": function_call_part})
                gemini_history.append({"role": "model", "parts": parts})
            elif msg["role"] == "tool":
                function_response_part = {"name": msg["name"], "response": {"result": msg["content"]}}
                if msg.get("tool_call_id"):
                    function_response_part["id"] = msg["tool_call_id"]
                gemini_history.append({
                    "role": "user",
                    "parts": [{"functionResponse": function_response_part}]
                })
        return gemini_history

def loop_local_lock(owner: Any, attr: str) -> "asyncio.Lock":
    """An asyncio.Lock stored on `owner`, recreated whenever the running event
    loop changes.

    A lock cannot simply be built in `__init__` here. On Python 3.9 (the
    floor this package supports) `asyncio.Lock()` binds to an event loop when
    it is created, and every synchronous entry point in this library starts a
    *fresh* loop via `asyncio.run()` - so a lock made at hire() time would
    belong to a loop that no longer exists by the time anyone waits on it, and
    would raise instead of guarding anything.

    Rebinding per loop is not a weakening of the guarantee: a lock only ever
    has to serialize coroutines running on the same loop, and two different
    loops (a fresh `asyncio.run()` per sync call) cannot be running each
    other's coroutines concurrently in the first place.

    Must be called from inside a running loop, which every caller here is.
    """
    loop = asyncio.get_running_loop()
    existing = getattr(owner, attr, None)
    if existing is not None and existing[0] is loop:
        return existing[1]
    lock = asyncio.Lock()
    setattr(owner, attr, (loop, lock))
    return lock


def run_coroutine_blocking(coro_factory: Callable[[], Any], what: str = "this call") -> Any:
    """Runs a coroutine to completion from synchronous code, and survives
    being called from inside somebody else's running event loop.

    Every synchronous entry point in this library (`Agent.chat`,
    `Company.run`, ...) is a thin wrapper over its async twin, so that there
    is one implementation of each rather than two that drift. Plain
    `asyncio.run()` would be enough - except that `asyncio.run()` raises
    `RuntimeError: asyncio.run() cannot be called from a running event loop`
    the moment somebody calls the sync API from inside an `async def`, which
    is exactly what a FastAPI route handler doing `result = company.run(task)`
    looks like. Letting that native error surface would make the simple sync
    API unusable in the frameworks people most want to use it from.

    So: if there is no loop running, `asyncio.run()`. If there is, warn loudly
    and run the coroutine on a fresh thread, which has no loop of its own and
    so may legally start one.

    The warning is not decoration. The recovery produces a *correct result*
    and nothing more: the caller's event loop stays blocked for the whole
    duration anyway, because Python only hands control back to a loop at an
    `await`, and no amount of cleverness in the callee can grant that on the
    caller's behalf. The only real fix is at the call site, which is what the
    warning says. Nothing here silently approves, retries or skips anything -
    the coroutine runs exactly once, and its exceptions are re-raised.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())  # the ordinary case: no loop, no drama

    warnings.warn(
        f"{what} was called from inside a running event loop. It has been run on a helper "
        f"thread and the result is correct, but your event loop stayed blocked for the whole "
        f"call - only an `await` at your call site can hand control back to it, which a "
        f"synchronous method cannot do for you. Use the async twin instead (e.g. "
        f"`await company.run_async(task)` / `await agent.chat_async(text)`), or "
        f"`await asyncio.to_thread(company.run, task)` if you must keep the sync call.",
        RuntimeWarning,
        stacklevel=3,
    )

    results: list = []
    errors: list = []

    def _runner():
        try:
            results.append(asyncio.run(coro_factory()))
        except BaseException as e:  # noqa: BLE001 - re-raised on the calling thread below
            errors.append(e)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return results[0]


class ToolControlFlow(Exception):
    """Base class for exceptions that must travel *through* a tool call
    untouched, instead of being turned into a message for the model.

    `ToolRegistry.execute()` deliberately converts any exception a tool raises
    into a `"Tool Execution Failure (...)"` string, because a tool that
    crashed is ordinarily something the model should see and work around -
    that is the whole reason a bad argument doesn't kill a run.

    A small number of exceptions are not that. They are not "this tool
    failed", they are "this run must stop, or wait". The one that exists today
    is `company.escalation.EscalationUnresolved`: a human was asked whether to
    continue and said no. Because `delegate_to_<name>` is an ordinary tool
    (Employee.delegate_tool), a decline raised inside a delegated employee
    used to be caught by execute() and handed back to the *manager's* model as
    a failed-tool string - so the manager read "that didn't work", tried
    something else, and the run finished normally reporting success. The human
    said stop and nothing stopped, which is the exact inversion of the
    0-means-always-ask-a-human-first rule the budget and escalation systems are
    built on.

    Anything subclassing this is re-raised by execute() rather than
    stringified. It is a base class rather than a hardcoded tuple of types
    inside execute() so that core.py needs no import of the company layer (the
    dependency arrow points one way, see company/employee.py's docstring), and
    so a future "this run is paused waiting on a human" signal can opt in by
    inheriting rather than by editing a list here.
    """


class RunPaused(ToolControlFlow):
    """Raised when an escalation handler asks for the run to *wait* rather than
    approving or declining - see company.escalation.ESCALATION_PENDING.

    Lives here, next to ToolControlFlow, for the same reason that class does:
    it has to escape `ToolRegistry.execute()` intact, and core.py must not
    import the company layer to make that happen. It carries the escalation
    event as an opaque attribute - core never inspects it.

    What makes a pause resumable is not this exception, which only unwinds the
    stack. It is that each Agent it passes through stores the turn it was in
    the middle of (`Agent._pending_turn`): the provider response that asked for
    the tools, and the outputs of the tool calls that had already finished.
    Python cannot serialize a paused call stack, but it does not have to - a
    turn is fully described by those two things, and a chain of delegations is
    just a stack of turns, each one held by the Agent it belongs to.
    """

    def __init__(self, event: Any = None, task: Optional[str] = None):
        self.event = event
        self.task = task
        super().__init__(f"run paused awaiting a decision on: {getattr(event, 'message', event)!r}")


def extract_json(text: str) -> Any:
    """Parse the JSON value out of a model reply.

    Tolerates the usual wrapping - a ```json fence, a sentence before or
    after - by trying, in order: the whole text, the first fenced block, then
    the first '{' or '[' decoded with raw_decode. Raises ValueError (with the
    JSON error message) if none of them parse.
    """
    import re

    if text is None:
        raise ValueError("empty reply")
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        first_error = e
    fence = re.search(r"```(?:json|JSON)?\s*\n?(.*?)```", candidate, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    # Earliest opener first, so "[{...}, {...}]" comes back as the list rather
    # than as its first element.
    for start, char in enumerate(candidate):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError(str(first_error))


def _add_anthropic_cache_breakpoints(payload: dict) -> None:
    """Mark the stable prefix of an Anthropic request as cacheable: the
    system prompt, the tool list, and the conversation so far (the last
    message). Later requests that share that prefix are billed at the cache
    read rate - most of a tool loop, or a long chat, re-sends exactly that.

    Works on copies: history/tool dicts may be shared with the stored
    conversation, which must never carry provider bookkeeping."""
    marker = {"type": "ephemeral"}
    system = payload.get("system")
    if isinstance(system, str) and system:
        payload["system"] = [{"type": "text", "text": system, "cache_control": marker}]
    tools = payload.get("tools")
    if tools:
        payload["tools"] = list(tools[:-1]) + [dict(tools[-1], cache_control=marker)]
    messages = payload.get("messages") or []
    if messages:
        last = dict(messages[-1])
        content = last.get("content")
        if isinstance(content, str) and content:
            last["content"] = [{"type": "text", "text": content, "cache_control": marker}]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            last["content"] = list(content[:-1]) + [dict(content[-1], cache_control=marker)]
        payload["messages"] = list(messages[:-1]) + [last]


def _merge_extra(payload: dict, extra: Optional[dict]) -> dict:
    """Overlay `extra` onto a request body. Nested dicts merge one level
    deep, so extra_params={"options": {"temperature": 0}} on Ollama keeps
    the num_predict this library already put in "options"."""
    if not extra:
        return payload
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            merged = dict(payload[key])
            merged.update(value)
            payload[key] = merged
        else:
            payload[key] = value
    return payload


def ollama_base_url() -> str:
    """Where Ollama listens: OLLAMA_HOST if set (the variable Ollama itself
    reads - "host:port", "0.0.0.0", or a full URL), else localhost:11434."""
    import os
    host = (os.environ.get("OLLAMA_HOST") or "").strip().rstrip("/")
    if not host:
        return "http://localhost:11434"
    if "://" not in host:
        host = "http://" + host
    scheme, rest = host.split("://", 1)
    hostname, _, port = rest.partition(":")
    if hostname in ("0.0.0.0", ""):
        hostname = "localhost"  # a bind-all address is not something to connect to
    return f"{scheme}://{hostname}:{port or '11434'}"


def _safe_print(text: str, **kwargs: Any) -> None:
    """print() that cannot crash a run over an encoding. On Windows, stdout
    piped into another process (exactly how a coding agent or CI captures
    output) is often cp1252, where '➔' and the spinner glyphs raise
    UnicodeEncodeError - inside Agent.chat(), for a status message."""
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        ascii_text = text.replace("➔", "->").encode("ascii", errors="replace").decode("ascii")
        print(ascii_text, **kwargs)


def _stdout_is_tty(stream: Any = None) -> bool:
    try:
        import sys
        stream = stream or sys.stdout
        return bool(stream and stream.isatty())
    except Exception:
        return False


_PRIMITIVE_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_schema_for(annotation: Any) -> dict:
    """A JSON-schema fragment for one parameter's type hint.

    Deliberately forgiving: anything it does not recognise becomes
    {"type": "string"}, which is what every parameter used to be. A tool
    whose hints are exotic still registers; it just gets a looser schema.
    """
    import enum
    import typing

    if annotation in _PRIMITIVE_JSON_TYPES:
        return {"type": _PRIMITIVE_JSON_TYPES[annotation]}
    if annotation in (list, tuple, set, frozenset):
        return {"type": "array"}
    if annotation is dict:
        return {"type": "object"}
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = [m.value for m in annotation]
        out = _json_schema_for(type(values[0])) if values else {"type": "string"}
        out["enum"] = values
        return out

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union or type(annotation).__name__ == "UnionType":
        non_none = [a for a in args if a is not type(None)]
        # Optional[X] is just X to the model - whether it may be omitted is
        # what "required" says, and that comes from the default value.
        return _json_schema_for(non_none[0]) if len(non_none) == 1 else {"type": "string"}
    if origin is typing.Literal:
        values = list(args)
        out = _json_schema_for(type(values[0])) if values else {"type": "string"}
        out["enum"] = values
        return out
    if origin in (list, tuple, set, frozenset) or (
        origin is not None and getattr(origin, "__name__", "") in ("Sequence", "Iterable", "Collection")
    ):
        out = {"type": "array"}
        item_args = [a for a in args if a is not Ellipsis]
        if len(set(item_args)) == 1:
            out["items"] = _json_schema_for(item_args[0])
        return out
    if origin is dict or (origin is not None and getattr(origin, "__name__", "") == "Mapping"):
        return {"type": "object"}
    return {"type": "string"}


def _param_docs_from_docstring(doc: str) -> dict:
    """Per-parameter descriptions from a docstring's `Args:` section
    (Google style) or `:param name:` lines (Sphinx style). Returns {} when
    the docstring has neither - the caller keeps its generic fallback."""
    import re

    docs: dict = {}
    for match in re.finditer(r"^\s*:param\s+(?:[\w\[\], .]+\s+)?(\w+)\s*:\s*(.+)$", doc, re.MULTILINE):
        docs[match.group(1)] = match.group(2).strip()
    if docs:
        return docs

    lines = doc.splitlines()
    in_args = False
    current = None
    args_indent = None
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(Args|Arguments|Parameters|Params)\s*:\s*$", stripped):
            in_args, current, args_indent = True, None, None
            continue
        if not in_args:
            continue
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if re.match(r"^[A-Z][A-Za-z ]*:\s*$", stripped) and (args_indent is None or indent < args_indent):
            break  # next section, e.g. "Returns:"
        entry = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
        if entry and (args_indent is None or indent <= args_indent):
            args_indent = indent if args_indent is None else args_indent
            current = entry.group(1)
            docs[current] = entry.group(2).strip()
        elif current is not None:
            docs[current] = (docs[current] + " " + stripped).strip()
    return docs


class ToolRegistry:
    """ Tool Registry Class:
    This class needs to handle tool storage and usage.
    Methods to register tool use tool.
    It should be able to export for all AI models and work with all providers.
    Again, custom providers are possible to use but a function to specify the structure is required."""

    def __init__(self):
        self.functions_maps = {}
        self.schemas = {} 
    
    def register(self, python_function, schema: dict = None):
        """Register a callable as a tool.

        With no `schema`, one is generated from the function itself:

        - the docstring becomes the tool description;
        - each parameter's JSON type comes from its type hint - `str`, `int`,
          `float`, `bool`, `list[...]`/`tuple`/`set`, `dict`, `Optional[X]`,
          `Literal[...]` and `Enum` subclasses are understood; anything else
          is sent as a string;
        - a parameter's description is taken from a Google-style `Args:`
          section or Sphinx `:param name:` lines when the docstring has one;
        - parameters without a default are required.
        """
        if schema is None:
            func_name = python_function.__name__
            func_doc = inspect.getdoc(python_function) or "No description available."

            try:
                type_hints = get_type_hints(python_function)
            except Exception:
                # Unresolvable forward references should cost precision in the
                # schema, never the ability to register the tool at all.
                type_hints = {}
            signature = inspect.signature(python_function)
            param_docs = _param_docs_from_docstring(func_doc)
            properties = {}
            required_params = []

            for param_name, param in signature.parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue  # *args/**kwargs have no meaningful JSON-schema shape
                prop = _json_schema_for(type_hints.get(param_name, str))
                prop["description"] = param_docs.get(param_name) or f"The {param_name} parameter"
                properties[param_name] = prop
                if param.default == inspect.Parameter.empty:
                    required_params.append(param_name)

            schema = {
                "name": func_name,
                "description": func_doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params
                }
            }
        else:
            func_name = schema["name"]

        self.functions_maps[func_name] = python_function
        self.schemas[func_name] = schema 
    
    def _prepare_args(self, name: str, args) -> tuple:
        """Normalizes a provider's raw argument payload into kwargs.

        Returns (kwargs, None) or (None, error_string). Split out of execute()
        so the sync and async dispatch paths share one definition of what a
        provider is allowed to send - two copies of this coercion is exactly
        how a model's odd-but-tolerated argument shape starts working on one
        path and failing on the other.
        """
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                # Auto-wrap single primitive string inputs into standard key payload
                sig = inspect.signature(self.functions_maps[name])
                params = list(sig.parameters.keys())
                if len(params) == 1:
                    args = {params[0]: args}

        if not isinstance(args, dict):
            return None, f"Error: Arguments for tool '{name}' must be passed as a dictionary."
        return self._coerce_enums(name, args), None

    def _coerce_enums(self, name: str, args: dict) -> dict:
        """A model can only send an Enum parameter's *value* (the schema
        lists values); turn it back into the member the function declared."""
        import enum
        try:
            hints = get_type_hints(self.functions_maps[name])
        except Exception:
            return args
        out = dict(args)
        for key, value in args.items():
            hint = hints.get(key)
            if isinstance(hint, type) and issubclass(hint, enum.Enum) and not isinstance(value, hint):
                try:
                    out[key] = hint(value)
                except ValueError:
                    pass  # leave it; the tool's own error is more specific
        return out

    @staticmethod
    def _failure_text(e: BaseException) -> str:
        """The one wording for 'your tool call blew up', shared by both paths."""
        return (f"Tool Execution Failure ({e.__class__.__name__}): {str(e)}. "
                f"Please adjust your input arguments.")

    def execute(self, name: str, args: dict) -> str:
        """Invokes a registered tool, safely converting crashes into explicit textual logs."""
        if name not in self.functions_maps:
            return f"Error: Tool '{name}' is not registered in this system."

        try:
            kwargs, error = self._prepare_args(name, args)
            if error is not None:
                return error
            res = self.functions_maps[name](**kwargs)
            return self._stringify_tool_output(res)
        except ToolControlFlow:
            # Not a tool failure - a signal about the run itself. Must reach
            # the caller intact instead of becoming text for the model. See
            # ToolControlFlow's docstring for what happened when it didn't.
            raise
        except Exception as e:
            return self._failure_text(e)

    async def execute_async(self, name: str, args: dict) -> str:
        """The awaitable form of execute(), with the same contract: returns a
        string for every ordinary outcome, re-raises ToolControlFlow.

        A tool defined with `async def` is awaited directly. An ordinary
        synchronous tool - which is every tool anyone has written against this
        library so far, `delegate_to_<name>` included - is run through
        `asyncio.to_thread`, because calling it inline would block the event
        loop for its whole duration and defeat the point of being here.

        Running sync tools on a worker thread has a second, load-bearing
        effect: a worker thread has no event loop of its own, so a sync tool
        that internally calls back into `Agent.chat()` (again, this is exactly
        what `delegate_to_<name>` does) gets a clean `asyncio.run()` rather
        than tripping the nested-loop fallback on every single delegation.
        """
        if name not in self.functions_maps:
            return f"Error: Tool '{name}' is not registered in this system."

        func = self.functions_maps[name]
        if not inspect.iscoroutinefunction(func):
            return await asyncio.to_thread(self.execute, name, args)

        try:
            kwargs, error = self._prepare_args(name, args)
            if error is not None:
                return error
            res = await func(**kwargs)
            return self._stringify_tool_output(res)
        except ToolControlFlow:
            raise
        except Exception as e:
            return self._failure_text(e)

    def export_for(self, provider: str, special_format: Callable[[list], list] = None) -> list:
        if not self.schemas:
            return []

        schema_list = list(self.schemas.values()) 

        if special_format and callable(special_format):
            return special_format(schema_list)

        if provider in ["ollama", "openai"]:
            return [{"type": "function", "function": schema} for schema in schema_list]
        elif provider == "anthropic":
            return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in schema_list]
        elif provider == "gemini":
            return [{"functionDeclarations": schema_list}]

    @staticmethod
    def _stringify_tool_output(out) -> str:
        """Serializes tool return values into valid JSON strings for model consumption."""
        if isinstance(out, str):
            return out
        try:
            return json.dumps(out)
        except TypeError:
            return str(out)

class Agent:
    """ Agent Class:
    This is the core agent class that needs to have several key features:
    This agent handles the Conversation and Tool object.
    When an AI responds asking for the results of certain tools, this must run it; however, there 
    should be a max to how many times this iteration can happen - function to change max iterations.
    A function to switch APIs and all - will make later additions easier.
    Add tool and send request to AI model functions are needed.
    An overall Chat function that handles sending the payload to the AI model and handles the response. """
    
    # Used only when no model= is given. These go stale as providers retire
    # models (the previous table pointed at claude-3-5-sonnet-20241022 and
    # gpt-4o-mini, both retired or deprecated by late 2026), so each one can
    # be overridden without a code change via LLMADAPT_<PROVIDER>_MODEL, e.g.
    # LLMADAPT_ANTHROPIC_MODEL=claude-haiku-5-5. Passing model= always wins.
    DEFAULT_MODELS = {
        "gemini": "gemini-3.6-flash",
        "anthropic": "claude-sonnet-5-5",
        "openai": "gpt-5.4-mini",
        "ollama": "llama3.1:8b"
    }

    # Transient HTTP statuses worth retrying - the same set the official
    # Anthropic and OpenAI SDKs retry: request timeout, conflict/lock, rate
    # limit, and every 5xx (including Anthropic's 529 "overloaded").
    RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

    @classmethod
    def default_model_for(cls, provider: str) -> str:
        """The model used when none is passed: `LLMADAPT_<PROVIDER>_MODEL`
        from the environment (or a .env file) if set, else DEFAULT_MODELS."""
        from .env import env_value
        override = env_value(f"LLMADAPT_{provider.upper()}_MODEL")
        return override or cls.DEFAULT_MODELS.get(provider, "custom-model")

    def __init__(self, provider: str, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, is_local: bool = False, system_instruction: str = "",
                 max_tool_iterations: int = 6,
                 max_tokens: Optional[int] = None, compression_policy: Optional[CompressionPolicy] = None,
                 history_compaction: Optional[HistoryCompactionPolicy] = None,
                 archive: Optional[Any] = None,
                 key_alias: Optional[str] = None, key_env: Optional[str] = None,
                 max_context_tokens: Optional[int] = None,
                 skills: Optional[list] = None, personality: Optional[Any] = None,
                 tools: Optional[list] = None,
                 verbose: Optional[bool] = None,
                 request_timeout: Optional[float] = 600.0,
                 max_retries: int = 2,
                 extra_params: Optional[dict] = None,
                 prompt_cache: bool = False):
        """
        Most callers need only `provider` (and `model`). The rest, briefly:

        skills / personality: names from the built-in preset catalog
            (`llmadapt.SKILLS.names()`, `llmadapt.PERSONALITIES.names()`) or
            Skill/Personality objects. They are composed into the system
            instruction, after `system_instruction`, exactly as
            `Company.hire()` does it.
        tools: callables (or a toolkit list from `llmadapt.toolkits`) to
            register straight away - same as calling `add_tool` on each.
        verbose: print stage messages and a spinner while working. None
            (default) reads LLMADAPT_VERBOSE from the environment and falls
            back to True; the spinner is only ever drawn on a real terminal.
        request_timeout: seconds before an HTTP request is abandoned (None
            waits forever, which is what every request did before).
        max_retries: extra attempts for transient failures - 429, 5xx,
            Anthropic's 529, dropped connections - with exponential backoff
            that honours a Retry-After header.
        extra_params: extra fields merged into every request body, for knobs
            this library does not model itself - e.g. {"temperature": 0.2},
            {"reasoning_effort": "low"} or Ollama {"options": {...}}.
        prompt_cache: Anthropic only - mark the system prompt, tools and
            conversation prefix as cacheable, so repeated requests (every step
            of a tool loop, every turn of a chat) re-read them at ~10% of the
            input price. The first write costs 25% extra, so it pays off from
            the second request on. Other providers cache automatically.
        """
        self.archive = archive
        if skills or personality:
            from .presets import compose_system_instruction
            system_instruction = compose_system_instruction(
                base=system_instruction, personality=personality, skills=skills or ())
        self.conversation = Conversation(system_instruction=system_instruction, archive=archive)
        self.tool_registry = ToolRegistry()
        self.change_api(provider=provider, model=model, base_url=base_url, api_key=api_key,
                        key_alias=key_alias, key_env=key_env)
        self.set_max_tool_iterations(max_tool_iterations)
        self.max_tokens = max_tokens
        self.thinking_stage = "initial state"
        if verbose is None:
            from .env import env_value
            flag = (env_value("LLMADAPT_VERBOSE") or "").strip().lower()
            verbose = flag not in ("0", "false", "no", "off")
        self.verbose = bool(verbose)
        # Where stage messages and the spinner go. None means sys.stdout at
        # print time; the CLI and MCP server point it at sys.stderr so stdout
        # carries only the answer (or, for MCP, only the protocol).
        self.status_stream = None
        self.request_timeout = request_timeout
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        self.max_retries = max_retries
        self.extra_params = dict(extra_params or {})
        self.prompt_cache = bool(prompt_cache)
        for tool_func in tools or ():
            self.add_tool(tool_func)

        # A separate concept from max_tokens (the per-request *output* cap
        # sent to the provider): this is a whole-conversation budget tracked
        # via tokens_used()/tokens_left(), set here or later through
        # set_max_context_tokens(). None (the default) means no budget to
        # count down from - tokens_left() then returns None rather than a
        # number that would imply one.
        self.max_context_tokens: Optional[int] = None
        self.set_max_context_tokens(max_context_tokens)

        self.is_local = is_local

        # Defaults to a disabled policy (compress() is a no-op passthrough) so any existing
        # Agent(...) call that doesn't pass this keeps behaving exactly like it did before -
        # see CompressionPolicy in compressor.py for why this is a policy object rather than
        # the agent owning/constructing a compressor instance.
        self.compression_policy = compression_policy or CompressionPolicy()

        # The other half of "installing the compressor on an agent".
        # CompressionPolicy shrinks one tool result; this compacts the
        # conversation as a whole once it has outgrown its trigger.
        #
        # This class was written alongside CompressionPolicy and then never
        # wired to anything - it had no call site anywhere in the library and
        # was not even exported, so history compaction was a feature you had
        # to reach into compressor.py and drive yourself. It is a real option
        # now, and None still means "never touch history", so an Agent built
        # the old way behaves exactly as it did.
        self.history_compaction = history_compaction

        # Real per-provider usage, captured from each response as it comes back
        # (see _extract_usage/_record_usage) - not estimated. company.py's budget
        # governance (budget.py) reads total_tokens_used directly; usage_log keeps
        # the per-call detail in case anything needs to reconstruct it later.
        self.total_tokens_used = 0
        self.usage_log = []

        # Pause/resume bookkeeping. `_pending_turn` is the one durable piece:
        # {"response": <the raw provider response that asked for tools>,
        #  "completed": {"<index>": "<already-finished tool output>"}}, set when
        # a RunPaused unwinds through this agent and cleared when the turn is
        # picked back up. It is plain JSON-able data on purpose - state.py
        # carries it, which is what lets a run resume in a different process.
        # The other two are scratch, alive only within one turn.
        self._pending_turn: Optional[dict] = None
        self._replay_results: Optional[dict] = None
        self._partial_results: dict = {}

        # How many of one turn's tool calls may be in flight at once. A model
        # is free to ask for twenty tools in a single turn, and with delegation
        # wired as an ordinary tool that can mean twenty employees starting
        # work simultaneously - twenty provider requests, twenty conversations
        # growing, twenty charges against the budget, all before anything comes
        # back. A cap turns that into a queue.
        #
        # 8 rather than "unlimited" because unlimited is not actually a policy,
        # it just borrows one from somewhere else: sync tools run on asyncio's
        # default thread pool, which caps at min(32, cpu_count + 4), so an
        # uncapped fan-out silently inherits a limit that depends on the
        # machine it runs on. A number here is a decision; that was an
        # accident. Set it higher when the work really is I/O-bound and the
        # provider can take it, or None/0 for genuinely no cap.
        self.max_parallel_tools: Optional[int] = 8

    def reset_usage(self):
        """Zeroes the usage counters. Useful for a long-lived Agent being
        reused across many separate tasks where each task's cost should be
        measured independently."""
        self.total_tokens_used = 0
        self.usage_log = []

    def set_max_context_tokens(self, max_context_tokens: Optional[int]) -> "Agent":
        """Sets (or clears, with None) the whole-conversation token budget
        that tokens_left() counts down from.

        Deliberately a separate knob from max_tokens: max_tokens is the
        per-request *output* cap sent to the provider (Anthropic's
        max_tokens, Gemini's maxOutputTokens, ...), while max_context_tokens
        is a budget you track the whole conversation against, checked with
        tokens_used()/tokens_left() rather than enforced by the provider.
        """
        if max_context_tokens is not None:
            if not isinstance(max_context_tokens, int) or isinstance(max_context_tokens, bool) or max_context_tokens < 0:
                raise ValueError("max_context_tokens must be a non-negative integer, or None")
        self.max_context_tokens = max_context_tokens
        return self

    def tokens_used(self) -> int:
        """Estimated tokens the next request would cost: system instruction +
        conversation history + registered tool schemas, added up with
        ContextCompressor.token_estimate - the same no-external-tokenizer
        heuristic CompressionPolicy uses. This is an estimate, not an exact
        provider count (no external tokenizer is used anywhere in this
        library, on purpose - see compressor.py) - treat it as a
        budget-tracking signal, not a billing figure.
        """
        total = 0
        system_instruction = self.conversation.system_instruction
        if system_instruction:
            total += ContextCompressor.token_estimate(system_instruction)

        for msg in self.conversation.history:
            content = msg.get("content")
            if isinstance(content, str):
                total += ContextCompressor.token_estimate(content)
            elif content is not None:
                # A structured/native content block (e.g. a provider-native
                # tool-use block) - stringify it so it still counts for
                # something, rather than being silently skipped.
                total += ContextCompressor.token_estimate(json.dumps(content, default=str))
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                total += ContextCompressor.token_estimate(json.dumps(tool_calls, default=str))

        if self.tool_registry.schemas:
            schema_text = json.dumps(list(self.tool_registry.schemas.values()), default=str)
            total += ContextCompressor.token_estimate(schema_text)

        return total

    def tokens_left(self) -> Optional[int]:
        """max_context_tokens - tokens_used(), floored at 0. Returns None if
        no max_context_tokens budget is set - there is nothing to count down
        from."""
        if self.max_context_tokens is None:
            return None
        return max(0, self.max_context_tokens - self.tokens_used())

    def _extract_usage(self, res: dict) -> dict:
        """Pulls real input/output token counts out of a raw provider
        response. Every provider shapes this differently and some responses
        (e.g. a scripted test double) won't have it at all - missing fields
        default to 0 rather than raising, since usage reporting is a
        best-effort governance signal, not something that should ever break
        an actual chat turn."""
        try:
            if self.provider == "anthropic":
                u = res.get("usage") or {}
                # With prompt caching, input_tokens counts only the uncached
                # part; the cached prefix is reported separately. All of it was
                # processed, so all of it counts toward budgets.
                cache_write = u.get("cache_creation_input_tokens", 0) or 0
                cache_read = u.get("cache_read_input_tokens", 0) or 0
                out = {"input": (u.get("input_tokens", 0) or 0) + cache_write + cache_read,
                       "output": u.get("output_tokens", 0) or 0}
                if cache_write or cache_read:
                    out.update({"cache_write": cache_write, "cache_read": cache_read})
                return out
            if self.provider in ("openai", "custom"):
                u = res.get("usage") or {}
                return {"input": u.get("prompt_tokens", 0) or 0, "output": u.get("completion_tokens", 0) or 0}
            if self.provider == "gemini":
                u = res.get("usageMetadata") or {}
                return {"input": u.get("promptTokenCount", 0) or 0, "output": u.get("candidatesTokenCount", 0) or 0}
            if self.provider == "ollama":
                return {"input": res.get("prompt_eval_count", 0) or 0, "output": res.get("eval_count", 0) or 0}
        except Exception:
            pass
        return {"input": 0, "output": 0}

    def _record_usage(self, res: dict) -> None:
        usage = self._extract_usage(res)
        turn_total = usage["input"] + usage["output"]
        self.total_tokens_used += turn_total
        entry = {"input": usage["input"], "output": usage["output"], "total": turn_total}
        for key in ("cache_write", "cache_read"):
            if usage.get(key):
                entry[key] = usage[key]
        entry["model"] = self.model
        self.usage_log.append(entry)

    def cost_estimate(self, catalog: Optional[Any] = None) -> Optional[float]:
        """Rough USD cost of everything in usage_log, from the per-1K prices
        in an ApiModelCatalog (the built-in seed table by default).

        Local providers cost 0.0. Returns None if any call used a model the
        catalog has no price for - an honest "unknown" beats a confident zero.
        Cached input is priced at 10% (reads) and 125% (writes) of the input
        rate, Anthropic's standard multipliers. An estimate for "roughly what
        did that cost", not a bill: prices drift (see policy.py).
        """
        if self.provider == "ollama" or self.is_local:
            return 0.0
        if catalog is None:
            from .policy import ApiModelCatalog
            catalog = ApiModelCatalog()
        total = 0.0
        for entry in self.usage_log:
            spec = catalog.find(entry.get("model", self.model)) if hasattr(catalog, "find") else None
            if spec is None:
                return None
            cache_write = entry.get("cache_write", 0)
            cache_read = entry.get("cache_read", 0)
            plain_input = max(0, entry["input"] - cache_write - cache_read)
            total += (plain_input * spec.cost_per_1k_input
                      + cache_write * spec.cost_per_1k_input * 1.25
                      + cache_read * spec.cost_per_1k_input * 0.10
                      + entry["output"] * spec.cost_per_1k_output) / 1000.0
        return round(total, 6)

    def tool(self, python_function: Optional[Callable] = None, schema: dict = None):
        """Decorator to register a Python function as a tool.

        All three spellings work:

            @agent.tool
            def f(x: str): ...

            @agent.tool()
            def g(x: str): ...

            @agent.tool(schema={...})
            def h(x: str): ...
        """
        if python_function is None:
            def decorator(func):
                self.add_tool(func, schema)
                return func
            return decorator
        self.add_tool(python_function, schema)
        return python_function
    
    def _update_stage(self, stage: str, detail: str = "", thinking_visible: bool = True) -> None:
        """Internal helper to safely mutate system tracking state and print visible console pipelines."""
        self.thinking_stage = stage
        if not thinking_visible: return
        if detail:
            _safe_print(f"[{self.provider.upper()}] ➔ {stage}: {detail}", end="\n", flush=True,
                        file=getattr(self, "status_stream", None))
        else:
            _safe_print(f"[{self.provider.upper()}] ➔ {stage}...", end="\n", flush=True,
                        file=getattr(self, "status_stream", None))

    def pin_context(self, text: str, reason: str = "") -> dict:
        """Add a fact this agent must not lose to compaction. See
        Conversation.pin for what belongs here and what doesn't."""
        return self.conversation.pin(text, reason=reason)

    def pin_last(self, reason: str = ""):
        """Protect the most recent message from compaction."""
        return self.conversation.pin_last(reason=reason)

    def _maybe_compact_history(self):
        """Run the history compaction policy, if one is set, before building a
        request.

        At the top of generate() rather than inside the tool loop: compaction
        is threshold-triggered, and re-running it after every tool iteration
        would mean paying for it (a summarizer call, in mode="agent") several
        times inside one turn for no extra benefit.

        Assigns in place, exactly as compaction.compact_company() does and for
        the same reason - anything holding a reference to conversation.history
        must not be left pointing at a stale list.
        """
        policy = self.history_compaction
        if policy is None or getattr(policy, "mode", "off") == "off":
            return
        before = len(self.conversation.history)
        try:
            compacted = policy.compact(self.conversation.history)
        except Exception as e:
            # Same contract as mode="agent" falling back to algorithmic: a
            # compaction failure must never be a way to break a run. The
            # untouched history is still perfectly usable, just larger.
            logger.warning("history compaction failed (%r) - continuing uncompacted", e)
            return
        if compacted is self.conversation.history or len(compacted) == before:
            return
        self.conversation.history[:] = compacted
        if self.archive is not None:
            # The transcript itself is already archived message-by-message as
            # it was written, so this records only that a lossy step happened
            # and how much it removed - enough to line the live history up
            # against the archived one afterwards.
            self.archive.append("history_compacted", messages_before=before,
                                messages_after=len(compacted), mode=getattr(policy, "mode", "?"))

    def set_max_tool_iterations(self, n: int):
        """Sets the upper threshold for consecutive tool execution cycles."""
        if not isinstance(n, int) or n < 1:
            raise ValueError("max_tool_iterations must be an integer >= 1")
        self.max_tool_iterations = n

    def change_api(self, provider: str, model: Optional[str] = None, base_url: Optional[str] = None,
                    api_key: Optional[str] = None,
                    key_alias: Optional[str] = None, key_env: Optional[str] = None):
        """Re-routes active transport endpoints with fluent default model fallbacks.

        key_alias/key_env steer where a missing api_key is looked up - see
        env.resolve_api_key for the full order. They matter when several
        endpoints share a provider *protocol* but not an account: OpenRouter,
        Together, Groq and a local vLLM are all provider="openai" to this
        transport, and OPENAI_API_KEY is the wrong key for every one of them.
        """
        self.provider = provider.lower()
        self.model = model or self.default_model_for(self.provider)
        self.api_key = api_key
        self.key_alias = key_alias
        self.key_env = key_env

        if base_url:
            self.url = base_url
        elif self.provider == "gemini":
            self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        elif self.provider == "openai":
            self.url = "https://api.openai.com/v1/chat/completions"
        elif self.provider == "anthropic":
            self.url = "https://api.anthropic.com/v1/messages"
        elif self.provider == "ollama":
            self.url = ollama_base_url() + "/api/chat"
        else:
            self.url = ""

        # A local provider needs no key, so don't go looking for one and don't
        # report its absence as a problem.
        if self.provider == "ollama":
            self.api_key_source = "passed explicitly" if self.api_key else "not needed (local provider)"
        else:
            self.api_key, self.api_key_source = resolve_api_key(
                provider=self.provider, explicit=self.api_key,
                alias=key_alias, key_env=key_env,
            )
            if not self.api_key:
                # Warn, never raise. The endpoint may be a local one behind a
                # base_url that needs no key, and an Agent that refuses to be
                # constructed gives a worse error than the provider's own 401 -
                # which at least proves the request was attempted. The message
                # names every variable that was tried, because "no key" without
                # that list tells you nothing about what to set.
                logger.warning("Agent(provider=%r, model=%r): no API key - %s",
                                self.provider, self.model, self.api_key_source)

    def switch_api(self, provider: str, model: Optional[str] = None, base_url: Optional[str] = None,
                   api_key: Optional[str] = None,
                   key_alias: Optional[str] = None, key_env: Optional[str] = None) -> "Agent":
        """Alias of change_api() (the name the README has always used), and
        chainable. The conversation and the registered tools carry over."""
        self.change_api(provider=provider, model=model, base_url=base_url, api_key=api_key,
                        key_alias=key_alias, key_env=key_env)
        return self

    def add_tool(self, python_function: Callable, schema: dict = None):
        self.tool_registry.register(python_function, schema)

    def add_tools(self, functions) -> "Agent":
        """Register several tools at once - e.g. a toolkit from
        `llmadapt.toolkits` such as `filesystem_tools(".")`. Chainable."""
        for func in functions:
            self.add_tool(func)
        return self

    def _send_request(self, payload: dict, headers: dict) -> dict:
        """POSTs one request with the standard library and returns the decoded
        JSON body.

        Transient failures (RETRYABLE_STATUS, dropped connections, timeouts)
        are retried up to `max_retries` times with exponential backoff -
        1s, 2s, 4s... capped at 30s, or the server's Retry-After when it sends
        one. Anything else (401, 400, ...) fails immediately: retrying a bad
        key or a malformed request only delays the same error.
        """
        if not self.url:
            raise ValueError(f"API endpoint URL is not configured for provider '{self.provider}'. "
                             f"Please supply a valid base_url when calling change_api().")

        data_bytes = json.dumps(payload).encode("utf-8")
        attempts = getattr(self, "max_retries", 0) + 1
        timeout = getattr(self, "request_timeout", None)
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            req = urllib.request.Request(self.url, data=data_bytes, headers=headers, method="POST")
            retry_after: Optional[float] = None
            try:
                if timeout is None:
                    response_cm = urllib.request.urlopen(req)
                else:
                    response_cm = urllib.request.urlopen(req, timeout=timeout)
                with response_cm as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                last_error = RuntimeError(f"\n[{self.provider.upper()} HTTP Error {e.code}]: {body}")
                if e.code not in self.RETRYABLE_STATUS and e.code < 500:
                    raise last_error
                header = e.headers.get("Retry-After") if e.headers else None
                try:
                    retry_after = float(header) if header else None
                except ValueError:
                    retry_after = None
            except urllib.error.URLError as e:
                last_error = RuntimeError(f"\n[Network Unreachable]: Check route {self.url}. Reason: {e.reason}")
            except (TimeoutError, ConnectionError, OSError) as e:
                last_error = RuntimeError(f"\n[Network Error]: {self.url}: {e.__class__.__name__}: {e}")
            if attempt + 1 < attempts:
                delay = retry_after if retry_after is not None else min(30.0, 2.0 ** attempt)
                logger.warning("%s request failed (%s); retrying in %.1fs (attempt %d of %d)",
                               self.provider, str(last_error).strip()[:200], delay, attempt + 2, attempts)
                time.sleep(delay)
        raise last_error  # type: ignore[misc]

    def chat(self, user_input: str, custom_format_func: Callable[[list], list] = None, max_tokens: int = None,
             thinking_visible: Optional[bool] = None, spinner: bool = True) -> str:
        """The synchronous entry point, unchanged for every caller that isn't
        writing async code: hand it a string, get a string back, no event loop
        required anywhere in sight.

        It is now a wrapper over chat_async() rather than a second
        implementation. There used to be two near-identical request loops here
        - `generate` and `generate_async` differed only in how they made the
        HTTP call - and the library already treats that kind of duplication as
        a bug waiting to happen (see ReviewMode's history in company/team.py).
        One of them had to become the real one, and async is the only choice
        that can host the other: `asyncio.run()` can drive a coroutine, but no
        synchronous function can host an `await`.
        """
        return run_coroutine_blocking(
            lambda: self.chat_async(user_input, custom_format_func, max_tokens=max_tokens,
                                    thinking_visible=thinking_visible, spinner=spinner),
            what="Agent.chat()",
        )

    def chat_json(self, user_input: str, schema_hint: Optional[Any] = None, retries: int = 1,
                  **chat_kwargs: Any) -> Any:
        """Ask for a JSON answer and get it back parsed (dict, list, ...).

        `schema_hint` - a JSON-schema dict, an example object, or plain prose -
        is appended to the request so the model knows the shape you want.
        If the reply does not parse, the model is shown the parse error and
        asked again, up to `retries` more times; after that a ValueError is
        raised carrying the last raw reply. Works with every provider because
        it relies only on instructions and tolerant parsing (code fences and
        leading chatter are stripped), not on a provider-specific JSON mode.
        """
        return run_coroutine_blocking(
            lambda: self.chat_json_async(user_input, schema_hint=schema_hint, retries=retries, **chat_kwargs),
            what="Agent.chat_json()",
        )

    async def chat_json_async(self, user_input: str, schema_hint: Optional[Any] = None, retries: int = 1,
                              **chat_kwargs: Any) -> Any:
        """The awaitable form of chat_json()."""
        instruction = "Respond with only a single valid JSON value - no prose, no code fences."
        if schema_hint is not None:
            hint = schema_hint if isinstance(schema_hint, str) else json.dumps(schema_hint, indent=2)
            instruction += f"\nThe JSON must match this shape:\n{hint}"
        reply = await self.chat_async(f"{user_input}\n\n{instruction}", **chat_kwargs)
        for attempt in range(retries + 1):
            try:
                return extract_json(reply)
            except ValueError as e:
                if attempt >= retries:
                    raise ValueError(f"model did not return valid JSON after {retries + 1} attempt(s): {e}. "
                                     f"Last reply: {reply[:500]!r}") from None
                reply = await self.chat_async(
                    f"That reply was not valid JSON ({e}). Reply again with only the JSON value.",
                    **chat_kwargs)

    def generate(self, custom_format_func: Callable[[list], list] = None, max_tokens: int = None,
                 thinking_visible: Optional[bool] = None, spinner: bool = True) -> str:
        """Generate a response from the current conversation and tools.

        Synchronous wrapper over generate_async() - see chat() for why this
        direction and not the other."""
        return run_coroutine_blocking(
            lambda: self.generate_async(custom_format_func, max_tokens=max_tokens,
                                        thinking_visible=thinking_visible, spinner=spinner),
            what="Agent.generate()",
        )

    # Helper methods for generating payloads
    def gen_payload(self, history, tools, headers, max_tokens: int):
        if self.provider == "gemini":
            payload = self.gen_payload_gemini(history, tools, headers, max_tokens)
        
        elif self.provider == "anthropic":
            payload = self.gen_payload_anthropic(history, tools, headers, max_tokens)

        elif self.provider in ["openai", "custom"]:
            payload = self.gen_payload_openai_custom(history, tools, headers, max_tokens)

        elif self.provider == "ollama":
            payload = self.gen_payload_ollama(history, tools, headers, max_tokens)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Please review how to configure custom providers.")
        return _merge_extra(payload, getattr(self, "extra_params", None))

    def gen_payload_gemini(self, history, tools, headers, max_tokens: int):
        if self.api_key: headers["x-goog-api-key"] = self.api_key
        payload = {"contents": history}
        if tools: payload["tools"] = tools
        if self.conversation.system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": self.conversation.system_instruction}]}
        if max_tokens is not None:
            payload["generationConfig"] = {"maxOutputTokens": max_tokens}
        return payload
    
    # Anthropic requires max_tokens on every request. Current models think
    # before answering by default and thinking counts toward this cap, so the
    # old 4096 could be spent entirely on thinking, leaving an empty answer.
    ANTHROPIC_DEFAULT_MAX_TOKENS = 16000

    def gen_payload_anthropic(self, history, tools, headers, max_tokens: int):
        if max_tokens is None: max_tokens = self.ANTHROPIC_DEFAULT_MAX_TOKENS
        headers.update({"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        payload = {"model": self.model, "messages": history, "max_tokens": max_tokens}
        if tools: payload["tools"] = tools
        if self.conversation.system_instruction: payload["system"] = self.conversation.system_instruction
        if getattr(self, "prompt_cache", False):
            _add_anthropic_cache_breakpoints(payload)
        return payload

    def gen_payload_openai_custom(self, history, tools, headers, max_tokens: int):
        if self.api_key: 
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": history}
        if tools: 
            payload["tools"] = tools
        if max_tokens is not None:
            if self.model.startswith(("o1", "o3", "o4", "gpt-5")):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
        return payload
    
    def gen_payload_ollama(self, history, tools, headers, max_tokens: int):
        payload = {"model": self.model, "messages": history, "stream": False}
        if tools: payload["tools"] = tools
        if max_tokens is not None: payload["options"] = {"num_predict": max_tokens}
        return payload
    
    # Helper method for processing completed responses
    async def _run_tool_calls(self, calls, thinking_visible=False) -> list:
        """Runs a turn's tool calls concurrently and returns their outputs in
        call order.

        `calls` is the provider-neutral shape every process_*_response method
        normalizes to before dispatching: a list of
        {"id": str|None, "name": str, "args": dict|str}. Each provider parses
        its own response into that shape and writes the results back in its
        own format; what happens in between - invoke, compress, keep the
        stages narrated - is identical for all four, and this is the only
        place it is written.

        Because the calls run concurrently, the order in which the tools
        *start* is not guaranteed - only the order of the returned results is.

        This is the line that makes a delegation hierarchy worth having. A
        manager splitting work three ways emits three tool calls in one turn;
        run one after another they cost the sum of three round-trips, and the
        two reports not currently being waited on sit idle. Here they cost the
        slowest one.

        `return_exceptions=True` is not politeness, it is the difference
        between a sibling failing and a sibling being abandoned: without it,
        gather propagates the first exception immediately while the other
        coroutines are still running, and their results - including any work
        already charged to the budget - are dropped on the floor with nobody
        waiting on them. With it, everyone finishes and then the failures are
        dealt with:

        - a ToolControlFlow (an escalation a human declined, and later a
          pause) is re-raised, because it is a statement about the run rather
          than about one tool, and the run is what has to stop;
        - anything else becomes the same "Tool Execution Failure" string a
          crash inside execute_async() would have produced, so one bad tool
          call is something the model can see and work around, exactly as it
          was when these ran one at a time.

        Ordering is unaffected: gather returns results in the order the
        coroutines were passed, not the order they finished, so writing them
        back against `calls` stays correct.
        """
        replay = self._replay_results or {}
        limit = self.max_parallel_tools
        gate = asyncio.Semaphore(limit) if limit and limit > 0 else None

        async def _one(index, call):
            # A result carried over from before a pause is returned as-is and
            # the tool is NOT run again. That is the whole point of recording
            # it: the sibling calls that completed before somebody asked for a
            # human already had their effects, and re-running them on resume
            # would repeat those effects - a second delegation, a second write,
            # a second charge against the budget.
            if str(index) in replay:
                return replay[str(index)]
            if gate is None:
                return await _invoke(call)
            async with gate:
                return await _invoke(call)

        async def _invoke(call):
            self._update_stage("Running Tool Local Process", f"Invoking function '{call['name']}'",
                               thinking_visible=thinking_visible)
            out = await self.tool_registry.execute_async(call["name"], call["args"])
            return self.compression_policy.compress(out)

        raw = await asyncio.gather(*(_one(i, call) for i, call in enumerate(calls)),
                                   return_exceptions=True)

        control_flow = next((r for r in raw if isinstance(r, ToolControlFlow)), None)
        if control_flow is not None:
            if isinstance(control_flow, RunPaused):
                # Hand the caller everything that DID finish, keyed by position
                # rather than by tool-call id: gemini and ollama can both omit
                # ids, and the index is stable because the very same provider
                # response is what gets replayed.
                #
                # Only successes are recorded. If two calls paused at once -
                # perfectly possible now that they run concurrently - one pause
                # is surfaced and the other simply is not marked finished, so on
                # resume it runs again and asks again. That is one extra
                # question for a human rather than one unanswered question
                # treated as answered, which is the right way round for the one
                # mechanism in this library whose entire job is to stop and ask.
                self._partial_results = {
                    str(i): out for i, out in enumerate(raw) if isinstance(out, str)
                }
            raise control_flow

        results = []
        for out in raw:
            if isinstance(out, BaseException):
                out = self.compression_policy.compress(ToolRegistry._failure_text(out))
            results.append(out)
        return results

    async def process_response(self, res, thinking_visible=False):
        self._record_usage(res)  # every provider path funnels through here, sync or async

        if self.provider == "gemini":
            output = await self.process_gemini_response(res, thinking_visible=thinking_visible)
            if output is not None: return output
            return None
        
        elif self.provider == "anthropic":
            output = await self.process_anthropic_response(res, thinking_visible=thinking_visible)
            if output is not None: return output
            return None

        elif self.provider in ["openai", "custom"]:
            output = await self.process_openai_custom_response(res, thinking_visible=thinking_visible)
            if output is not None: return output
            return None

        elif self.provider == "ollama":
            output = await self.process_ollama_response(res, thinking_visible=thinking_visible)
            if output is not None: return output
            return None
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Please review how to configure custom providers.")

    async def process_gemini_response(self, res, thinking_visible=False):
        if 'candidates' not in res or not res['candidates']:
            error_msg = res.get('error', {}).get('message', 'Unknown Gemini API Error')
            raise RuntimeError(f"Gemini API empty response or error: {error_msg}")

        parts = res['candidates'][0]['content']['parts']
        function_calls = [p['functionCall'] for p in parts if 'functionCall' in p]
        text_parts = [p['text'] for p in parts if 'text' in p]

        if function_calls:
            self._update_stage("Tool asked by AI: \n", str([f"   -{fc}" for fc in function_calls]), thinking_visible=thinking_visible)
            tool_calls = [{"id": function_call.get("id", ""),
                            "function": {"name": function_call['name'], "arguments": function_call.get('args', {})}}
                            for function_call in function_calls]
            self.conversation.add_model_msg(
                text="".join(text_parts) or None,
                tool_calls=tool_calls,
                native={"role": "model", "parts": parts},
                native_provider="gemini"
            )
            calls = [{"id": fc.get("id", ""), "name": fc["name"], "args": fc.get("args", {})}
                     for fc in function_calls]
            outputs = await self._run_tool_calls(calls, thinking_visible=thinking_visible)

            for call, out_str in zip(calls, outputs):
                name, call_id = call["name"], call["id"]
                function_response = {"name": name, "response": {"result": out_str}}
                if call_id: function_response["id"] = call_id
                self.conversation.history.append({
                    "role": "tool", "name": name, "content": out_str, "tool_call_id": call_id,
                    "_native": {"role": "user", "parts": [{"functionResponse": function_response}]},
                    "_native_provider": "gemini"
                })
            self._update_stage("Tool results sent to AI", thinking_visible=thinking_visible)
            return None
        
        text = "".join(text_parts)
        self.conversation.add_model_msg(text=text)
        self._update_stage("Success", thinking_visible=thinking_visible)
        return text

    async def process_anthropic_response(self, res, thinking_visible=False):
        t_calls, final_text = [], ""
        for block in res.get("content", []):
            if block["type"] == "text": final_text += block["text"]
            elif block["type"] == "tool_use":
                t_calls.append({"id": block["id"], "function": {"name": block["name"], "arguments": block["input"]}})
        
        if t_calls:
            self._update_stage("Tool asked by AI: \n", str([f"   -{fc}" for fc in t_calls]), thinking_visible=thinking_visible)
            self.conversation.add_model_msg(
                text=final_text if final_text else None, tool_calls=t_calls,
                native={"role": "assistant", "content": res.get("content", [])}, native_provider="anthropic"
            )
            self.conversation.history[-1]["_native_model"] = self.model
            calls = [{"id": tc["id"], "name": tc["function"]["name"], "args": tc["function"]["arguments"]}
                     for tc in t_calls]
            outputs = await self._run_tool_calls(calls, thinking_visible=thinking_visible)
            tool_result_blocks = [
                {"type": "tool_result", "tool_use_id": call["id"], "content": out_str}
                for call, out_str in zip(calls, outputs)
            ]

            native_user_msg = {"role": "user", "content": tool_result_blocks}
            for tc, block in zip(t_calls, tool_result_blocks):
                self.conversation.history.append({
                    "role": "tool", "name": tc["function"]["name"], "content": block["content"], "tool_call_id": tc["id"],
                    "_native": native_user_msg, "_native_provider": "anthropic"
                })
            self._update_stage("Tool results sent to AI", thinking_visible=thinking_visible)
            return None
        
        if not final_text and res.get("stop_reason") == "max_tokens":
            raise RuntimeError(
                f"{self.model} used its whole max_tokens ({res.get('usage', {}).get('output_tokens', '?')} "
                f"output tokens) before writing an answer - usually on thinking. Raise max_tokens "
                f"(Agent(max_tokens=...) or chat(..., max_tokens=...)).")
        self.conversation.add_model_msg(text=final_text)
        self._update_stage("Success", thinking_visible=thinking_visible)
        return final_text

    async def process_openai_custom_response(self, res, thinking_visible=False):
        if not res.get('choices'):
            error_msg = res.get('error', {}).get('message', 'Unknown OpenAI API Error') \
                if isinstance(res.get('error'), dict) else str(res.get('error', 'Unknown OpenAI API Error'))
            raise RuntimeError(f"OpenAI API empty response or error: {error_msg}")

        msg = res['choices'][0]['message']
        if msg.get("tool_calls"):
            self._update_stage("Tool asked by AI: \n", str([f"   -{fc}" for fc in msg["tool_calls"]]), thinking_visible=thinking_visible)
            # Handed the provider's raw tool_calls, exactly as before -
            # add_model_msg does its own normalization (and its own JSON
            # decoding of string arguments) on the way into history. The
            # dispatch list built below is a separate, deliberately
            # provider-neutral shape and must not be substituted here.
            self.conversation.add_model_msg(text=msg.get("content"), tool_calls=msg["tool_calls"])
            # One deliberate difference from the pre-refactor loop: arguments
            # are decoded for every call up front, so a provider sending
            # malformed JSON fails before any tool in the turn has run rather
            # than half way through it. Both versions end the run with a
            # JSONDecodeError; this one does it without leaving side effects
            # from the calls that already went through.
            calls = [{
                "id": tc["id"],
                "name": tc["function"]["name"],
                "args": (json.loads(tc["function"]["arguments"])
                         if isinstance(tc["function"]["arguments"], str)
                         else tc["function"]["arguments"]),
            } for tc in msg["tool_calls"]]
            outputs = await self._run_tool_calls(calls, thinking_visible=thinking_visible)
            for call, out_str in zip(calls, outputs):
                self.conversation.add_tool_response(call["name"], out_str, call["id"])
            self._update_stage("Tool results sent to AI", thinking_visible=thinking_visible)
            return None
        
        text = msg.get("content", "")
        if not text and res['choices'][0].get("finish_reason") == "length":
            raise RuntimeError(
                f"{self.model} hit its token limit before writing an answer (finish_reason=length) - "
                f"raise max_tokens.")
        self.conversation.add_model_msg(text=text)
        self._update_stage("Success", thinking_visible=thinking_visible)
        return text    

    async def process_ollama_response(self, res, thinking_visible=False):
        msg = res.get("message", {})
        if msg.get("tool_calls"):
            self._update_stage("Tool asked by AI: \n", str([f"   -{fc}" for fc in msg["tool_calls"]]), thinking_visible=thinking_visible)
            
            sanitized_calls = []
            for tc in msg["tool_calls"]:
                func = tc.get("function") or {}
                args = func.get("arguments") or {}
                if isinstance(args, str) and args.strip():
                    try: args = json.loads(args)
                    except json.JSONDecodeError: args = {}
                elif isinstance(args, str): args = {}
                sanitized_calls.append({
                    "id": tc.get("id"),
                    "function": {"name": func.get("name"), "arguments": args}
                })

            self.conversation.add_model_msg(text=msg.get("content"), tool_calls=sanitized_calls)
            calls = [{"id": tc.get("id"), "name": tc["function"]["name"], "args": tc["function"]["arguments"]}
                     for tc in sanitized_calls]
            outputs = await self._run_tool_calls(calls, thinking_visible=thinking_visible)
            for call, out_str in zip(calls, outputs):
                self.conversation.add_tool_response(call["name"], out_str, call["id"])
            self._update_stage("Tool results sent to AI", thinking_visible=thinking_visible)
            return None
        
        text = msg.get("content", "")
        self.conversation.add_model_msg(text=text)
        self._update_stage("Success", thinking_visible=thinking_visible)
        return text

    # Other helper methods
    def change_default_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens

    def set_system(self, instruction: str) -> "Agent":
        """Chainable method to update system instructions."""
        self.conversation.change_system_instruction(instruction)
        return self

    def set_max_tokens(self, tokens: int) -> "Agent":
        """Chainable method to update output token limits."""
        self.max_tokens = tokens
        return self

    def set_compression_policy(self, policy: CompressionPolicy) -> "Agent":
        """Chainable method to change how this agent compresses tool output before it hits
        history. Pass CompressionPolicy(enabled=False) to switch it back off."""
        self.compression_policy = policy
        return self

    def set_max_parallel_tools(self, limit: Optional[int]) -> "Agent":
        """How many of a single turn's tool calls may run at once.

        Defaults to 8. `None` or 0 means no cap from this library - note that
        does not mean no cap at all, since synchronous tools still queue on
        asyncio's default thread pool (min(32, cpu_count + 4) workers), so
        "unlimited" quietly becomes "whatever this machine allows". Prefer
        naming a number.

        Raise it when the tools are I/O-bound and the far end can take the
        load; lower it to 1 to get the old strictly-sequential behaviour back
        without giving up any of the async machinery.
        """
        self.max_parallel_tools = limit
        return self

    def set_iterations(self, max_iterations: int) -> "Agent":
        """Chainable method to set tool execution limits."""
        self.set_max_tool_iterations(max_iterations)
        return self

    def with_tool(self, func: Callable, schema: dict = None) -> "Agent":
        """Chainable method to register a tool."""
        self.add_tool(func, schema)
        return self

    async def _animate_spinner(self, thinking_visible: bool):
        """Asynchronous terminal spinner running on the asyncio event loop."""
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0
        try:
            while True:
                if thinking_visible:
                    _safe_print(f"\r[{self.provider.upper()}] ➔ Thinking {spinner_frames[spinner_idx]} ", end="",
                                flush=True, file=getattr(self, "status_stream", None))
                    spinner_idx = (spinner_idx + 1) % len(spinner_frames)
                await asyncio.sleep(0.08)  # Relinquishes control cleanly to event loop
        except asyncio.CancelledError:
            # Clear spinner line cleanly when the network task completes
            _safe_print("\r" + " " * 50 + "\r", end="", flush=True, file=getattr(self, "status_stream", None))

    async def chat_async(self, user_input: str, custom_format_func: Callable[[list], list] = None, max_tokens: int = None,
                 thinking_visible: Optional[bool] = None, spinner: bool = True) -> str:
            # Resuming a paused run re-enters this agent through exactly the
            # call that paused it, carrying the same task text. The turn it was
            # in the middle of is still recorded, so this is a continuation, not
            # a new question - adding the user message again would put a
            # duplicate in the history and ask the model to start over.
            if self._pending_turn is None:
                self.conversation.add_user_msg(user_input)

            return await self.generate_async(custom_format_func, max_tokens=max_tokens,
                                             thinking_visible=thinking_visible, spinner=spinner)
    
    async def generate_async(
        self, custom_format_func: Callable[[list], list] = None, 
        max_tokens: int = None, thinking_visible: Optional[bool] = None,
        spinner: bool = True) -> str:
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if thinking_visible is None:
            thinking_visible = getattr(self, "verbose", True)
        # A spinner redraws one line with "\r"; in a log file, a pipe, or the
        # captured output of a coding agent's shell that is just noise.
        spinner = bool(spinner and thinking_visible and _stdout_is_tty(getattr(self, "status_stream", None)))
        # Off the loop: in mode="agent" this makes a full summarizer round-trip,
        # and doing that inline would stall every other coroutine sharing the loop
        # for the length of an LLM call.
        await asyncio.to_thread(self._maybe_compact_history)
        self._update_stage("initial state", thinking_visible=thinking_visible)
        
        iterations = 0 
        while True:
            iterations += 1
            if iterations > self.max_tool_iterations:
                raise RuntimeError(f"Exceeded max_tool_iterations ({self.max_tool_iterations})")

            replaying = self._pending_turn is not None
            if replaying:
                # --- 0. RESUME: this turn was already asked and answered ---
                # The provider told us which tools to run and some of them ran
                # before a human was asked. Re-sending the request would cost a
                # second call and could come back asking for something else
                # entirely, stranding the work that already happened.
                pending = self._pending_turn
                self._pending_turn = None
                self._replay_results = dict(pending.get("completed") or {})
                response = pending["response"]
            else:
                history = self.conversation.export_for(self.provider, special_format=custom_format_func,
                                                       model=self.model)
                tools = self.tool_registry.export_for(self.provider, special_format=custom_format_func)
                headers = {"Content-Type": "application/json"}

                # --- 1. COMPILE THE TARGET PAYLOAD ---
                payload = self.gen_payload(history, tools, headers, max_tokens)

                # --- 2. SEND THE REQUEST ---
                self._update_stage("Request sent to AI", f"Payload Size: {len(history)} turns", thinking_visible=thinking_visible)

                spinner_async_task = None
                if spinner:
                    spinner_async_task = asyncio.create_task(self._animate_spinner(thinking_visible))

                response = None
                try:
                    # Offloads urllib request to thread pool; returns result directly or raises exception
                    response = await asyncio.to_thread(self._send_request, payload, headers)
                except Exception as err:
                    # Preserves explicit error logging before bubbling up
                    self._update_stage("Network/API Exception", str(err), thinking_visible=thinking_visible)
                    raise err
                finally:
                    if spinner_async_task:
                        spinner_async_task.cancel()
                        try:
                            await spinner_async_task
                        except asyncio.CancelledError:
                            pass

            # --- 3. PROCESS COMPLETED RESPONSE OBJECTS ---
            # Where history stood before this turn wrote anything, so a pause
            # can put it back. process_response appends the assistant message
            # before dispatching the tools it asked for; leaving that message
            # behind with no tool results after it is a shape every provider
            # rejects, and it is also simply untrue - that turn did not happen.
            history_mark = len(self.conversation.history)
            self._partial_results = {}
            try:
                output = await self.process_response(response, thinking_visible=thinking_visible)
            except RunPaused:
                del self.conversation.history[history_mark:]
                self._pending_turn = {"response": response, "completed": dict(self._partial_results)}
                raise
            finally:
                self._replay_results = None
                self._partial_results = {}

            if output is not None: 
                return output
            continue # continue to the next iteration - left in case of changes later


def test():
    # llmadapt's own .env loader, not the third-party `dotenv` package - which
    # had no business being imported from a library whose first stated
    # convention is zero third-party dependencies. Agent resolves the key
    # itself now, so this call only makes the demo's behaviour explicit.
    from .env import load_env
    load_env()
    agent = Agent('gemini')  # default model: Agent.default_model_for('gemini')
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = agent.chat(user_input)
        print(f"Agent: {response}")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = agent.chat(user_input, thinking_visible=False, spinner=False)
        print(f"Agent: {response}")
    
    agent = Agent('ollama')
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = agent.chat(user_input)
        print(f"Agent: {response}")

if __name__ == "__main__":
    test()

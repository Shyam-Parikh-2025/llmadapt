# Tests for the "ready to use" additions to Agent and ToolRegistry: retries and
# timeouts, schema generation from richer type hints, chat_json/extract_json,
# skills= on a plain Agent, extra_params, switch_api, the tool() decorator,
# verbosity, OLLAMA_HOST and Anthropic thinking blocks across a model switch.
# Same style as the rest of the suite: plain asserts + prints, fully offline.
import enum
import io
import json
import os
import sys
import urllib.error
from typing import Dict, List, Literal, Optional

import llmadapt.core as core
from llmadapt import Agent, ToolRegistry, extract_json


class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        return self.responses.pop(0)


def quiet(provider="anthropic", **kw):
    kw.setdefault("api_key", "k")
    if provider in ("ollama", "custom"):
        kw.setdefault("base_url", "http://localhost/none")
    return Agent(provider=provider, model=kw.pop("model", "m"), verbose=False, **kw)


# --- 1. schema generation understands real type hints ------------------------
class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


SEEN = {}


def paint(path: str, colors: List[str], mode: Literal["fast", "slow"] = "fast",
          retries: Optional[int] = None, meta: Dict[str, int] = None, color: Color = Color.RED) -> str:
    """Paint a thing.

    Args:
        path: Where to paint.
        colors: Which colours, in order.
        mode: How carefully.
    """
    SEEN["color"] = color
    return "ok"


reg = ToolRegistry()
reg.register(paint)
props = reg.schemas["paint"]["parameters"]["properties"]
assert props["path"] == {"type": "string", "description": "Where to paint."}, props["path"]
assert props["colors"]["type"] == "array" and props["colors"]["items"] == {"type": "string"}
assert props["mode"]["enum"] == ["fast", "slow"] and props["mode"]["type"] == "string"
assert props["retries"]["type"] == "integer"
assert props["meta"]["type"] == "object"
assert props["color"]["enum"] == ["red", "blue"]
assert props["mode"]["description"] == "How carefully."
assert props["meta"]["description"] == "The meta parameter"  # undocumented -> old fallback
assert reg.schemas["paint"]["parameters"]["required"] == ["path", "colors"]
print("PASS: schemas cover list/Literal/Optional/dict/Enum and read Args: docs")

assert reg.execute("paint", {"path": "p", "colors": ["a"], "color": "blue"}) == "ok"
assert SEEN["color"] is Color.BLUE
print("PASS: an Enum argument arrives as the Enum member, not the raw string")


def sphinx(a: int, b: "list[int]" = None) -> str:
    """Add things.

    :param a: the first
    :param list b: the rest
    """
    return "x"


reg.register(sphinx)
sp = reg.schemas["sphinx"]["parameters"]["properties"]
assert sp["a"] == {"type": "integer", "description": "the first"}, sp
assert sp["b"]["type"] == "array" and sp["b"]["description"] == "the rest", sp
print("PASS: Sphinx-style :param lines are read too")


# --- 2. the tool() decorator works with and without parentheses --------------
agent = quiet()


@agent.tool
def one(x: str) -> str:
    """One."""
    return x


@agent.tool()
def two(x: str) -> str:
    """Two."""
    return x


assert {"one", "two"} <= set(agent.tool_registry.schemas)
assert one("a") == "a" and two("b") == "b"
print("PASS: @agent.tool and @agent.tool() both register and return the function")


# --- 3. skills/personality/tools on a plain Agent ----------------------------
agent = quiet(system_instruction="Base.", skills=["python", "testing"], personality=None, tools=[one])
si = agent.conversation.system_instruction
assert si.startswith("Base.") and "## Your skills" in si and "**python**" in si and "**testing**" in si
assert "## Hard constraints" in si
assert "one" in agent.tool_registry.schemas
try:
    quiet(skills=["pyhton"])
    assert False, "a typo'd skill must not silently produce an agent with no skills"
except KeyError as e:
    assert "python" in str(e)
print("PASS: Agent(skills=..., tools=...) composes the system instruction and registers tools")


# --- 4. extra_params reach the request body, merging nested dicts -------------
agent = quiet("ollama", max_tokens=50, extra_params={"options": {"temperature": 0}, "keep_alive": "5m"})
agent._send_request = FakeResponder([{"message": {"content": "hi"}}])
agent.chat("x")
body = agent._send_request.calls[0]
assert body["options"] == {"num_predict": 50, "temperature": 0}, body["options"]
assert body["keep_alive"] == "5m"
agent = quiet("openai", extra_params={"reasoning_effort": "low", "temperature": 0.2})
agent._send_request = FakeResponder([{"choices": [{"message": {"content": "hi"}}]}])
agent.chat("x")
assert agent._send_request.calls[0]["reasoning_effort"] == "low"
print("PASS: extra_params are merged into the payload (one level deep for dicts)")


# --- 5. chat_json parses tolerant replies and re-asks on garbage --------------
assert extract_json('{"a": 1}') == {"a": 1}
assert extract_json('Here you go:\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
assert extract_json('Result: [{"a": 1}, {"b": 2}] - done') == [{"a": 1}, {"b": 2}]
try:
    extract_json("no json here")
    assert False
except ValueError:
    pass
print("PASS: extract_json handles bare, fenced and embedded JSON and rejects prose")

agent = quiet()
agent._send_request = FakeResponder([
    {"content": [{"type": "text", "text": "Sorry, here it is: {oops"}]},
    {"content": [{"type": "text", "text": '```json\n{"ok": true}\n```'}]},
])
assert agent.chat_json("give json", schema_hint={"ok": "bool"}, retries=1) == {"ok": True}
first_prompt = agent._send_request.calls[0]["messages"][0]["content"]
assert "only a single valid JSON value" in first_prompt and '"ok": "bool"' in first_prompt
assert "not valid JSON" in agent._send_request.calls[1]["messages"][-1]["content"]
agent = quiet()
agent._send_request = FakeResponder([{"content": [{"type": "text", "text": "nope"}]}])
try:
    agent.chat_json("give json", retries=0)
    assert False
except ValueError as e:
    assert "nope" in str(e)
print("PASS: chat_json retries once with the parse error, then raises with the raw reply")


# --- 6. switch_api exists (the README always used it) and keeps history -------
agent = quiet()
agent._send_request = FakeResponder([{"content": [{"type": "text", "text": "a"}]}])
agent.chat("hello")
assert agent.switch_api("openai", model="gpt-x", api_key="k2") is agent
assert agent.provider == "openai" and agent.model == "gpt-x"
assert any(m.get("content") == "hello" for m in agent.conversation.history)
print("PASS: switch_api() switches provider/model, keeps the conversation, and chains")


# --- 7. HTTP: retries transient errors, fails fast on others, sets a timeout ---
class FakeHTTPResponse:
    def __init__(self, body):
        self.body = body

    def read(self):
        return json.dumps(self.body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, retry_after=None):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    import email.message
    msg = email.message.Message()
    for k, v in headers.items():
        msg[k] = v
    return urllib.error.HTTPError("http://x", code, "err", msg, io.BytesIO(b'{"error":"busy"}'))


script = []
seen_timeouts = []
sleeps = []


def fake_urlopen(req, timeout=None):
    seen_timeouts.append(timeout)
    item = script.pop(0)
    if isinstance(item, Exception):
        raise item
    return FakeHTTPResponse(item)


real_urlopen, real_sleep = core.urllib.request.urlopen, core.time.sleep
core.urllib.request.urlopen = fake_urlopen
core.time.sleep = lambda s: sleeps.append(s)
try:
    agent = quiet(max_retries=2, request_timeout=42)
    script[:] = [http_error(529), http_error(429, retry_after=7), {"ok": 1}]
    assert agent._send_request({"a": 1}, {}) == {"ok": 1}
    assert sleeps == [1.0, 7.0], sleeps
    assert seen_timeouts == [42, 42, 42], seen_timeouts
    print("PASS: 529/429 are retried with backoff, Retry-After is honoured, the timeout is passed")

    script[:] = [http_error(401), {"never": "reached"}]
    sleeps.clear()
    try:
        agent._send_request({}, {})
        assert False
    except RuntimeError as e:
        assert "401" in str(e) and sleeps == []
    print("PASS: a 401 fails immediately - retrying a bad key only delays the same error")

    agent = quiet(max_retries=1)
    script[:] = [urllib.error.URLError("refused"), urllib.error.URLError("refused")]
    try:
        agent._send_request({}, {})
        assert False
    except RuntimeError as e:
        assert "Network Unreachable" in str(e)
    print("PASS: connection failures are retried, then reported once retries run out")
finally:
    core.urllib.request.urlopen, core.time.sleep = real_urlopen, real_sleep


# --- 8. verbosity: quiet agents print nothing, status can go to stderr --------
buf_out, buf_err = io.StringIO(), io.StringIO()
real_out, real_err = sys.stdout, sys.stderr
sys.stdout, sys.stderr = buf_out, buf_err
try:
    agent = quiet()
    agent._send_request = FakeResponder([{"content": [{"type": "text", "text": "a"}]}])
    agent.chat("x")
    loud = Agent(provider="anthropic", model="m", api_key="k", verbose=True)
    loud.status_stream = sys.stderr
    loud._send_request = FakeResponder([{"content": [{"type": "text", "text": "a"}]}])
    loud.chat("x")
finally:
    sys.stdout, sys.stderr = real_out, real_err
assert buf_out.getvalue() == "", repr(buf_out.getvalue())
assert "Request sent to AI" in buf_err.getvalue()
print("PASS: verbose=False prints nothing; status_stream redirects stage messages")

os.environ["LLMADAPT_VERBOSE"] = "0"
try:
    assert Agent(provider="anthropic", model="m", api_key="k").verbose is False
finally:
    del os.environ["LLMADAPT_VERBOSE"]
assert Agent(provider="anthropic", model="m", api_key="k").verbose is True
print("PASS: LLMADAPT_VERBOSE=0 turns status output off by default")


# --- 9. OLLAMA_HOST is honoured -----------------------------------------------
for value, expected in [(None, "http://localhost:11434"), ("0.0.0.0", "http://localhost:11434"),
                        ("myhost:1234", "http://myhost:1234"), ("https://gpu.box:8443/", "https://gpu.box:8443")]:
    if value is None:
        os.environ.pop("OLLAMA_HOST", None)
    else:
        os.environ["OLLAMA_HOST"] = value
    assert core.ollama_base_url() == expected, (value, core.ollama_base_url())
os.environ["OLLAMA_HOST"] = "myhost:1234"
assert Agent(provider="ollama").url == "http://myhost:1234/api/chat"
os.environ.pop("OLLAMA_HOST", None)
print("PASS: OLLAMA_HOST picks the Ollama endpoint, with a bind-all address mapped to localhost")


# --- 10. Anthropic thinking blocks are not replayed to a different model ------
agent = quiet(model="claude-a")
agent.add_tool(one)
agent._send_request = FakeResponder([
    {"content": [{"type": "thinking", "thinking": "hmm", "signature": "sig"},
                 {"type": "tool_use", "id": "t1", "name": "one", "input": {"x": "1"}}]},
    {"content": [{"type": "text", "text": "done"}]},
])
agent.chat("go")
replayed = agent._send_request.calls[1]["messages"][1]["content"]
assert any(b.get("type") == "thinking" for b in replayed), "same model: thinking must be replayed intact"
agent.switch_api("anthropic", model="claude-b", api_key="k")
exported = agent.conversation.export_for("anthropic", model="claude-b")
assistant_turn = [m for m in exported if m["role"] == "assistant" and isinstance(m["content"], list)][0]
assert not any(b.get("type") == "thinking" for b in assistant_turn["content"])
assert any(b.get("type") == "tool_use" for b in assistant_turn["content"])
print("PASS: thinking blocks replay to the model that wrote them, and are dropped after a model switch")


# --- 11. the safe printer survives a console that cannot encode the arrow -----
class Cp1252Stream(io.StringIO):
    def write(self, s):
        s.encode("cp1252")  # raises UnicodeEncodeError for '➔', like a Windows pipe
        return super().write(s)


stream = Cp1252Stream()
core._safe_print("[X] ➔ stage", file=stream)
assert stream.getvalue().strip() == "[X] -> stage", stream.getvalue()
print("PASS: status messages degrade to ASCII instead of crashing on a cp1252 pipe")


# --- 12. Anthropic prompt caching marks the stable prefix, on copies ----------
agent = quiet(model="claude-sonnet-5-5", system_instruction="S" * 50, prompt_cache=True)
agent.add_tool(one)
agent.add_tool(two)
agent._send_request = FakeResponder([
    {"content": [{"type": "tool_use", "id": "t1", "name": "one", "input": {"x": "1"}}],
     "usage": {"input_tokens": 10, "cache_creation_input_tokens": 1000, "output_tokens": 5}},
    {"content": [{"type": "text", "text": "done"}],
     "usage": {"input_tokens": 12, "cache_read_input_tokens": 1000, "output_tokens": 7}},
])
agent.chat("go")
first, second = agent._send_request.calls
assert first["system"] == [{"type": "text", "text": "S" * 50, "cache_control": {"type": "ephemeral"}}]
assert "cache_control" not in first["tools"][0] and first["tools"][-1]["cache_control"] == {"type": "ephemeral"}
assert first["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
assert second["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}  # tool_result block
assert not any("cache_control" in json.dumps(m) for m in agent.conversation.history), "history must stay clean"
assert agent.usage_log[0]["input"] == 1010 and agent.usage_log[0]["cache_write"] == 1000
assert agent.usage_log[1]["cache_read"] == 1000 and agent.total_tokens_used == 1010 + 5 + 1012 + 7
plain = quiet(model="m", system_instruction="S")
plain._send_request = FakeResponder([{"content": [{"type": "text", "text": "x"}]}])
plain.chat("go")
assert plain._send_request.calls[0]["system"] == "S", "caching is opt-in"
print("PASS: prompt_cache marks system/tools/last message, leaves history clean, counts cached tokens")


# --- 13. cost_estimate prices usage from the catalog, honestly -----------------
# 10 plain input + 1000 cache-write at $2/M*1.25, 5 out at $10/M; then 12 plain
# + 1000 cache-read at $2/M*0.1, 7 out at $10/M.
expected = (10 * 0.002 + 1000 * 0.002 * 1.25 + 5 * 0.010 + 12 * 0.002 + 1000 * 0.002 * 0.10 + 7 * 0.010) / 1000
assert abs(agent.cost_estimate() - expected) < 1e-9, (agent.cost_estimate(), expected)
assert plain.cost_estimate() is None, "unknown model -> None, not a confident zero"
assert Agent(provider="ollama", verbose=False).cost_estimate() == 0.0
print("PASS: cost_estimate uses catalog prices incl. cache multipliers; unknown -> None; local -> 0")


# --- 14. running out of max_tokens before answering is an error, not "" ------
agent = quiet()
agent._send_request = FakeResponder([{"content": [{"type": "thinking", "thinking": "..."}],
                                      "stop_reason": "max_tokens", "usage": {"output_tokens": 16000}}])
try:
    agent.chat("hard question")
    assert False, "an empty, truncated reply must not be returned as the answer"
except RuntimeError as e:
    assert "max_tokens" in str(e)
agent = quiet("openai")
agent._send_request = FakeResponder([{"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}])
try:
    agent.chat("x")
    assert False
except RuntimeError as e:
    assert "token limit" in str(e)
print("PASS: a reply truncated before any answer raises with advice instead of returning ''")

print("\nAll checks passed.")

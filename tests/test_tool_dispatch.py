# Test made with AI to speed up validation, same style as test_agent.py: plain
# asserts + prints, no pytest, fully offline against a FakeResponder standing
# in for the HTTP layer.
#
# Covers the shared tool-dispatch path (Agent._run_tool_calls) that the four
# process_*_response methods were collapsed onto, and the ToolControlFlow
# carve-out in ToolRegistry.execute that lets a run-level signal escape a tool
# call instead of being handed to the model as a failed-tool string.
import json

from llmadapt import Agent, ToolControlFlow, ToolRegistry
from llmadapt.compressor import CompressionPolicy


class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        return self.responses.pop(0)


CALLED = []


def alpha(x: str) -> str:
    """Record and echo."""
    CALLED.append(("alpha", x))
    return f"alpha:{x}"


def beta(x: str) -> str:
    """Record and echo."""
    CALLED.append(("beta", x))
    return f"beta:{x}"


def make_agent(provider):
    return Agent(provider=provider, model="m", api_key="test-key",
                 base_url="http://localhost/none" if provider in ("ollama", "custom") else None)


# --- 1. every provider dispatches a multi-call turn through the shared path,
#        in order, with results lined up against the calls that made them -----
#
# The four processors parse and write back differently; the invoke-and-compress
# step in the middle is now written once. These assert the seam holds for each.

PROVIDER_CASES = {
    "anthropic": (
        {"content": [
            {"type": "tool_use", "id": "c1", "name": "alpha", "input": {"x": "1"}},
            {"type": "tool_use", "id": "c2", "name": "beta", "input": {"x": "2"}},
        ]},
        {"content": [{"type": "text", "text": "done"}]},
    ),
    "openai": (
        {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "alpha", "arguments": json.dumps({"x": "1"})}},
            {"id": "c2", "type": "function",
             "function": {"name": "beta", "arguments": json.dumps({"x": "2"})}},
        ]}}]},
        {"choices": [{"message": {"content": "done"}}]},
    ),
    "ollama": (
        {"message": {"content": None, "tool_calls": [
            {"function": {"name": "alpha", "arguments": {"x": "1"}}},
            {"function": {"name": "beta", "arguments": {"x": "2"}}},
        ]}},
        {"message": {"content": "done"}},
    ),
    "gemini": (
        {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "alpha", "args": {"x": "1"}}},
            {"functionCall": {"name": "beta", "args": {"x": "2"}}},
        ]}}]},
        {"candidates": [{"content": {"parts": [{"text": "done"}]}}]},
    ),
}

for provider, responses in PROVIDER_CASES.items():
    CALLED.clear()
    agent = make_agent(provider)
    agent.add_tool(alpha)
    agent.add_tool(beta)
    agent._send_request = FakeResponder(list(responses))
    assert agent.chat("go") == "done", provider

    # Tool calls in one turn run concurrently (Agent._run_tool_calls gathers
    # them), so the order they *start* in is up to the scheduler - on Windows
    # and Linux it differs. What must hold is that each ran exactly once, and
    # (checked below) that results come back paired with the call that made them.
    assert sorted(CALLED) == [("alpha", "1"), ("beta", "2")], f"{provider}: {CALLED}"

    tool_msgs = [m for m in agent.conversation.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 2, f"{provider}: {len(tool_msgs)} tool messages"
    # Results must be paired with the call that produced them - a zip() over
    # two lists is exactly where an off-by-one would hide.
    assert tool_msgs[0]["content"] == "alpha:1", f"{provider}: {tool_msgs[0]}"
    assert tool_msgs[1]["content"] == "beta:2", f"{provider}: {tool_msgs[1]}"
    assert tool_msgs[0]["name"] == "alpha" and tool_msgs[1]["name"] == "beta", provider
print("PASS: all four providers dispatch multi-call turns through the shared path, results in call order")


# --- 2. history keeps add_model_msg's normalized shape, not the dispatch one -
# OpenAI sends arguments as a JSON *string*. Two separate things decode it: the
# dispatch list built in process_openai_custom_response, and add_model_msg on
# the way into history. The stored copy must keep add_model_msg's shape
# ({"id", "type", "function": {...}}) - substituting the provider-neutral
# dispatch shape here would quietly change what gets replayed to the API.
agent = make_agent("openai")
agent.add_tool(alpha)
agent._send_request = FakeResponder([
    {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "alpha", "arguments": json.dumps({"x": "9"})}},
    ]}}]},
    {"choices": [{"message": {"content": "ok"}}]},
])
agent.chat("go")
stored = [m for m in agent.conversation.history if m.get("tool_calls")][0]["tool_calls"][0]
assert set(stored) == {"id", "type", "function"}, stored
assert stored["id"] == "c1" and stored["type"] == "function"
assert stored["function"] == {"name": "alpha", "arguments": {"x": "9"}}, stored
assert "name" not in stored and "args" not in stored, f"dispatch shape leaked into history: {stored}"
print("PASS: history keeps add_model_msg's normalized tool_calls shape, not the dispatch shape")


# --- 3. compression is applied to every result, once ------------------------
agent = make_agent("anthropic")
agent.add_tool(alpha)
agent.set_compression_policy(CompressionPolicy(enabled=True, max_chars=200))
raw_arg = "z" * 4000
agent._send_request = FakeResponder([
    {"content": [{"type": "tool_use", "id": "c1", "name": "alpha",
                  "input": {"x": raw_arg}}]},
    {"content": [{"type": "text", "text": "ok"}]},
])
agent.chat("go")
compressed = [m for m in agent.conversation.history if m.get("role") == "tool"][0]["content"]
assert len(compressed) < len(f"alpha:{raw_arg}"), repr(compressed[:80])
assert len(compressed) <= 400, f"compression policy did not bite: {len(compressed)} chars"
print("PASS: the shared path still runs each result through the compression policy")


# --- 4. an ordinary tool crash is still a message for the model -------------
registry = ToolRegistry()


def explodes() -> str:
    """Always raises."""
    raise ValueError("boom")


registry.register(explodes)
out = registry.execute("explodes", {})
assert "Tool Execution Failure (ValueError)" in out and "boom" in out, out
assert registry.execute("nope", {}).startswith("Error: Tool 'nope' is not registered")
print("PASS: an ordinary tool exception is still converted to a model-visible string")


# --- 5. ...but a ToolControlFlow subclass escapes intact --------------------
class Halt(ToolControlFlow):
    pass


def halts() -> str:
    """Raises a control-flow signal."""
    raise Halt("stop the run")


registry.register(halts)
try:
    registry.execute("halts", {})
    assert False, "ToolControlFlow should not have been swallowed"
except Halt as e:
    assert str(e) == "stop the run"
print("PASS: a ToolControlFlow subclass is re-raised by ToolRegistry.execute, not stringified")


# --- 6. and it escapes through a whole Agent turn, not just the registry ----
agent = make_agent("anthropic")
agent.add_tool(halts)
agent._send_request = FakeResponder([
    {"content": [{"type": "tool_use", "id": "c1", "name": "halts", "input": {}}]},
    {"content": [{"type": "text", "text": "should never be reached"}]},
])
try:
    agent.chat("go")
    assert False, "ToolControlFlow should have propagated out of Agent.chat"
except Halt:
    pass
print("PASS: a ToolControlFlow signal propagates out of Agent.chat instead of continuing the turn")

print("\nAll checks passed.")

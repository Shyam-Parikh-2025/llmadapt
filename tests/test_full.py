import json
from llmadapt import Agent, ToolRegistry, Conversation

class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        return self.responses.pop(0)

def get_weather(city: str) -> dict:
    """Look up the weather."""
    return {"city": city, "tempF": 72}

def book_flight(dest: str) -> dict:
    """Book a flight."""
    return {"status": "booked", "dest": dest}

def flaky(x: int) -> int:
    """Always raises."""
    raise ValueError("boom")


# ---- 1. Anthropic parallel tool calls -> ONE user message (regression) ----
agent = Agent(provider="anthropic", api_key="k")
agent.add_tool(get_weather); agent.add_tool(book_flight)
responder = FakeResponder([
    {"content": [
        {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "NYC"}},
        {"type": "tool_use", "id": "c2", "name": "book_flight", "input": {"dest": "LAX"}},
    ]},
    {"content": [{"type": "text", "text": "All done!"}]},
])
agent._send_request = responder
assert agent.chat("weather and flight please") == "All done!"
second = responder.calls[1]
user_tool_msgs = [m for m in second["messages"] if m["role"]=="user" and isinstance(m["content"], list)
                  and any(b.get("type")=="tool_result" for b in m["content"])]
assert len(user_tool_msgs) == 1, user_tool_msgs
print("PASS 1: anthropic parallel tool calls bundle correctly")


# ---- 2. Default model fallback (NEW feature) ----
import os as _os
for _var in ("LLMADAPT_ANTHROPIC_MODEL", "LLMADAPT_GEMINI_MODEL"):
    _os.environ.pop(_var, None)
agent = Agent(provider="anthropic", api_key="k")  # no model given
assert agent.model == Agent.DEFAULT_MODELS["anthropic"] == "claude-sonnet-5-5", agent.model
agent2 = Agent(provider="gemini", api_key="k")
assert agent2.model == Agent.DEFAULT_MODELS["gemini"] == "gemini-3.6-flash", agent2.model
print("PASS 2: default model fallback works per provider")

# The defaults go stale as providers retire models, so each can be overridden
# from the environment without touching code - and model= still beats both.
_os.environ["LLMADAPT_ANTHROPIC_MODEL"] = "claude-haiku-5-5"
try:
    assert Agent(provider="anthropic", api_key="k").model == "claude-haiku-5-5"
    assert Agent(provider="anthropic", api_key="k", model="explicit").model == "explicit"
finally:
    del _os.environ["LLMADAPT_ANTHROPIC_MODEL"]
print("PASS 2b: LLMADAPT_<PROVIDER>_MODEL overrides the default, model= overrides that")


# ---- 3. Tool exception -> string error, NOT a crash (NEW feature) ----
agent = Agent(provider="anthropic", api_key="k")
agent.add_tool(flaky)
responder = FakeResponder([
    {"content": [{"type": "tool_use", "id": "c1", "name": "flaky", "input": {"x": 1}}]},
    {"content": [{"type": "text", "text": "Sorry, error."}]},
])
agent._send_request = responder
result = agent.chat("run flaky")
assert result == "Sorry, error."
tool_result = next(m for m in responder.calls[1]["messages"] if m["role"]=="user"
                    and isinstance(m["content"], list))["content"][0]
assert "Tool Execution Failure" in tool_result["content"]
assert "boom" in tool_result["content"]
print("PASS 3: tool exceptions become error strings, chat() doesn't crash")


# ---- 4. Unregistered tool -> string error, not exception ----
registry = ToolRegistry()
out = registry.execute("nonexistent", {})
assert "not registered" in out
print("PASS 4: missing tool returns error string")


# ---- 5. max_tool_iterations default is 6 now (changed from 10) ----
agent = Agent(provider="anthropic", api_key="k")
assert agent.max_tool_iterations == 6, agent.max_tool_iterations
print("PASS 5: max_tool_iterations defaults to 6")


# ---- 6. Anthropic max_tokens: explicit overrides default; default is 4096 ----
agent = Agent(provider="anthropic", api_key="k")
responder = FakeResponder([{"content": [{"type": "text", "text": "hi"}]}])
agent._send_request = responder
agent.chat("hello")
assert responder.calls[0]["max_tokens"] == 4096
print("PASS 6a: anthropic max_tokens defaults to 4096 when unset")

agent2 = Agent(provider="anthropic", api_key="k", max_tokens=100)
responder2 = FakeResponder([{"content": [{"type": "text", "text": "hi"}]}])
agent2._send_request = responder2
agent2.chat("hello")
assert responder2.calls[0]["max_tokens"] == 100
print("PASS 6b: anthropic max_tokens respects Agent(max_tokens=...)")

agent3 = Agent(provider="anthropic", api_key="k", max_tokens=100)
responder3 = FakeResponder([{"content": [{"type": "text", "text": "hi"}]}])
agent3._send_request = responder3
agent3.chat("hello", max_tokens=50)
assert responder3.calls[0]["max_tokens"] == 50
print("PASS 6c: per-call max_tokens overrides the agent default")


# ---- 7. OpenAI/custom: tool_calls arguments must be STRINGIFIED JSON on export ----
# This matters because OpenAI's real API requires tool_calls[].function.arguments
# to be a JSON string, not a raw object -- a real API would reject a dict here.
conv = Conversation()
conv.add_user_msg("what's the weather")
conv.add_model_msg(text=None, tool_calls=[{"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "NYC"}}}])
exported = conv.export_for("openai")
tc = exported[-1]["tool_calls"][0]
assert isinstance(tc["function"]["arguments"], str), f"expected str, got {type(tc['function']['arguments'])}"
parsed_back = json.loads(tc["function"]["arguments"])
assert parsed_back == {"city": "NYC"}
print("PASS 7: OpenAI export stringifies tool_calls arguments correctly")


# ---- 8. Gemini: missing/empty candidates raises a clear RuntimeError (NEW) ----
agent = Agent(provider="gemini", api_key="k")
responder = FakeResponder([{"candidates": []}])
agent._send_request = responder
try:
    agent.chat("hello")
    assert False, "expected RuntimeError"
except RuntimeError as e:
    assert "Gemini API" in str(e)
print("PASS 8: empty Gemini response raises a clear error")


# ---- 9. OpenAI: missing/empty choices raises a clear RuntimeError (NEW) ----
agent = Agent(provider="openai", api_key="k")
responder = FakeResponder([{"choices": []}])
agent._send_request = responder
try:
    agent.chat("hello")
    assert False, "expected RuntimeError"
except RuntimeError as e:
    assert "OpenAI API" in str(e)
print("PASS 9: empty OpenAI response raises a clear error")


# ---- 10. change_model_role + Gemini export (NEW: configurable model role) ----
conv = Conversation()
conv.change_model_role("model")  # e.g. if someone wants role="model" stored directly
conv.add_user_msg("hi")
conv.add_model_msg(text="hello there")
exported = conv.export_for("gemini")
model_msgs = [m for m in exported if m["role"] == "model"]
assert len(model_msgs) == 1, exported
print("PASS 10: change_model_role + gemini export works")


# ---- 11. change_system_instruction updates existing system message ----
conv = Conversation(system_instruction="be nice")
conv.change_system_instruction("be mean")
assert conv.history[0]["content"] == "be mean"
assert conv.system_instruction == "be mean"
print("PASS 11: change_system_instruction updates in place")


# ---- 12. set_max_tool_iterations still validates input ----
agent = Agent(provider="anthropic", api_key="k")
try:
    agent.set_max_tool_iterations(0)
    assert False
except ValueError:
    pass
print("PASS 12: set_max_tool_iterations rejects invalid values")


# ---- 13. Two sequential Anthropic round trips stay valid (regression) ----
agent = Agent(provider="anthropic", api_key="k")
agent.add_tool(get_weather); agent.add_tool(book_flight)
responder = FakeResponder([
    {"content": [{"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "NYC"}}]},
    {"content": [{"type": "text", "text": "Sunny."}]},
    {"content": [{"type": "tool_use", "id": "c2", "name": "book_flight", "input": {"dest": "LAX"}}]},
    {"content": [{"type": "text", "text": "Booked!"}]},
])
agent._send_request = responder
assert agent.chat("weather in NYC?") == "Sunny."
assert agent.chat("now book LAX") == "Booked!"
final = responder.calls[-1]
roles = [m["role"] for m in final["messages"]]
for i in range(len(roles)-1):
    assert roles[i] != roles[i+1], f"consecutive same roles: {roles}"
print("PASS 13: two sequential anthropic round trips stay valid")

print("\nAll checks passed.")


# ---- 18. Ollama: tool_call_id is preserved and survives a provider switch ----
def get_weather_ollama(city: str) -> dict:
    """Look up the weather."""
    return {"city": city, "tempF": 72}

agent = Agent(provider="ollama")
agent.add_tool(get_weather_ollama)
responder = FakeResponder([
    {"message": {"tool_calls": [{"id": "call_xyz", "function": {"name": "get_weather_ollama", "arguments": {"city": "NYC"}}}]}},
    {"message": {"content": "sunny"}},
])
agent._send_request = responder
agent.chat("weather?")
tool_msg = [m for m in agent.conversation.history if m["role"] == "tool"][-1]
assert tool_msg.get("tool_call_id") == "call_xyz", f"tool_call_id missing or wrong: {tool_msg}"

# switching providers mid-conversation must still produce a valid OpenAI payload
agent.change_api(provider="openai", api_key="fake")
exported = agent.conversation.export_for("openai")
tool_export = next(m for m in exported if m["role"] == "tool")
assert tool_export.get("tool_call_id") == "call_xyz"
print("PASS 18: Ollama tool_call_id is preserved and survives a provider switch")


# ---- 19. Ollama: assistant text alongside a tool call is preserved, not dropped ----
agent = Agent(provider="ollama")
agent.add_tool(get_weather_ollama)
responder = FakeResponder([
    {"message": {"content": "Let me check that for you.",
                 "tool_calls": [{"id": "c1", "function": {"name": "get_weather_ollama", "arguments": {"city": "NYC"}}}]}},
    {"message": {"content": "It's sunny."}},
])
agent._send_request = responder
agent.chat("weather?")
assistant_msg = next(m for m in agent.conversation.history if m["role"] == "assistant")
assert assistant_msg.get("content") == "Let me check that for you.", f"text was dropped: {assistant_msg}"
print("PASS 19: Ollama preserves assistant text alongside tool calls")

print("\nAll Ollama-fix checks passed.")
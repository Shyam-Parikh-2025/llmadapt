# Tests for llmadapt.mcp_server - the MCP (JSON-RPC over stdio) server. The
# delegated model is a fake Agent factory, so this is fully offline. The last
# section also launches the real `python -m llmadapt mcp` subprocess and talks
# to it over pipes, the way Claude Code would.
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

from llmadapt import Agent
from llmadapt.mcp_server import LlmadaptMCPServer


class FakeResponder:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def __call__(self, payload, headers):
        self.calls.append(payload)
        return {"content": [{"type": "text", "text": self.text}], "usage": {"input_tokens": 5, "output_tokens": 3}}


CREATED = []


def fake_factory(reply):
    def factory(**kwargs):
        agent = Agent(**{**kwargs, "provider": "anthropic", "api_key": "k"})
        agent._send_request = FakeResponder(reply)
        CREATED.append((kwargs, agent))
        return agent
    return factory


root = tempfile.mkdtemp(prefix="llmadapt_mcp_")
try:
    with open(os.path.join(root, "app.py"), "w", encoding="utf-8") as f:
        f.write("def add(a: int, b: int) -> int:\n    \"\"\"Add.\"\"\"\n    return a - b\n")
    with open(os.path.join(root, "build.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(["warming up"] * 200 + [f"step {i}" for i in range(500)]))

    server = LlmadaptMCPServer(provider="anthropic", model="fake-model", root=root, toolkits=["fs"],
                               agent_factory=fake_factory("the answer"))

    def rpc(method, params=None, msg_id=1):
        message = {"jsonrpc": "2.0", "method": method}
        if msg_id is not None:
            message["id"] = msg_id
        if params is not None:
            message["params"] = params
        return server.handle(message)

    init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "test", "version": "0"}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["serverInfo"]["name"] == "llmadapt"
    assert "tools" in init["result"]["capabilities"] and init["result"]["instructions"]
    assert rpc("notifications/initialized", msg_id=None) is None
    assert rpc("ping")["result"] == {}
    print("PASS: initialize echoes the protocol version, notifications get no reply, ping works")

    tools = {t["name"]: t for t in rpc("tools/list")["result"]["tools"]}
    assert set(tools) == {"ask_model", "review_code", "outline_code", "compress_file", "list_skills"}
    for tool in tools.values():
        assert tool["description"] and tool["inputSchema"]["type"] == "object"
    assert "python" in tools["ask_model"]["inputSchema"]["properties"]["skills"]["items"]["enum"]
    assert tools["ask_model"]["inputSchema"]["properties"]["toolkits"]["items"]["enum"] == ["fs"]
    print("PASS: tools/list describes all five tools with JSON schemas, skills and toolkits as enums")

    def call(name, **arguments):
        response = rpc("tools/call", {"name": name, "arguments": arguments})
        result = response["result"]
        return result["content"][0]["text"], result["isError"]

    text, is_error = call("ask_model", prompt="Explain", files=["app.py"], skills=["python"], toolkits=["fs"])
    assert not is_error and text.startswith("the answer") and "anthropic/fake-model, 8 tokens" in text, text
    kwargs, agent = CREATED[-1]
    sent = agent._send_request.calls[0]
    assert "def add" in sent["messages"][0]["content"], "files= must be appended to the prompt"
    assert "**python**" in sent["system"]
    assert "read_file" in agent.tool_registry.schemas
    assert kwargs["verbose"] is False and agent.status_stream is sys.stderr
    print("PASS: ask_model appends files, applies skills and enabled toolkits, and reports usage")

    text, is_error = call("ask_model", prompt="x", toolkits=["shell"])
    assert is_error and "not enabled" in text
    text, is_error = call("ask_model", prompt="x", files=["../../etc/passwd"])
    assert is_error and "outside the allowed root" in text
    print("PASS: toolkits not enabled at startup and paths outside the root are refused as tool errors")

    server._agent_factory = fake_factory('{"bugs": 1}')
    text, is_error = call("ask_model", prompt="count bugs", json_output=True)
    assert not is_error and json.loads(text.split("\n\n[llmadapt")[0]) == {"bugs": 1}
    print("PASS: json_output returns parsed, pretty JSON")

    server._agent_factory = fake_factory("Line 3 subtracts instead of adding.")
    text, is_error = call("review_code", path="app.py", focus="correctness")
    assert not is_error and "subtracts" in text
    sent = CREATED[-1][1]._send_request.calls[0]
    assert "correctness" in sent["messages"][0]["content"] and "**code-review**" in sent["system"]
    text, is_error = call("review_code")
    assert is_error and "path" in text
    print("PASS: review_code reviews a file with the code-review skill and validates its input")

    text, is_error = call("outline_code", path="app.py")
    assert not is_error and "def add(a: int, b: int) -> int" in text and "L1" in text
    text, is_error = call("compress_file", path="build.log", max_chars=800)
    assert not is_error and len(text) < 1500 and "step 499" in text, len(text)
    text, is_error = call("list_skills")
    assert "code-review:" in text
    print("PASS: outline_code, compress_file and list_skills work without any LLM call")

    text, is_error = call("nope")
    assert is_error and "unknown tool" in text
    text, is_error = call("outline_code", wrong_arg="x")
    assert is_error and "bad arguments" in text
    assert rpc("does/not/exist")["error"]["code"] == -32601
    print("PASS: unknown tools, bad arguments and unknown methods produce proper errors")

    # Over a byte stream, as on real stdio: one JSON object per line out.
    stdin = io.BytesIO(b'{"jsonrpc":"2.0","id":7,"method":"ping"}\nnot json\n\n'
                       b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')

    class Out:
        def __init__(self):
            self.buffer = io.BytesIO()

    out = Out()
    server.serve_forever(stdin=stdin, stdout=out)
    lines = out.buffer.getvalue().decode().splitlines()
    assert json.loads(lines[0]) == {"jsonrpc": "2.0", "id": 7, "result": {}}
    assert json.loads(lines[1])["error"]["code"] == -32700
    assert len(lines) == 2
    print("PASS: serve_forever answers line by line, reports parse errors, stays silent for notifications")

    # The real thing: a subprocess speaking MCP over its stdin/stdout.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src") \
        + os.pathsep + env.get("PYTHONPATH", "")
    env["LLMADAPT_PROVIDER"] = "anthropic"  # never probes or calls anything in this test
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "outline_code", "arguments": {"path": "app.py"}}},
    ]
    proc = subprocess.run([sys.executable, "-m", "llmadapt", "mcp", "--root", root],
                          input="\n".join(json.dumps(m) for m in messages) + "\n",
                          capture_output=True, text=True, timeout=60, env=env)
    out_lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert [m["id"] for m in out_lines] == [1, 2, 3], (proc.stdout, proc.stderr)
    assert "def add" in out_lines[2]["result"]["content"][0]["text"]
    assert "llmadapt MCP server" in proc.stderr, "the startup banner must go to stderr, never stdout"
    print("PASS: `python -m llmadapt mcp` speaks clean JSON-RPC on stdout, logs only to stderr")
finally:
    shutil.rmtree(root, ignore_errors=True)

print("\nAll checks passed.")

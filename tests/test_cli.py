# Tests for the `llmadapt` command line. Runs `python -m llmadapt ...` in
# subprocesses. The `ask` tests talk to a tiny fake Ollama server on
# 127.0.0.1 (http.server in a thread), so everything stays offline.
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
REQUESTS = []


class FakeOllama(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "tiny:1b"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append(payload)
        last = payload["messages"][-1]["content"]
        if "json" in last.lower():
            content = 'Sure: {"files": 2}'
        else:
            content = f"ANSWER({len(last)})"
        self._send({"message": {"role": "assistant", "content": content},
                    "prompt_eval_count": 11, "eval_count": 4})


server = http.server.HTTPServer(("127.0.0.1", 0), FakeOllama)
threading.Thread(target=server.serve_forever, daemon=True).start()
port = server.server_address[1]

env = dict(os.environ)
env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
env["PYTHONIOENCODING"] = "utf-8"
for var in ("LLMADAPT_PROVIDER", "LLMADAPT_OLLAMA_MODEL"):
    env.pop(var, None)


def run(*args, stdin=None, extra_env=None):
    e = dict(env, **(extra_env or {}))
    return subprocess.run([sys.executable, "-m", "llmadapt", *args], input=stdin, capture_output=True,
                          text=True, timeout=120, env=e, encoding="utf-8")


tmp = tempfile.mkdtemp(prefix="llmadapt_cli_")
try:
    proc = run("--version")
    assert proc.returncode == 0 and proc.stdout.startswith("llmadapt "), proc
    proc = run()
    assert proc.returncode == 0 and "doctor" in proc.stdout and "mcp" in proc.stdout
    print("PASS: --version and bare `llmadapt` (help) work")

    proc = run("doctor")
    assert proc.returncode == 0, proc.stderr
    assert "API keys" in proc.stdout and f"Ollama: running at http://127.0.0.1:{port}" in proc.stdout
    assert "tiny:1b" in proc.stdout and "would use: ollama / tiny:1b" in proc.stdout, proc.stdout
    print("PASS: doctor reports keys, finds the running Ollama and its models, and names the default")

    proc = run("ask", "What is this?", stdin="some piped context\n")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("ANSWER("), repr(proc.stdout)
    assert proc.stdout.count("\n") == 1, "stdout must carry only the answer"
    sent = REQUESTS[-1]
    assert sent["model"] == "tiny:1b", "default model not installed -> first installed one is used"
    assert "What is this?" in sent["messages"][-1]["content"] and "some piped context" in sent["messages"][-1]["content"]
    print("PASS: ask auto-picks the running Ollama, adds piped stdin as context, prints only the answer")

    with open(os.path.join(tmp, "x.py"), "w") as f:
        f.write("def f():\n    return 1\n")
    proc = run("ask", "review", "--file", os.path.join(tmp, "x.py"), "--skills", "code-review,python",
               "--system", "Be brief.", "--usage", "-v", "--no-stdin")
    assert proc.returncode == 0, proc.stderr
    sent = REQUESTS[-1]
    system = sent["messages"][0]
    assert system["role"] == "system" and system["content"].startswith("Be brief.") and "**code-review**" in system["content"]
    assert "def f()" in sent["messages"][-1]["content"]
    assert "tokens used: 15" in proc.stderr and "[llmadapt] ollama / tiny:1b" in proc.stderr
    assert "Request sent to AI" in proc.stderr and "Request sent" not in proc.stdout
    print("PASS: --file/--skills/--system shape the request; -v and --usage report on stderr only")

    proc = run("ask", "list files as json", "--json", "--no-stdin")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"files": 2}
    print("PASS: ask --json prints parsed, pretty JSON")

    proc = run("ask", "x", "--no-stdin", "--provider", "ollama", extra_env={"OLLAMA_HOST": "127.0.0.1:9"})
    assert proc.returncode == 1 and "llmadapt ask failed" in proc.stderr and "Is Ollama running?" in proc.stderr
    assert "Traceback" not in proc.stderr
    print("PASS: a provider that cannot be reached gives a one-line error and exit code 1, no traceback")

    proc = run("outline", os.path.join(SRC, "llmadapt", "budget.py"))
    assert proc.returncode == 0 and "class BudgetLedger" in proc.stdout and "L" in proc.stdout
    proc = run("stub", os.path.join(tmp, "x.py"))
    assert proc.returncode == 0 and "def f" in proc.stdout
    proc = run("compress", "--max-chars", "300", stdin="\n".join(["same"] * 100 + [f"n{i}" for i in range(300)]))
    assert proc.returncode == 0 and len(proc.stdout) < 1000 and "n299" in proc.stdout
    print("PASS: outline, stub and compress work offline (compress reads stdin)")

    proc = run("list", "skills")
    assert proc.returncode == 0 and "agentic-coding" in proc.stdout and "code-review" in proc.stdout
    rows = json.loads(run("list", "toolkits", "--json").stdout)
    assert {r["name"] for r in rows} >= {"fs", "fs-write", "code", "shell", "python", "web"}
    print("PASS: list shows skills and toolkits, as text or JSON")

    for name in ("a.txt", "bb.txt", "ccc.txt"):
        with open(os.path.join(tmp, name), "w") as f:
            f.write(name * 10)
    files = [os.path.join(tmp, n) for n in ("a.txt", "bb.txt", "ccc.txt")]
    before = len(REQUESTS)
    proc = run("map", "Summarize", *files, "--concurrency", "2", "--usage")
    assert proc.returncode == 0, proc.stderr
    blocks = proc.stdout.strip().split("\n\n")
    assert [b.splitlines()[0] for b in blocks] == [f"### {p}" for p in files], proc.stdout
    assert len(REQUESTS) - before == 3
    assert "free (local)" in proc.stderr and "tokens used: 45" in proc.stderr, proc.stderr
    proc = run("map", "Summarize", files[0], "--jsonl")
    row = json.loads(proc.stdout)
    assert row["file"] == files[0] and row["answer"].startswith("ANSWER(")
    proc = run("map", "Summarize", files[0], os.path.join(tmp, "nope.txt"))
    assert proc.returncode == 2 and "no such file" in proc.stderr
    print("PASS: map runs one prompt per file concurrently, keeps input order, reports usage")

    proc = run("outline", os.path.join(tmp, "missing.py"))
    assert proc.returncode == 2 and "llmadapt outline:" in proc.stderr and "Traceback" not in proc.stderr
    proc = run("ask", "x", "--no-stdin", "--skills", "pyhton")
    assert proc.returncode == 2 and "did you mean 'python'" in proc.stderr, proc.stderr
    print("PASS: user mistakes (missing file, typo'd skill) are one-line errors with exit code 2")
finally:
    server.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)

print("\nAll checks passed.")

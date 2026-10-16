# Tests for llmadapt.toolkits - the ready-made tools. Offline except for
# subprocesses of this same Python interpreter; no network.
import os
import shutil
import sys
import tempfile

from llmadapt import Agent, ToolRegistry
from llmadapt.toolkits import (
    TOOLKITS, code_tools, filesystem_tools, outline_source, python_tools, shell_tools, toolkit, web_tools,
)

root = tempfile.mkdtemp(prefix="llmadapt_toolkits_")
outside = tempfile.mkdtemp(prefix="llmadapt_outside_")
try:
    os.makedirs(os.path.join(root, "pkg"))
    os.makedirs(os.path.join(root, "node_modules", "junk"))
    with open(os.path.join(root, "pkg", "mod.py"), "w", encoding="utf-8") as f:
        f.write('"""A module."""\n\n\nclass Thing(Base):\n    """A thing."""\n\n'
                '    def go(self, n: int = 1) -> str:\n        """Go n times."""\n        return "x" * n\n\n\n'
                'async def helper(a, *args, **kw):\n    return a\n')
    with open(os.path.join(root, "notes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(f"line {i}" for i in range(1, 51)))
    with open(os.path.join(root, "node_modules", "junk", "x.py"), "w") as f:
        f.write("needle = 1\n")
    with open(os.path.join(outside, "secret.txt"), "w") as f:
        f.write("top secret")

    # --- every toolkit produces tools ToolRegistry can schema-ize ------------
    for name in TOOLKITS:
        for tool in toolkit(name, root=root):
            reg = ToolRegistry()
            reg.register(tool)
            schema = reg.schemas[tool.__name__]
            assert schema["description"] and schema["description"] != "No description available.", tool
            for pname, prop in schema["parameters"]["properties"].items():
                assert prop["description"] != f"The {pname} parameter", (tool.__name__, pname)
    print("PASS: every toolkit's tools register with real descriptions for every parameter")

    fs = {t.__name__: t for t in filesystem_tools(root)}
    assert set(fs) == {"read_file", "list_files", "search_files"}, "read-only by default"
    out = fs["read_file"]("notes.txt", start_line=10, max_lines=3)
    assert "   10  line 10" in out and "   12  line 12" in out and "line 13" not in out.split("\n")[3]
    assert "start_line=13" in out
    print("PASS: read_file numbers lines and says how to read the rest")

    listing = fs["list_files"](".", recursive=True)
    assert "pkg/mod.py" in listing and "notes.txt" in listing and "node_modules" not in listing
    assert fs["list_files"]("pkg", pattern="*.txt") == "(no matching entries)"
    print("PASS: list_files skips junk directories like node_modules")

    hits = fs["search_files"](r"def \w+", file_glob="*.py")
    assert "pkg/mod.py:7:" in hits and "node_modules" not in hits, hits
    assert fs["search_files"]("NEEDLE", ignore_case=True) == "(no matches)"  # only in skipped dir
    print("PASS: search_files greps with line numbers and respects the skip list")

    for escape in ("../" + os.path.basename(outside) + "/secret.txt", os.path.join(outside, "secret.txt")):
        try:
            fs["read_file"](escape)
            assert False, f"escaped the root via {escape}"
        except PermissionError:
            pass
    if hasattr(os, "symlink"):
        try:
            os.symlink(outside, os.path.join(root, "link"))
            try:
                fs["read_file"]("link/secret.txt")
                assert False, "escaped the root through a symlink"
            except PermissionError:
                pass
        except (OSError, NotImplementedError):
            pass  # symlinks need privileges on some Windows setups
    print("PASS: paths outside the root - relative, absolute or via symlink - are refused")

    # Through an Agent, a refusal is a tool error string for the model, not a crash.
    reg = ToolRegistry()
    reg.register(fs["read_file"])
    assert "PermissionError" in reg.execute("read_file", {"path": "../x"})
    print("PASS: a refused path reaches the model as a tool failure message")

    rw = {t.__name__: t for t in filesystem_tools(root, allow_write=True)}
    assert "write_file" in rw and "edit_file" in rw
    rw["write_file"]("new/dir/a.txt", "alpha\nbeta\n")
    assert open(os.path.join(root, "new", "dir", "a.txt")).read() == "alpha\nbeta\n"
    assert rw["edit_file"]("new/dir/a.txt", "beta", "gamma") == "edited new/dir/a.txt"
    assert "not found" in rw["edit_file"]("new/dir/a.txt", "zzz", "q")
    rw["write_file"]("dup.txt", "x x")
    assert "appears 2 times" in rw["edit_file"]("dup.txt", "x", "y")
    print("PASS: write_file/edit_file exist only when allowed, and edit_file needs a unique match")

    # --- code tools ---------------------------------------------------------
    outline = outline_source(open(os.path.join(root, "pkg", "mod.py")).read(), "mod.py")
    assert '"""A module."""' in outline
    assert "L4" in outline and "class Thing(Base)" in outline and "# A thing." in outline
    assert "def go(self, n: int=1) -> str" in outline and "# Go n times." in outline
    assert "async def helper(a, *args, **kw)" in outline
    js = "export async function load(url) {\n  return 1;\n}\nclass Store {}\nconst x = 1;\n"
    js_outline = outline_source(js, "a.js")
    assert "L1" in js_outline and "function load" in js_outline and "class Store" in js_outline
    assert "const x" not in js_outline
    code = {t.__name__: t for t in code_tools(root)}
    assert "class Thing" in code["outline_file"]("pkg/mod.py")
    assert "def go" in code["stub_file"]("pkg/mod.py")
    print("PASS: outline_source maps Python (via ast) and other languages (via declarations)")

    # --- processes ----------------------------------------------------------
    (run_python,) = python_tools(timeout=30)
    out = run_python("print(6 * 7)")
    assert out.startswith("exit code 0") and "42" in out, out
    out = run_python("import sys; sys.exit(3)")
    assert out.startswith("exit code 3"), out
    assert "timed out" in python_tools(timeout=1)[0]("import time; time.sleep(5)")
    print("PASS: run_python reports exit code and output, and enforces its timeout")

    (run_command,) = shell_tools(root, allowed_commands=("python", "python3", os.path.basename(sys.executable)))
    exe = sys.executable if " " not in sys.executable else "python"
    out = run_command(f'"{exe}" -c "print(123)"' if os.name == "nt" else f"{exe} -c 'print(123)'")
    assert "123" in out and out.startswith("exit code 0"), out
    assert "not an allowed command" in run_command("curl http://example.com")
    assert "not an allowed command" in shell_tools(root, allowed_commands=())[0]("python -V")
    assert "outside the allowed root" in ToolRegistry._failure_text(PermissionError("path '..' is outside the allowed root"))
    try:
        run_command("python -V", cwd="..")
        assert False
    except PermissionError:
        pass
    print("PASS: run_command enforces the allowlist and the root, and returns exit code + output")

    # --- web: only http(s) ----------------------------------------------------
    (fetch_url,) = web_tools()
    assert "only http" in fetch_url("file:///etc/passwd")
    print("PASS: fetch_url refuses non-http(s) URLs (no local file reads)")

    # --- by name --------------------------------------------------------------
    names = [t.__name__ for t in toolkit("fs,code", "fs-write", root=root)]
    assert names.count("read_file") == 1 and "write_file" in names and "outline_file" in names
    try:
        toolkit("filesytem", root=root)
        assert False
    except KeyError as e:
        assert "Available" in str(e)
    agent = Agent(provider="ollama", verbose=False, tools=toolkit("fs", root=root))
    assert {"read_file", "list_files", "search_files"} <= set(agent.tool_registry.schemas)
    print("PASS: toolkit() combines by name, de-duplicates, and plugs straight into Agent(tools=...)")

    try:
        filesystem_tools(os.path.join(root, "does-not-exist"))
        assert False
    except NotADirectoryError:
        pass
    print("PASS: a toolkit rooted at a missing folder fails at construction, not mid-run")
finally:
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(outside, ignore_errors=True)

print("\nAll checks passed.")

"""toolkits.py - ready-made tools an Agent can use straight away.

Skills (presets/skills.py) tell a model *how* to work; toolkits give it hands.
Each factory returns a plain list of functions, so they work anywhere a tool
does:

    from llmadapt import Agent
    from llmadapt.toolkits import filesystem_tools, code_tools

    agent = Agent("ollama", tools=filesystem_tools("./my_project") + code_tools("./my_project"))
    agent.chat("Which modules in this project touch the network?")

or by name, which is what the CLI and MCP server use:

    from llmadapt.toolkits import toolkit
    agent.add_tools(toolkit("fs", "code", root="./my_project"))

Safety, stated plainly rather than implied:

- Every file tool is confined to `root`. Paths are resolved (symlinks
  included) and anything that lands outside the root is refused.
- Writing is opt-in: `filesystem_tools(root)` is read-only, and only
  `filesystem_tools(root, allow_write=True)` (toolkit name "fs-write") adds
  write_file/edit_file.
- `shell_tools` and `python_tools` run real processes on your machine. The
  command allowlist is a guard against mistakes, NOT a sandbox: `python` or
  `git` on the allowlist can still do anything those programs can do. Give
  them to a model you would trust at your own terminal, or run the whole
  thing inside a container.

Standard library only, like the rest of llmadapt.
"""

import ast
import fnmatch
import html.parser
import os
import re
import shlex
import subprocess
import sys
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

__all__ = [
    "filesystem_tools", "code_tools", "shell_tools", "python_tools", "web_tools",
    "toolkit", "TOOLKITS", "outline_source", "DEFAULT_SHELL_COMMANDS",
]

# Directories nobody wants a model to wade through.
_SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
              ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
              ".idea", ".vscode", ".next", "target", ".gradle", "egg-info"}


def _skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or name.endswith(".egg-info")


class _Root:
    """Resolves model-supplied paths against one directory and refuses any
    that escape it."""

    def __init__(self, root: Union[str, os.PathLike]):
        self.path = os.path.realpath(os.path.abspath(os.fspath(root)))
        if not os.path.isdir(self.path):
            raise NotADirectoryError(f"toolkit root does not exist or is not a directory: {self.path}")

    def resolve(self, relative: str) -> str:
        relative = (relative or ".").strip()
        candidate = os.path.realpath(os.path.join(self.path, relative))
        try:
            inside = os.path.commonpath([candidate, self.path]) == self.path
        except ValueError:  # different drives on Windows
            inside = False
        if not inside:
            raise PermissionError(f"path {relative!r} is outside the allowed root")
        return candidate

    def show(self, absolute: str) -> str:
        rel = os.path.relpath(absolute, self.path)
        return "." if rel == "." else rel.replace(os.sep, "/")


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    if b"\x00" in raw[:4096]:
        raise ValueError("binary file - not shown")
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------


def filesystem_tools(root: Union[str, os.PathLike] = ".", allow_write: bool = False,
                     max_read_lines: int = 2000) -> List[Callable]:
    """read_file / list_files / search_files, confined to `root`; plus
    write_file / edit_file when `allow_write=True`."""
    base = _Root(root)

    def read_file(path: str, start_line: int = 1, max_lines: int = 400) -> str:
        """Read a text file and return its lines prefixed with line numbers.

        Args:
            path: File path relative to the project root.
            start_line: First line to return (1-based).
            max_lines: Most lines to return in one call; read further by calling again with a later start_line.
        """
        target = base.resolve(path)
        text = _read_text(target)
        lines = text.splitlines()
        start = max(1, int(start_line))
        count = max(1, min(int(max_lines), max_read_lines))
        chunk = lines[start - 1:start - 1 + count]
        body = "\n".join(f"{start + i:>5}  {line}" for i, line in enumerate(chunk))
        remaining = len(lines) - (start - 1 + len(chunk))
        footer = f"\n... [{remaining} more lines; call again with start_line={start + len(chunk)}]" if remaining > 0 else ""
        return f"{base.show(target)} ({len(lines)} lines)\n{body}{footer}"

    def list_files(path: str = ".", pattern: str = "*", recursive: bool = False, max_entries: int = 300) -> str:
        """List files and folders under a directory.

        Args:
            path: Directory relative to the project root.
            pattern: Glob that file names must match, e.g. "*.py".
            recursive: Also list everything in subdirectories (common junk such as .git and node_modules is skipped).
            max_entries: Stop after this many entries.
        """
        target = base.resolve(path)
        if not os.path.isdir(target):
            raise NotADirectoryError(f"{path!r} is not a directory")
        out: List[str] = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(target):
                dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
                for name in sorted(filenames):
                    if fnmatch.fnmatch(name, pattern):
                        out.append(base.show(os.path.join(dirpath, name)))
                        if len(out) >= max_entries:
                            return "\n".join(out) + f"\n... [stopped at {max_entries} entries]"
        else:
            for name in sorted(os.listdir(target)):
                full = os.path.join(target, name)
                if os.path.isdir(full):
                    out.append(base.show(full) + "/")
                elif fnmatch.fnmatch(name, pattern):
                    out.append(base.show(full))
                if len(out) >= max_entries:
                    return "\n".join(out) + f"\n... [stopped at {max_entries} entries]"
        return "\n".join(out) if out else "(no matching entries)"

    def search_files(pattern: str, path: str = ".", file_glob: str = "*", ignore_case: bool = False,
                     max_results: int = 100) -> str:
        """Search file contents with a regular expression, like grep -rn.

        Args:
            pattern: Regular expression to look for.
            path: Directory (or single file) relative to the project root.
            file_glob: Only search files whose name matches this glob, e.g. "*.py".
            ignore_case: Match case-insensitively.
            max_results: Stop after this many matching lines.
        """
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        target = base.resolve(path)
        files: Iterable[str]
        if os.path.isfile(target):
            files = [target]
        else:
            def walk():
                for dirpath, dirnames, filenames in os.walk(target):
                    dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
                    for name in sorted(filenames):
                        if fnmatch.fnmatch(name, file_glob):
                            yield os.path.join(dirpath, name)
            files = walk()
        hits: List[str] = []
        for file_path in files:
            try:
                text = _read_text(file_path)
            except (OSError, ValueError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{base.show(file_path)}:{number}: {line.strip()[:300]}")
                    if len(hits) >= max_results:
                        return "\n".join(hits) + f"\n... [stopped at {max_results} matches]"
        return "\n".join(hits) if hits else "(no matches)"

    tools: List[Callable] = [read_file, list_files, search_files]
    if not allow_write:
        return tools

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file with the given content.

        Args:
            path: File path relative to the project root. Parent folders are created.
            content: The complete new contents of the file.
        """
        target = base.resolve(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        return f"wrote {len(content)} characters to {base.show(target)}"

    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace one exact, unique piece of text in a file.

        Args:
            path: File path relative to the project root.
            old_text: Text currently in the file. Must appear exactly once - include surrounding lines to make it unique.
            new_text: The replacement text.
        """
        target = base.resolve(path)
        text = _read_text(target)
        count = text.count(old_text)
        if count == 0:
            return "Error: old_text was not found in the file - read the file again and copy the text exactly."
        if count > 1:
            return f"Error: old_text appears {count} times - include more surrounding lines so it is unique."
        with open(target, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace(old_text, new_text, 1))
        return f"edited {base.show(target)}"

    return tools + [write_file, edit_file]


# ---------------------------------------------------------------------------
# Code understanding
# ---------------------------------------------------------------------------

_NON_PY_DEF = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|async\s+|pub(?:\([^)]*\))?\s+)*"
    r"(?:function\*?|class|interface|struct|enum|trait|impl|type|fn|func|def|module|namespace)\s+[\w$<>.:]+"
)


def _py_signature(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases) if hasattr(ast, "unparse") else ""
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    if hasattr(ast, "unparse"):
        args = ast.unparse(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    else:  # pragma: no cover - Python < 3.9 has no ast.unparse
        args, returns = "...", ""
    return f"{prefix} {node.name}({args}){returns}"


def outline_source(source: str, filename: str = "") -> str:
    """A compact map of a source file: one line per class/function with its
    line number and the first line of its docstring.

    Python is parsed properly (ast); other languages get a regex pass that
    catches the usual declaration keywords (function, class, fn, func, struct,
    interface, ...). Much smaller than the file - typically 5-15% - which is
    the point: it tells a model (or a person) where to look before reading.
    """
    if filename.endswith((".py", ".pyi")) or (not filename and _looks_like_python(source)):
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"(could not parse as Python: {e})"
        lines: List[str] = []

        def visit(body, depth):
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node) or ""
                    first = doc.strip().splitlines()[0] if doc.strip() else ""
                    deco = "".join(f"@{ast.unparse(d)} " for d in node.decorator_list) if hasattr(ast, "unparse") else ""
                    line = f"L{node.lineno:<5} {'    ' * depth}{deco}{_py_signature(node)}"
                    if first:
                        line += f"  # {first[:100]}"
                    lines.append(line)
                    if isinstance(node, ast.ClassDef):
                        visit(node.body, depth + 1)

        module_doc = ast.get_docstring(tree)
        if module_doc:
            lines.append(f'"""{module_doc.strip().splitlines()[0][:120]}"""')
        visit(tree.body, 0)
        return "\n".join(lines) if lines else "(no classes or functions)"

    out = []
    for number, line in enumerate(source.splitlines(), 1):
        if _NON_PY_DEF.match(line):
            out.append(f"L{number:<5} {line.strip()[:160]}")
    return "\n".join(out) if out else "(no declarations recognised)"


def _looks_like_python(source: str) -> bool:
    return bool(re.search(r"^\s*(def|class|import|from)\s", source, re.MULTILINE))


def code_tools(root: Union[str, os.PathLike] = ".") -> List[Callable]:
    """outline_file / stub_file: read a file's shape without paying for
    every line of it. Confined to `root`."""
    from .compressor import ContextCompressor

    base = _Root(root)

    def outline_file(path: str) -> str:
        """Show the classes and functions in a source file, with line numbers and one-line docstrings.

        Args:
            path: Source file relative to the project root.
        """
        target = base.resolve(path)
        return f"{base.show(target)}\n" + outline_source(_read_text(target), filename=target)

    def stub_file(path: str) -> str:
        """Show a Python file with function bodies removed: signatures, type hints and docstrings only.

        Args:
            path: Python file relative to the project root.
        """
        target = base.resolve(path)
        return ContextCompressor.code_to_stub(_read_text(target))

    return [outline_file, stub_file]


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

# Read-mostly developer commands. A guard against slips, not a sandbox -
# see the module docstring.
DEFAULT_SHELL_COMMANDS = (
    "git", "ls", "dir", "cat", "type", "head", "tail", "wc", "grep", "rg", "find",
    "python", "python3", "py", "pytest", "pip", "uv", "node", "npm", "npx", "tsc",
    "cargo", "go", "make", "ruff", "mypy", "black",
)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    from .compressor import ContextCompressor
    return ContextCompressor.compress_tool_output(text, max_chars=limit)


def shell_tools(root: Union[str, os.PathLike] = ".",
                allowed_commands: Union[Sequence[str], str, None] = DEFAULT_SHELL_COMMANDS,
                timeout: float = 120.0, max_output_chars: int = 12000) -> List[Callable]:
    """run_command: run one program (no shell pipes or redirection) in `root`.

    `allowed_commands` lists the program names a model may start; pass "*" to
    allow anything. Not a security boundary - see the module docstring.
    """
    base = _Root(root)
    allow_any = allowed_commands == "*"
    allowed = set() if allow_any else {c.lower() for c in (allowed_commands or ())}

    def run_command(command: str, cwd: str = ".") -> str:
        """Run a single command (no pipes, redirection or &&) and return its exit code and output.

        Args:
            command: The command line, e.g. "pytest -q tests/test_x.py" or "git diff --stat".
            cwd: Working directory relative to the project root.
        """
        argv = shlex.split(command, posix=(os.name != "nt"))
        if os.name == "nt":
            # Non-POSIX shlex keeps the quote characters on each token.
            argv = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" else a for a in argv]
        if not argv:
            return "Error: empty command"
        program = os.path.splitext(os.path.basename(argv[0]))[0].lower()
        if not allow_any and program not in allowed:
            return (f"Error: '{program}' is not an allowed command. Allowed: "
                    f"{', '.join(sorted(allowed)) or '(none)'}")
        workdir = base.resolve(cwd)
        try:
            proc = subprocess.run(argv, cwd=workdir, capture_output=True, text=True,
                                  timeout=timeout, errors="replace",
                                  shell=(os.name == "nt" and program in ("dir", "type")))
        except FileNotFoundError:
            return f"Error: program not found: {argv[0]}"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout:g}s"
        output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return f"exit code {proc.returncode}\n{_clip(output, max_output_chars)}"

    return [run_command]


def python_tools(timeout: float = 60.0, cwd: Optional[Union[str, os.PathLike]] = None,
                 max_output_chars: int = 12000) -> List[Callable]:
    """run_python: execute a Python snippet in a fresh interpreter (isolated
    mode, `-I`) and return what it printed. Real code on your machine - not a
    sandbox."""
    workdir = os.fspath(cwd) if cwd is not None else None

    def run_python(code: str) -> str:
        """Run a Python 3 snippet in a fresh interpreter and return its exit code and printed output.

        Args:
            code: Complete Python source. Use print() for anything you want to see.
        """
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=workdir, capture_output=True,
                                  text=True, timeout=timeout, errors="replace")
        except subprocess.TimeoutExpired:
            return f"Error: timed out after {timeout:g}s"
        output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return f"exit code {proc.returncode}\n{_clip(output, max_output_chars)}"

    return [run_python]


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------


class _TextExtractor(html.parser.HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre"}

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1

    def handle_data(self, data):
        if not self._skipping:
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        joined = re.sub(r"[ \t\r\f\v]+", " ", joined)
        return re.sub(r"\n\s*\n+", "\n\n", joined).strip()


def web_tools(timeout: float = 20.0, max_chars: int = 20000,
              user_agent: str = "llmadapt/1.0 (+https://github.com/Shyam-Parikh-2025/llmadapt)") -> List[Callable]:
    """fetch_url: GET an http(s) page and return its readable text."""

    def fetch_url(url: str) -> str:
        """Download a web page (http or https) and return its readable text, with HTML tags stripped.

        Args:
            url: The full URL, starting with http:// or https://.
        """
        if not url.lower().startswith(("http://", "https://")):
            return "Error: only http:// and https:// URLs are allowed"
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(max_chars * 8)
        charset = "utf-8"
        match = re.search(r"charset=([\w-]+)", content_type)
        if match:
            charset = match.group(1)
        text = raw.decode(charset, errors="replace")
        if "html" in content_type.lower() or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            parser = _TextExtractor()
            parser.feed(text)
            text = parser.text()
        return _clip(text, max_chars)

    return [fetch_url]


# ---------------------------------------------------------------------------
# By name
# ---------------------------------------------------------------------------

TOOLKITS: Dict[str, Callable[..., List[Callable]]] = {
    "fs": lambda root=".", **kw: filesystem_tools(root, **kw),
    "fs-write": lambda root=".", **kw: filesystem_tools(root, allow_write=True, **kw),
    "code": lambda root=".", **kw: code_tools(root),
    "shell": lambda root=".", **kw: shell_tools(root, **kw),
    "python": lambda root=".", **kw: python_tools(cwd=root, **kw),
    "web": lambda root=".", **kw: web_tools(**kw),
}

TOOLKIT_DESCRIPTIONS: Dict[str, str] = {
    "fs": "read_file, list_files, search_files - read-only, confined to the root folder",
    "fs-write": "fs plus write_file and edit_file",
    "code": "outline_file, stub_file - a file's classes/functions without the bodies",
    "shell": "run_command - one allowlisted program at a time, in the root folder (not a sandbox)",
    "python": "run_python - run a snippet in a fresh interpreter (not a sandbox)",
    "web": "fetch_url - download a page as plain text",
}


def toolkit(*names: str, root: Union[str, os.PathLike] = ".") -> List[Callable]:
    """Tools for one or more toolkit names, de-duplicated - e.g.
    `toolkit("fs", "code", root=".")`. A comma-separated string works too
    ("fs,code"), which is how the CLI passes them through."""
    from .suggest import did_you_mean

    flat: List[str] = []
    for name in names:
        flat.extend(part.strip() for part in str(name).split(",") if part.strip())
    tools: List[Callable] = []
    seen = set()
    for name in flat:
        if name not in TOOLKITS:
            raise KeyError(f"unknown toolkit {name!r}{did_you_mean(name, TOOLKITS)}. "
                           f"Available: {', '.join(sorted(TOOLKITS))}")
        for tool in TOOLKITS[name](root):
            if tool.__name__ not in seen:
                seen.add(tool.__name__)
                tools.append(tool)
    return tools

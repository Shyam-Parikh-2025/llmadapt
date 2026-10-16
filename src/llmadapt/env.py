"""env.py - reading API keys out of the environment, and out of a `.env` file,
without a third-party dependency.

Two things were wrong before this module existed.

**The library never read `.env`.** `core.py` resolved a missing key with
`os.getenv(f"{provider.upper()}_API_KEY")`, which reads the *process*
environment only. A `.env` file sitting in the project root - the normal way
people keep a key out of their shell history - was never opened by anything in
`src/`. The only `load_dotenv()` calls were in a test scratchpad and in
`core.py`'s own demo block, and `dotenv` is a third-party package, which has no
business being imported by a library whose first stated convention is zero
third-party dependencies. So the key was there, the code looked like it should
find it, and it didn't.

**One key per provider isn't enough.** It is fine for the big three - one
account, one key, many models - and wrong the moment two endpoints share a
provider *protocol* but not an account. OpenRouter, Together, Groq, Fireworks
and a local vLLM are all `provider="openai"` as far as the transport is
concerned, and they do not share a key. Resolution therefore checks a more
specific name before falling back to the provider's.

Resolution order, most specific first:

1. an explicit `api_key=` passed to `Agent`/`hire()`/`model_map` - always wins,
   nothing here overrules what the caller stated
2. `model_map[rank]["key_env"]` - a variable name the config names outright,
   which is the escape hatch for anything the conventions below don't cover
3. `<ALIAS>_API_KEY`, where alias is `model_map[rank]["alias"]` - the name you
   give one endpoint ("openrouter", "together") when several share a provider
4. `<PROVIDER>_API_KEY` - the original behaviour, unchanged, so every existing
   config keeps working untouched

Nothing here ever logs a key. It logs *which source* a key came from, which is
the thing you actually need when the answer is "no key found" and you are sure
you set one.
"""

import logging
import os
import re
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "load_env", "env_value", "resolve_api_key", "key_env_candidates",
    "find_env_file", "parse_env_text", "DEFAULT_ENV_FILENAME",
]

DEFAULT_ENV_FILENAME = ".env"

# How far up the tree to look for a .env before giving up. Bounded so a script
# run from somewhere unexpected cannot end up reading a file from the user's
# home directory - or worse, from the filesystem root - and silently picking up
# credentials that belong to a different project.
MAX_UPWARD_LEVELS = 6

# Parsed .env contents, loaded at most once per path. os.environ is NOT
# modified: the file is a *fallback* consulted only for names the real
# environment does not define, so an exported variable always wins and nothing
# in this library can change the environment out from under the host process.
_loaded: Dict[str, Dict[str, str]] = {}
_lock = threading.Lock()
_auto_load_done = False

# KEY=value, tolerating: leading "export", spaces around the "=" (the user's own
# .env is written `GEMINI_API_KEY = ...`, which a naive split on "=" would read
# as a key named "GEMINI_API_KEY "), and quoted values.
_LINE = re.compile(r"""
    ^\s*
    (?:export\s+)?
    ([A-Za-z_][A-Za-z0-9_]*)      # key
    \s*=\s*
    (.*?)                          # value, stripped/unquoted below
    \s*$
""", re.VERBOSE)


def parse_env_text(text: str) -> Dict[str, str]:
    """Parse `.env` contents into a dict.

    Deliberately a small, predictable subset rather than an imitation of
    python-dotenv: KEY=value, `export KEY=value`, `#` comments, blank lines,
    and single or double quotes around the value. No variable interpolation and
    no multi-line values - both are features whose absence is obvious and whose
    half-working presence would be a trap.

    A line that doesn't match is skipped with a warning rather than raising. A
    malformed `.env` should cost you one key and a log line, not the ability to
    start the program.
    """
    values: Dict[str, str] = {}
    for number, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            logger.warning("env: skipping unparseable line %d of the .env file", number)
            continue
        key, value = match.group(1), match.group(2)
        # Strip a trailing comment only on an unquoted value - inside quotes a
        # '#' is part of the secret, and some keys really do contain one.
        if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key] = value
    return values


def find_env_file(start: Optional[str] = None, filename: str = DEFAULT_ENV_FILENAME) -> Optional[str]:
    """The nearest `.env` at or above `start`, or None.

    Walks upward because running a script from `tests/` or `scripts/` is normal
    and the file lives at the project root. Bounded by MAX_UPWARD_LEVELS, and
    stops at a filesystem root, so it cannot wander into a home directory and
    pick up another project's credentials.
    """
    directory = os.path.abspath(start or os.getcwd())
    for _ in range(MAX_UPWARD_LEVELS):
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:  # filesystem root
            break
        directory = parent
    return None


def load_env(path: Optional[str] = None, start: Optional[str] = None,
             force: bool = False) -> Dict[str, str]:
    """Load a `.env` into this module's fallback table and return what it read.

    Called automatically the first time a key is resolved, so the common case
    needs no setup. Explicit calls are for pointing at a specific file, or for
    re-reading one after it changed.

    **Never touches `os.environ`.** The file is consulted only for names the
    real environment does not already define, so exporting a variable in your
    shell always overrides the file, and importing llmadapt cannot change the
    environment of the program that imported it.
    """
    global _auto_load_done
    resolved = path or find_env_file(start)
    with _lock:
        if resolved is None:
            _auto_load_done = True
            return {}
        if not force and resolved in _loaded:
            return dict(_loaded[resolved])
        try:
            with open(resolved, "r", encoding="utf-8") as handle:
                values = parse_env_text(handle.read())
        except OSError as e:
            logger.warning("env: could not read %s (%s) - continuing without it", resolved, e)
            values = {}
        _loaded[resolved] = values
        _auto_load_done = True
        if values:
            # Names only. A log line naming a key's *value* is how a secret ends
            # up in a bug report.
            logger.debug("env: loaded %d name(s) from %s: %s",
                         len(values), resolved, ", ".join(sorted(values)))
        return dict(values)


def _ensure_auto_loaded() -> None:
    # Reads the flag but never assigns it - load_env() owns that - so no
    # `global` declaration is needed or wanted here.
    if not _auto_load_done:
        load_env()


def env_value(name: str) -> Optional[str]:
    """The value of `name` from the real environment, else from a loaded
    `.env`. `os.environ` always wins."""
    if not name:
        return None
    live = os.environ.get(name)
    if live:
        return live
    _ensure_auto_loaded()
    with _lock:
        for values in _loaded.values():
            found = values.get(name)
            if found:
                return found
    return None


def _normalize(text: str) -> str:
    """A model or alias turned into a legal environment-variable name:
    upper-cased, with every run of non-alphanumerics collapsed to one
    underscore. 'claude-opus-4-6' -> 'CLAUDE_OPUS_4_6'."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_").upper()


def key_env_candidates(provider: Optional[str] = None, alias: Optional[str] = None,
                       key_env: Optional[str] = None) -> List[str]:
    """The environment-variable names that would be tried, most specific first.

    Exposed rather than kept private because "which variable should I set?" is
    the question a user has when a key isn't found, and the honest answer is
    this list. `Company.key_sources()` prints it.
    """
    names: List[str] = []
    if key_env:
        names.append(str(key_env))
    if alias:
        names.append(f"{_normalize(alias)}_API_KEY")
    if provider:
        names.append(f"{_normalize(provider)}_API_KEY")
        # Google's own SDKs and docs use GOOGLE_API_KEY for the same Gemini
        # key, so a user following them has set that one, not ours.
        if _normalize(provider) == "GEMINI":
            names.append("GOOGLE_API_KEY")
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def resolve_api_key(
    provider: Optional[str] = None,
    explicit: Optional[str] = None,
    alias: Optional[str] = None,
    key_env: Optional[str] = None,
    required: bool = False,
) -> Tuple[Optional[str], str]:
    """Find the API key for one endpoint. Returns `(key, source)`.

    `source` is a human-readable description of *where* the key came from
    ("the ANTHROPIC_API_KEY environment variable", "passed explicitly") - or,
    when nothing was found, a sentence naming every variable that was tried.
    Returning the provenance rather than just the key is the point: a missing
    key is the single most common setup failure here, and "no key" on its own
    tells you nothing about which name you were supposed to set.

    Never raises for a missing key unless `required=True`. A local provider
    needs no key at all, and an Agent that is about to fail on a real request
    gives a better error than one that refused to be constructed.
    """
    if explicit:
        return explicit, "passed explicitly"

    candidates = key_env_candidates(provider, alias, key_env)
    for name in candidates:
        value = env_value(name)
        if value:
            source = f"the {name} environment variable"
            if not os.environ.get(name):
                source += " (from a .env file)"
            return value, source

    tried = ", ".join(candidates) or "(no candidate names)"
    source = f"not found - tried {tried}"
    if required:
        raise ValueError(
            f"No API key for provider {provider!r}. Set one of: {tried} - "
            f"in your environment or in a .env file, or pass api_key= directly."
        )
    return None, source

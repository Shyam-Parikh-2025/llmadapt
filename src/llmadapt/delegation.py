"""delegation.py - Phase 6: the two task-decomposition strategies.

Both live here rather than in two modules because they are the same idea at
different granularities - a more capable agent decides the *shape* of the work,
cheaper agents fill it in - and splitting them would invite two competing
decomposition systems, which is exactly what the roadmap warned against when
it folded "plan-then-execute" into this phase.

**stub-and-fill** - the architect emits a module of function signatures,
docstrings and type hints with empty bodies; implementers fill the bodies one
at a time; the results are reassembled into one module. Structured output, for
code.

**plan-then-execute** - the planner emits a numbered plan; each step is
executed in order, with earlier results available as context; an optional
synthesis pass turns the step results into one answer. Unstructured output,
for anything.

Reuse, not reinvention: `ContextCompressor.code_to_stub()` (compressor.py)
already turns real code into exactly the signature+docstring+`...` form the
architect is asked to produce. It is used here to normalize the architect's
output and to summarize already-written code as context, rather than writing a
second AST layer for the same job.

**Fixed by rank in v1.** Who architects and who implements comes from an
ordered rank preference list, not from a model deciding how deep to decompose.
The roadmap is explicit that dynamic decomposition depth is a research-adjacent
stretch goal with no clean formula - so this picks the most senior available
employee to architect and the cheapest available ones to implement, and says so
rather than pretending to be adaptive.

What this does NOT do, stated plainly:
- By default it does not execute or test the generated code. Assembly is
  checked with `compile()` for syntax only; a stub whose body is confidently
  wrong will assemble cleanly. Pass `verify=python_verifier(test_code)` (or any
  callable) to run the assembled module plus your tests in a subprocess and
  report failures on the result - opt-in, because it executes model-written
  code on your machine.
- It does not verify that an implementer honored the signature it was given.
  If the implementer renames the function, that stub is reported unfilled
  rather than silently mismatched, but a changed *parameter list* with the
  same name is accepted as-is.
- The planner's step list is parsed from prose with a numbered-list regex.
  Models are good at numbered lists and bad at guarantees; when nothing parses
  the whole task becomes a single step rather than failing.
"""

import ast
import asyncio
import copy
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .compressor import ContextCompressor
from .core import run_coroutine_blocking
from .router import RoleRank

# ---------------------------------------------------------------------------
# Who does what - fixed by rank, per the v1 decision
# ---------------------------------------------------------------------------

# Most senior first: the architect is the first rank in this list that this
# company actually has an employee for. Architecting is the leverage point -
# a bad interface makes every filled body wrong - so it goes to the most
# capable tier available.
DEFAULT_ARCHITECT_RANKS: Sequence[str] = (
    RoleRank.C_SUITE, RoleRank.GENERAL_MANAGER, RoleRank.MANAGER, RoleRank.SENIOR,
    RoleRank.JUNIOR, RoleRank.INTERN, RoleRank.VOLUNTEER,
)

# Cheapest first: bodies are the high-volume, low-leverage part, which is the
# whole economic point of splitting the work this way.
DEFAULT_IMPLEMENTER_RANKS: Sequence[str] = (
    RoleRank.INTERN, RoleRank.JUNIOR, RoleRank.VOLUNTEER, RoleRank.SENIOR,
    RoleRank.MANAGER, RoleRank.GENERAL_MANAGER, RoleRank.C_SUITE,
)

# A hard cap on how many stubs one architect pass may produce, and how many
# steps one plan may contain. Both are runaway guards in the same spirit as
# Phase 3's budget ceiling and Phase 5's max_review_rounds: each stub and each
# step is a real model call, and a model asked to decompose a vague task will
# happily produce forty of them.
MAX_STUBS = 20
MAX_PLAN_STEPS = 12

# How many lines of real logic (docstring excluded) an architect may write in a
# function before it stops counting as the "small enough to just do in one
# line" shortcut the ARCHITECT_PROMPT allows, and gets demoted back into the
# stub pool for an implementer to fill properly.
#
# The rule the architect is given is narrow on purpose - a one-line body AND no
# docstring - so this check is narrow to match. A function the architect wrote
# *with* a docstring is a deliberate helper, which extract_stubs has always
# kept verbatim in the preamble; that behaviour is unchanged. What this guards
# is the specific failure the shortcut invites: the architect claiming it and
# then writing twelve lines underneath, which is the whole job done unreviewed
# by the one agent this strategy exists to keep out of implementation.
MAX_ARCHITECT_BODY_LINES = 1


# ---------------------------------------------------------------------------
# Prompts (module-level so they are inspectable and overridable)
# ---------------------------------------------------------------------------

# The one-line shortcut below is a deliberate, bounded exception to "do NOT
# implement anything", resolved rather than left open: a trivial helper costs a
# whole implementer round-trip to fill, which is real money at the cheap tiers
# this library targets, and an implementer adds nothing to `return x + 1`. The
# trade it makes is that an architect-written body is never reviewed by a
# second agent, so two things keep it honest - extract_stubs records every one
# of them on `StubPlan.architect_implemented` (and stub_and_fill logs them, so
# they are visible in the activity log rather than indistinguishable from
# implementer work), and anything longer than MAX_ARCHITECT_BODY_LINES is
# demoted back into the stub pool instead of being taken at its word.
ARCHITECT_PROMPT = """You are the architect. Do NOT implement anything.

TASK:
{task}

Produce a single Python module containing ONLY:
- the imports the implementation will need
- for each piece of work: a function (or class) signature with full type hints
  and a complete docstring describing exactly what it must do, what it takes,
  what it returns, and any edge cases the implementer must handle
- a body of exactly `...` for every one of them
- in the case that the function needs decorators, include them
- if a function is private, prefix it with an underscore
- if a function is small enough that it can be implemented in one line, do not do docstring for it and just do it.

Write no implementation logic. The docstring is your specification - an
implementer will see only the module you write here, so anything you leave
unsaid will be guessed at. Aim for {max_stubs} functions at most.

Reply with the module inside one ```python code block and nothing else."""

IMPLEMENTER_PROMPT = """You are implementing ONE function from a module another\
 engineer designed. Do not redesign it.

THE MODULE'S INTERFACE (for context - do not reimplement these):
{interface}

IMPLEMENT EXACTLY THIS FUNCTION:
{stub}

Rules:
- Keep the name, parameters, type hints and docstring exactly as given.
- Write the real body. No `...`, no `TODO`, no `raise NotImplementedError`.
- Use only the imports shown in the interface, plus the standard library.

Reply with the complete function inside one ```python code block and nothing else."""

PLANNER_PROMPT = """You are planning how to complete a task. Do NOT do the task.

TASK:
{task}

Write a numbered plan of at most {max_steps} steps. Each step must be one
self-contained instruction that someone could carry out without seeing the
others. Write one step per line, numbered "1.", "2.", and so on, with no
sub-bullets and no commentary before or after the list."""

STEP_PROMPT = """You are carrying out one step of a larger plan.

OVERALL TASK:
{task}

WHAT HAS BEEN DONE SO FAR:
{context}

YOUR STEP:
{step}

Do only this step. Reply with its result and nothing else."""

SYNTHESIS_PROMPT = """You planned this task and the steps have now been carried out.

ORIGINAL TASK:
{task}

STEP RESULTS:
{results}

Produce the final answer to the original task, drawing on the step results.
Do not describe the process - deliver the result itself."""


# ---------------------------------------------------------------------------
# Stub extraction / assembly
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_block(text: str) -> str:
    """The first fenced code block in `text`, or the whole string if there is
    none. Models wrap code in fences the overwhelming majority of the time and
    prose around it the rest, so taking the first block and falling back to the
    raw text handles both without a parser."""
    match = _CODE_BLOCK.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


# The three node kinds this module ever treats as "a thing to fill in".
# Named because several helpers below reach for `.name` and `.lineno`, which the
# base ast.AST does not carry - annotating them as AST typechecked only because
# nothing was looking.
_Definition = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]


def _source_of(code: str, node: ast.AST) -> str:
    """The source text of a top-level `node`, **decorators included**.

    `ast.get_source_segment()` alone is not enough here. Since Python 3.8 a
    decorated def's `lineno` points at the `def` keyword, not at the first
    decorator, so the decorators sit outside the segment it returns - they are
    not in the returned stub, and because the decorators belong to the same
    node they are not emitted as preamble either. They simply vanish, and the
    assembled module compiles cleanly without them.

    That is the single worst failure mode this module has (a syntactically
    valid module that quietly does the wrong thing - the same reason a
    surviving `...` is treated as an error), and it is invisible: an
    `@property` or an `@functools.cache` that disappears changes behaviour
    without changing whether anything parses. `StubPlan.interface()` made it
    worse by going through `code_to_stub()`, which unparses the AST and so
    *keeps* decorators - the implementer saw a decorator the assembler was
    about to drop.

    So widen the range to the first decorator by hand. Only top-level nodes
    are passed here (both callers iterate `tree.body`), so the lines are at
    column 0 and can be sliced directly.
    """
    decorators = list(getattr(node, "decorator_list", []) or [])
    end = getattr(node, "end_lineno", None)
    if decorators and end is not None:
        lines = code.splitlines()
        start = min(d.lineno for d in decorators) - 1
        # A decorator node's lineno is the line its `@` is on, but scan up
        # anyway rather than trusting that across versions and across
        # decorators written over several lines.
        while start > 0 and not lines[start].lstrip().startswith("@"):
            start -= 1
        if 0 <= start < len(lines):
            return "\n".join(lines[start:end]).strip("\n")
    segment = ast.get_source_segment(code, node)
    return segment if segment is not None else ast.unparse(node)


def _decorator_texts(node: ast.AST) -> Tuple[str, ...]:
    """Each decorator on `node`, normalized through `ast.unparse`.

    Unparsed rather than compared as raw text so that whitespace and quoting
    differences between what the architect wrote and what the implementer
    echoed back don't read as a changed decorator.
    """
    return tuple(ast.unparse(d) for d in getattr(node, "decorator_list", []) or [])


def _logic_line_count(node: ast.AST) -> int:
    """Lines of real logic in a def's body, docstring excluded.

    Same measurement `ContextCompressor.code_to_stub` makes when deciding
    whether a body is short enough to keep, and made the same way for the same
    reason: unparsed rather than counted off the source, so formatting
    (a body split over three lines for line-length reasons) doesn't change the
    verdict.
    """
    body = list(getattr(node, "body", []))
    if not body:
        return 0
    if (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if not body:
        return 0
    return len("\n".join(ast.unparse(stmt) for stmt in body).splitlines())


def _strip_to_stub(code: str, node: "_Definition") -> str:
    """An implemented def rewritten back into a stub, body replaced by `...`.

    Used when the architect took the one-line shortcut and then wrote more
    than MAX_ARCHITECT_BODY_LINES, so the function goes back into the pool for
    an implementer. Rebuilt from the AST (decorators, signature and type hints
    survive `ast.unparse`) rather than by slicing text, since the body is
    exactly the part being discarded.
    """
    clone = copy.deepcopy(node)
    clone.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]  # type: ignore[attr-defined]
    return ast.unparse(ast.fix_missing_locations(clone))


def _is_stub_body(node: ast.AST) -> bool:
    """Whether a def's body is empty in the stub sense: nothing but a
    docstring, `pass`, and/or `...`."""
    body = list(getattr(node, "body", []))
    if not body:
        return True
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if stmt.value.value is Ellipsis or isinstance(stmt.value.value, str):
                continue
        return False
    return True


@dataclass
class Stub:
    """One thing an implementer will be asked to fill in."""

    name: str
    kind: str  # "function" | "async function" | "class"
    source: str  # the stub as written by the architect, decorators included
    docstring: str = ""
    signature: str = ""
    # Normalized decorator expressions, so the filled implementation can be
    # checked for having kept them - see _implementation_for.
    decorators: Tuple[str, ...] = ()
    # Set when the architect implemented this itself but wrote too much to
    # qualify for the one-line shortcut, so it was demoted back into the pool.
    demoted_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "signature": self.signature,
                "docstring": self.docstring, "source": self.source,
                "decorators": list(self.decorators),
                "demoted_reason": self.demoted_reason}


@dataclass
class StubPlan:
    """What the architect produced: the stubs to fill, plus everything else in
    the module (imports, constants, anything already implemented) kept verbatim
    as `preamble` so assembly can put it back untouched."""

    module_source: str
    stubs: List[Stub] = field(default_factory=list)
    preamble: str = ""
    parse_error: Optional[str] = None
    # Functions the architect implemented itself under the prompt's one-line
    # shortcut. They are legitimately in the preamble and no implementer is
    # paid to fill them - but they are also the only code in the assembled
    # module that no second agent ever looked at, so they are named here (and
    # logged by stub_and_fill) rather than being indistinguishable from work
    # an implementer did.
    architect_implemented: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.parse_error is None

    @property
    def demoted(self) -> List[str]:
        """Stubs that exist because the architect over-implemented them."""
        return [s.name for s in self.stubs if s.demoted_reason]

    def interface(self) -> str:
        """The stub module as an implementer should see it - normalized through
        compressor.code_to_stub so a chatty architect that half-implemented
        something still hands over a clean interface."""
        return ContextCompressor.code_to_stub(self.module_source)

    def to_dict(self) -> Dict[str, Any]:
        return {"stubs": [s.to_dict() for s in self.stubs], "preamble": self.preamble,
                "parse_error": self.parse_error,
                "architect_implemented": list(self.architect_implemented),
                "demoted": self.demoted}


def extract_stubs(
    source: str,
    max_stubs: int = MAX_STUBS,
    max_architect_body_lines: int = MAX_ARCHITECT_BODY_LINES,
) -> StubPlan:
    """Parse an architect's module into a StubPlan.

    A top-level `def`/`async def`/`class` whose body is only a docstring,
    `pass`, and/or `...` is a stub to fill. Everything else - imports,
    constants, fully-written helpers the architect decided to provide - is
    preamble and is preserved verbatim.

    Decorators travel with the thing they decorate, in both directions - see
    `_source_of` for why that needs saying.

    The prompt lets the architect implement a function outright when it fits on
    one line and needs no docstring. Those are recognised here rather than
    taken on trust:

    - implemented, no docstring, within `max_architect_body_lines` - allowed,
      kept in the preamble, and its name recorded on `architect_implemented`
      so the result object can say which code no implementer ever saw.
    - implemented, no docstring, longer than that - the architect claimed the
      shortcut and then wrote a real implementation. It is stripped back to a
      stub and put in the pool, carrying `demoted_reason`.
    - implemented **with** a docstring - a deliberate helper, which this
      function has always kept verbatim in the preamble. Unchanged.

    A module that doesn't parse comes back with `parse_error` set and no stubs,
    rather than raising: the architect is a language model, occasionally emits
    something unparseable, and the caller needs to be able to log that and
    escalate rather than crash the whole company.
    """
    code = extract_code_block(source)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return StubPlan(module_source=code, parse_error=f"architect's module did not parse: {e}")

    lines = code.splitlines()
    stubs: List[Stub] = []
    preamble_parts: List[str] = []
    architect_implemented: List[str] = []

    def make_stub(node: _Definition, source: str, demoted_reason: str = "") -> Stub:
        kind = ("class" if isinstance(node, ast.ClassDef)
                else "async function" if isinstance(node, ast.AsyncFunctionDef) else "function")
        signature = ""
        if node.lineno - 1 < len(lines):
            signature = lines[node.lineno - 1].strip()
        return Stub(
            name=node.name, kind=kind, source=source,
            docstring=ast.get_docstring(node) or "", signature=signature,
            decorators=_decorator_texts(node), demoted_reason=demoted_reason,
        )

    for node in tree.body:
        segment = _source_of(code, node)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            preamble_parts.append(segment)
            continue

        if _is_stub_body(node):
            stubs.append(make_stub(node, segment))
            continue

        if not isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
            # The one-line shortcut. A class is never eligible: "small enough
            # for one line" is a statement about a function body, and a
            # docstring-less class is far more likely to be a data holder the
            # architect meant as preamble than a claimed shortcut.
            body_lines = _logic_line_count(node)
            if body_lines > max_architect_body_lines:
                reason = (f"architect implemented it in {body_lines} lines without a docstring; "
                          f"the one-line shortcut allows at most {max_architect_body_lines}")
                stubs.append(make_stub(node, _strip_to_stub(code, node), demoted_reason=reason))
                continue
            architect_implemented.append(node.name)

        preamble_parts.append(segment)

    return StubPlan(
        module_source=code,
        stubs=stubs[:max_stubs],
        preamble="\n\n".join(preamble_parts).strip(),
        architect_implemented=architect_implemented,
    )


@dataclass
class FilledStub:
    stub: Stub
    implementation: str = ""
    employee: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.implementation.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.stub.name, "employee": self.employee,
                "ok": self.ok, "error": self.error}


def _implementation_for(stub: Stub, reply: str) -> FilledStub:
    """Pull the implementation of `stub` out of an implementer's reply.

    Accepts the two shapes models actually produce: the whole function
    redefined (the requested form), or - occasionally - the module with the
    one function filled in. Both are handled by parsing the reply and taking
    the definition whose name matches. A reply that parses but doesn't define
    the requested name, or that still has a stub body, is an error rather than
    a silent pass: a `...` that survives assembly would be a syntax-valid
    module that does nothing, which is the worst possible failure mode here.

    A dropped or rewritten **decorator** is treated the same way, and for the
    same reason. An implementer that returns the right function body without
    the `@property` it was given produces a module that compiles and behaves
    differently - exactly the silent-wrongness this function exists to catch,
    just one line higher up. Comparison is on unparsed decorator expressions,
    so reformatting is not a change.
    """
    code = extract_code_block(reply)
    if not code.strip():
        return FilledStub(stub=stub, error="implementer returned nothing")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return FilledStub(stub=stub, error=f"implementation did not parse: {e}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == stub.name:
            if _is_stub_body(node):
                return FilledStub(stub=stub, error="implementer returned another stub, not a body")
            got = _decorator_texts(node)
            if stub.decorators and got != stub.decorators:
                missing = [d for d in stub.decorators if d not in got]
                detail = (f"dropped {', '.join('@' + d for d in missing)}" if missing
                          else f"changed them to {', '.join('@' + d for d in got) or 'none'}")
                return FilledStub(
                    stub=stub,
                    error=f"implementer did not keep the decorators on {stub.name!r} ({detail})",
                )
            return FilledStub(stub=stub, implementation=_source_of(code, node).strip())

    return FilledStub(
        stub=stub,
        error=f"implementation did not define {stub.name!r} (defined: "
              f"{', '.join(n.name for n in tree.body if hasattr(n, 'name')) or 'nothing'})",
    )


@dataclass
class StubAndFillResult:
    """Everything the run produced, successes and failures together."""

    task: str
    plan: StubPlan
    filled: List[FilledStub] = field(default_factory=list)
    code: str = ""
    architect: Optional[str] = None
    syntax_error: Optional[str] = None
    # Set only when stub_and_fill was given verify=: None means "passed" (or
    # "not checked" - see `verified`), a string is what the verifier reported.
    verify_error: Optional[str] = None
    verified: bool = False

    @property
    def unfilled(self) -> List[str]:
        return [f.stub.name for f in self.filled if not f.ok]

    @property
    def architect_implemented(self) -> List[str]:
        """Functions the architect wrote itself under the one-line shortcut.

        Surfaced on the result, not just the plan, because "which parts of
        this module did no second agent look at?" is a question about the
        delivered artifact - the same class of question as `unfilled`.
        """
        return list(self.plan.architect_implemented)

    @property
    def ok(self) -> bool:
        return (self.plan.ok and not self.unfilled and self.syntax_error is None
                and self.verify_error is None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task, "architect": self.architect, "ok": self.ok,
            "stubs": [f.to_dict() for f in self.filled],
            "unfilled": self.unfilled, "syntax_error": self.syntax_error,
            "verified": self.verified, "verify_error": self.verify_error,
            "parse_error": self.plan.parse_error,
            "architect_implemented": self.architect_implemented,
            "demoted": self.plan.demoted,
        }


def assemble_module(plan: StubPlan, filled: Sequence[FilledStub]) -> str:
    """Reassemble preamble + implementations into one module.

    A stub that failed to fill is emitted as its original stub with a
    `# UNFILLED:` comment above it. Keeping it visible - rather than dropping
    it or raising - means the returned module is still a complete, readable
    artifact that says exactly where it is incomplete. Callers check
    `.unfilled` for the machine-readable version.
    """
    parts: List[str] = []
    if plan.preamble:
        parts.append(plan.preamble)
    for item in filled:
        if item.ok:
            parts.append(item.implementation)
        else:
            parts.append(f"# UNFILLED ({item.error}):\n{item.stub.source}")
    return "\n\n\n".join(p for p in parts if p.strip()) + "\n"


# ---------------------------------------------------------------------------
# Picking who does the work
# ---------------------------------------------------------------------------


def _employees_by_rank_preference(company: Any, preference: Sequence[str]) -> List[Any]:
    """Every employee, ordered by where their rank sits in `preference`.
    Employees whose rank isn't listed come last, in hiring order."""
    ordered: List[Any] = []
    for rank in preference:
        ordered.extend(e for e in company.employees.values() if e.rank == rank)
    ordered.extend(e for e in company.employees.values() if e not in ordered)
    return ordered


def choose_architect(company: Any, preference: Sequence[str] = DEFAULT_ARCHITECT_RANKS) -> Any:
    candidates = _employees_by_rank_preference(company, preference)
    if not candidates:
        raise ValueError("company has no employees - hire before delegating")
    return candidates[0]


def choose_implementers(
    company: Any,
    preference: Sequence[str] = DEFAULT_IMPLEMENTER_RANKS,
    exclude: Sequence[Any] = (),
) -> List[Any]:
    """The pool bodies get handed to, cheapest rank first.

    Falls back to including the architect when they are the only employee -
    a one-person company should still be able to run stub-and-fill (the
    architect just fills their own stubs), rather than erroring out on a
    structural technicality.
    """
    candidates = [e for e in _employees_by_rank_preference(company, preference) if e not in exclude]
    if not candidates:
        candidates = list(exclude)
    return candidates


# ---------------------------------------------------------------------------
# Strategy 1: stub-and-fill
# ---------------------------------------------------------------------------


def stub_and_fill(
    company: Any,
    task: str,
    architect: Optional[Any] = None,
    implementers: Optional[Sequence[Any]] = None,
    max_stubs: int = MAX_STUBS,
    architect_prompt: str = ARCHITECT_PROMPT,
    implementer_prompt: str = IMPLEMENTER_PROMPT,
    max_architect_body_lines: int = MAX_ARCHITECT_BODY_LINES,
    verify: Optional[Callable[[str], Optional[str]]] = None,
) -> StubAndFillResult:
    """Synchronous wrapper over stub_and_fill_async(). See that for the docs."""
    return run_coroutine_blocking(
        lambda: stub_and_fill_async(
            company, task, architect=architect, implementers=implementers,
            max_stubs=max_stubs, architect_prompt=architect_prompt,
            implementer_prompt=implementer_prompt,
            max_architect_body_lines=max_architect_body_lines, verify=verify),
        what="stub_and_fill()",
    )


async def stub_and_fill_async(
    company: Any,
    task: str,
    architect: Optional[Any] = None,
    implementers: Optional[Sequence[Any]] = None,
    max_stubs: int = MAX_STUBS,
    architect_prompt: str = ARCHITECT_PROMPT,
    implementer_prompt: str = IMPLEMENTER_PROMPT,
    max_architect_body_lines: int = MAX_ARCHITECT_BODY_LINES,
    verify: Optional[Callable[[str], Optional[str]]] = None,
) -> StubAndFillResult:
    """Architect writes the interface; implementers fill the bodies.

    `verify`, if given, is called with the assembled source once it compiles.
    Return None (or "") for a pass, or a string describing the failure, which
    lands on `result.verify_error` and makes `result.ok` false. Use
    `python_verifier(test_code)` to import the module and run tests against
    it in a subprocess. It runs off the event loop; an exception it raises is
    reported, never propagated.

    Implementers are assigned round-robin over the pool - deliberately not
    "best implementer per stub", because ranking stubs by difficulty would
    need exactly the dynamic-decomposition judgment this phase decided not to
    fake in v1.

    Never raises for a model-quality failure: an unparseable architect module,
    an implementer that returns another stub, an assembled module with a
    syntax error - all come back on the result object, logged to the company's
    activity log. It raises only for structural mistakes (no employees).
    """
    architect = architect or choose_architect(company)
    pool = list(implementers or choose_implementers(company, exclude=[architect]))

    company._log("stub_architect_start", employee=architect.name, task=task[:200])
    module_source = await architect.run_async(architect_prompt.format(task=task, max_stubs=max_stubs), company=company)
    plan = extract_stubs(module_source, max_stubs=max_stubs,
                         max_architect_body_lines=max_architect_body_lines)
    company._log("stub_architect_done", employee=architect.name,
                 stub_count=len(plan.stubs), parse_error=plan.parse_error,
                 architect_implemented=plan.architect_implemented,
                 demoted=plan.demoted)

    result = StubAndFillResult(task=task, plan=plan, architect=architect.name)
    if not plan.ok:
        result.code = plan.module_source
        return result
    if not plan.stubs:
        # The architect answered with something already complete (or with no
        # functions at all). Returning it verbatim is more useful than
        # inventing stubs to justify the strategy.
        result.code = plan.module_source
        return result

    interface = plan.interface()

    # The one place in this library where fan-out is unambiguously free: every
    # stub is filled from the same frozen interface and never sees another
    # stub's body, so N stubs are N independent calls. plan_then_execute below
    # is the opposite case and stays sequential.
    #
    # Round-robin can hand two stubs to the same implementer; Employee.run_async
    # takes that employee's own lock, so those two serialize while the rest keep
    # going. Nothing here has to know that.
    assignments = [(pool[index % len(pool)] if pool else architect, stub)
                   for index, stub in enumerate(plan.stubs)]

    async def _fill(implementer: Any, stub: "Stub") -> Any:
        return await implementer.run_async(
            implementer_prompt.format(interface=interface, stub=stub.source), company=company
        )

    replies = await asyncio.gather(
        *(_fill(implementer, stub) for implementer, stub in assignments),
        return_exceptions=True,
    )
    # gather(return_exceptions=True) so one implementer blowing up does not
    # leave its siblings running unattended with nobody holding their results.
    # The first real failure is still raised, exactly as the sequential version
    # did - it just happens after everyone has finished rather than mid-flight.
    for reply in replies:
        if isinstance(reply, BaseException):
            raise reply

    for (implementer, stub), reply in zip(assignments, replies):
        item = _implementation_for(stub, reply)
        item.employee = implementer.name
        result.filled.append(item)
        company._log("stub_filled", employee=implementer.name, stub=stub.name,
                     ok=item.ok, error=item.error)

    result.code = assemble_module(plan, result.filled)
    try:
        compile(result.code, "<assembled>", "exec")
    except SyntaxError as e:
        # Syntax only. Nothing here executes the module or checks that it does
        # what the docstrings promised - see the module docstring.
        result.syntax_error = str(e)
    if verify is not None and result.syntax_error is None and not result.unfilled:
        try:
            outcome = await asyncio.to_thread(verify, result.code)
        except Exception as e:  # noqa: BLE001 - a broken verifier is a result, not a crash
            outcome = f"verifier raised {e.__class__.__name__}: {e}"
        result.verified = True
        result.verify_error = outcome or None
        company._log("stub_verified", employee=architect.name, ok=result.verify_error is None,
                     error=(result.verify_error or "")[:500])
    company._log("stub_assembled", employee=architect.name, ok=result.ok,
                 unfilled=result.unfilled, syntax_error=result.syntax_error)
    return result


def python_verifier(test_code: str = "", timeout: float = 60.0,
                    python: Optional[str] = None) -> Callable[[str], Optional[str]]:
    """A `verify=` callable for stub_and_fill: runs the assembled module, then
    `test_code` (plain asserts, or anything that exits non-zero on failure),
    in a fresh interpreter (`python -I`) in a temporary directory.

    Returns None when the script exits 0, else the tail of its output. With no
    test_code it is an import check - catching NameErrors at module level,
    missing imports and the like, which compile() cannot.

    This executes code a model wrote. It is a subprocess in a temp directory,
    not a sandbox: only use it where you would run that code yourself.
    """
    interpreter = python or sys.executable

    def verify(code: str) -> Optional[str]:
        with tempfile.TemporaryDirectory(prefix="llmadapt_verify_") as workdir:
            path = os.path.join(workdir, "generated_module.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
                if test_code:
                    f.write("\n\n# ---- verification ----\n")
                    f.write(test_code)
                    f.write("\n")
            try:
                proc = subprocess.run([interpreter, "-I", path], cwd=workdir, capture_output=True,
                                      text=True, timeout=timeout, errors="replace")
            except subprocess.TimeoutExpired:
                return f"verification timed out after {timeout:g}s"
        if proc.returncode == 0:
            return None
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return f"exit code {proc.returncode}:\n" + ContextCompressor.compress_tool_output(output, max_chars=3000)

    return verify


# ---------------------------------------------------------------------------
# Strategy 2: plan-then-execute
# ---------------------------------------------------------------------------

_STEP_LINE = re.compile(r"^\s*(?:\d+)\s*[.)\]:-]\s+(.*\S)\s*$")


def parse_plan_steps(text: str, max_steps: int = MAX_PLAN_STEPS) -> List[str]:
    """Numbered steps out of a planner's reply.

    Matches "1.", "2)", "3 -", "4:" at the start of a line. Continuation lines
    (an indented wrap of the previous step) are appended to that step rather
    than dropped. If nothing matches, returns [] and the caller decides -
    `plan_then_execute` turns that into a single step containing the whole
    task, which degrades to plain execution instead of failing.
    """
    steps: List[str] = []
    for line in (text or "").splitlines():
        match = _STEP_LINE.match(line)
        if match:
            steps.append(match.group(1).strip())
        elif steps and line.strip() and line.startswith((" ", "\t")):
            steps[-1] += " " + line.strip()
    return steps[:max_steps]


@dataclass
class PlanStep:
    index: int
    instruction: str
    result: str = ""
    employee: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "instruction": self.instruction,
                "employee": self.employee, "error": self.error,
                "result_preview": self.result[:200]}


@dataclass
class PlanResult:
    task: str
    steps: List[PlanStep] = field(default_factory=list)
    answer: str = ""
    planner: Optional[str] = None
    degraded: bool = False  # True when no plan parsed and the task ran as one step

    @property
    def ok(self) -> bool:
        return all(s.error is None for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "planner": self.planner, "degraded": self.degraded,
                "ok": self.ok, "steps": [s.to_dict() for s in self.steps]}


def plan_then_execute(
    company: Any,
    task: str,
    planner: Optional[Any] = None,
    executors: Optional[Sequence[Any]] = None,
    max_steps: int = MAX_PLAN_STEPS,
    synthesize: bool = True,
    context_chars: int = 1200,
    planner_prompt: str = PLANNER_PROMPT,
    step_prompt: str = STEP_PROMPT,
    synthesis_prompt: str = SYNTHESIS_PROMPT,
) -> PlanResult:
    """Synchronous wrapper over plan_then_execute_async(). See that for the docs."""
    return run_coroutine_blocking(
        lambda: plan_then_execute_async(
            company, task, planner=planner, executors=executors, max_steps=max_steps,
            synthesize=synthesize, context_chars=context_chars,
            planner_prompt=planner_prompt, step_prompt=step_prompt,
            synthesis_prompt=synthesis_prompt),
        what="plan_then_execute()",
    )


async def plan_then_execute_async(
    company: Any,
    task: str,
    planner: Optional[Any] = None,
    executors: Optional[Sequence[Any]] = None,
    max_steps: int = MAX_PLAN_STEPS,
    synthesize: bool = True,
    context_chars: int = 1200,
    planner_prompt: str = PLANNER_PROMPT,
    step_prompt: str = STEP_PROMPT,
    synthesis_prompt: str = SYNTHESIS_PROMPT,
) -> PlanResult:
    """Plan first, then carry the steps out in order.

    Deliberately NOT parallelized, unlike stub_and_fill_async. Each step is
    handed a digest of what the earlier steps produced - that is the whole
    design of the strategy - so step 3 cannot start before step 2 has an
    answer. Fanning these out would not be faster, it would be wrong.

    Each step sees a compressed digest of what earlier steps produced, not
    their full output - `ContextCompressor.compress_tool_output` does the
    trimming, the same machinery Phase 0-3 already uses for tool results,
    because feeding every prior step's full text into every later step is how
    a five-step plan turns into a context-window failure.

    `synthesize=True` (the default) gives the planner a final pass to turn the
    step results into one answer. Set it False when the last step's output is
    already the deliverable and a synthesis pass would just paraphrase it at
    extra cost.

    A step that raises is recorded on that step and the run continues - one
    failed step out of six shouldn't discard the other five. `PlanResult.ok`
    reports whether every step succeeded.
    """
    planner = planner or choose_architect(company)
    pool = list(executors or choose_implementers(company, exclude=[planner]))

    company._log("plan_start", employee=planner.name, task=task[:200])
    plan_text = await planner.run_async(planner_prompt.format(task=task, max_steps=max_steps), company=company)
    instructions = parse_plan_steps(plan_text, max_steps=max_steps)

    result = PlanResult(task=task, planner=planner.name)
    if not instructions:
        # No parseable plan. Rather than fail, run the task as a single step -
        # the caller still gets an answer, and `degraded` says what happened.
        instructions = [task]
        result.degraded = True
    company._log("plan_made", employee=planner.name, step_count=len(instructions),
                 degraded=result.degraded)

    context_parts: List[str] = []
    for index, instruction in enumerate(instructions):
        executor = pool[index % len(pool)] if pool else planner
        context = "\n".join(context_parts) if context_parts else "(nothing yet - this is the first step)"
        step = PlanStep(index=index + 1, instruction=instruction, employee=executor.name)
        try:
            step.result = await executor.run_async(
                step_prompt.format(task=task, context=context, step=instruction), company=company
            )
        except Exception as e:  # a step failing shouldn't discard the rest of the plan
            step.error = f"{e.__class__.__name__}: {e}"
        result.steps.append(step)
        company._log("plan_step", employee=executor.name, step=step.index, error=step.error)
        digest = ContextCompressor.compress_tool_output(step.result or "", max_chars=context_chars)
        context_parts.append(f"Step {step.index} ({instruction}): {digest}")

    successful = [s for s in result.steps if s.error is None]
    if synthesize and successful:
        results_text = "\n\n".join(f"Step {s.index}: {s.instruction}\n{s.result}" for s in successful)
        result.answer = await planner.run_async(
            synthesis_prompt.format(task=task, results=results_text), company=company
        )
        company._log("plan_synthesized", employee=planner.name)
    elif successful:
        result.answer = successful[-1].result
    company._log("plan_done", employee=planner.name, ok=result.ok)
    return result


__all__ = [
    "Stub", "StubPlan", "FilledStub", "StubAndFillResult",
    "extract_code_block", "extract_stubs", "assemble_module", "stub_and_fill",
    "stub_and_fill_async", "plan_then_execute_async", "python_verifier",
    "PlanStep", "PlanResult", "parse_plan_steps", "plan_then_execute",
    "choose_architect", "choose_implementers",
    "DEFAULT_ARCHITECT_RANKS", "DEFAULT_IMPLEMENTER_RANKS",
    "MAX_STUBS", "MAX_PLAN_STEPS", "MAX_ARCHITECT_BODY_LINES",
]

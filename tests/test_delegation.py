# Phase 6 tests - delegation.py (stub-and-fill + plan-then-execute) and its
# Company.run_structured() entry point. Plain asserts + prints, no pytest, no
# network. Layers, bottom-up:
#   (1) pure parsing/assembly helpers - no agents involved at all
#   (2) rank-based architect/implementer selection
#   (3) stub_and_fill end to end against scripted agents, including the
#       failure modes (unparseable architect, implementer returning a stub)
#   (4) plan_then_execute, including a step that raises and a plan that
#       doesn't parse
#   (5) Company.run_structured dispatch
import json

from llmadapt.company import Company, EscalationDecision
from llmadapt.delegation import (
    DEFAULT_ARCHITECT_RANKS,
    assemble_module,
    choose_architect,
    choose_implementers,
    _implementation_for,
    extract_code_block,
    extract_stubs,
    parse_plan_steps,
    plan_then_execute,
    stub_and_fill,
)
from llmadapt.router import RoleRank


class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self._last = None

    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        if self.responses:
            self._last = self.responses.pop(0)
        return self._last


def text_response(text):
    return {"content": [{"type": "text", "text": text}]}


def block(code):
    return f"```python\n{code}\n```"


MODEL_MAP = {rank: {"provider": "anthropic", "model": "claude-x", "api_key": "k"}
             for rank in RoleRank.ORDER}


def make_company(**kwargs):
    def on_escalation(event):
        return EscalationDecision(approve=False)

    return Company(name="Deleg Co", model_map=MODEL_MAP, on_escalation=on_escalation, **kwargs)


STUB_MODULE = '''import json


MAX_ITEMS = 10


def load(path: str) -> dict:
    """Load a JSON file from path and return the parsed dict."""
    ...


def summarize(data: dict) -> str:
    """Return a one-line summary of data."""
    ...


def already_done(x: int) -> int:
    """This one the architect implemented itself."""
    return x + 1
'''


# ---------------------------------------------------------------------------
# Layer 1: pure parsing / assembly
# ---------------------------------------------------------------------------

assert extract_code_block("chatter\n```python\nx = 1\n```\nmore") == "x = 1"
assert extract_code_block("```\ny = 2\n```") == "y = 2"
assert extract_code_block("no fences here") == "no fences here"
assert extract_code_block("") == ""
print("PASS 1: code blocks are extracted from prose, with a raw-text fallback")

plan = extract_stubs(block(STUB_MODULE))
assert plan.ok and plan.parse_error is None
assert [s.name for s in plan.stubs] == ["load", "summarize"], [s.name for s in plan.stubs]
assert "import json" in plan.preamble and "MAX_ITEMS" in plan.preamble
assert "already_done" in plan.preamble, "an implemented function belongs in the preamble, not the stubs"
assert plan.stubs[0].docstring.startswith("Load a JSON file")
assert plan.stubs[0].signature.startswith("def load(")
print("PASS 2: extract_stubs splits empty-bodied defs from everything else")

# pass / bare ... / docstring-only all count as stub bodies; real logic doesn't.
variants = extract_stubs("def a():\n    pass\n\ndef b():\n    ...\n\ndef c():\n    \"\"\"doc\"\"\"\n\ndef d():\n    return 1\n")
assert [s.name for s in variants.stubs] == ["a", "b", "c"]
assert "def d" in variants.preamble
print("PASS 3: pass, ..., and docstring-only bodies are stubs; real logic is not")

broken = extract_stubs("def oops(:\n  ...")
assert not broken.ok and "did not parse" in broken.parse_error
assert broken.stubs == []
print("PASS 4: an unparseable architect module reports parse_error instead of raising")

capped = extract_stubs("\n".join(f"def f{i}():\n    ...\n" for i in range(30)), max_stubs=5)
assert len(capped.stubs) == 5
print("PASS 5: stub count is capped as a runaway guard")

# interface() normalizes through compressor.code_to_stub
interface = plan.interface()
assert "def load" in interface and "def summarize" in interface
print("PASS 6: StubPlan.interface() reuses compressor.code_to_stub for the implementer's view")

assert parse_plan_steps("1. first\n2) second\n3 - third\n4: fourth") == \
       ["first", "second", "third", "fourth"]
assert parse_plan_steps("Here is the plan:\n1. alpha\n   continued\n2. beta") == \
       ["alpha continued", "beta"]
assert parse_plan_steps("no numbers at all") == []
assert len(parse_plan_steps("\n".join(f"{i}. step" for i in range(1, 40)), max_steps=6)) == 6
print("PASS 7: plan steps parse from several numbering styles, with wraps joined and a cap")


# ---------------------------------------------------------------------------
# Layer 2: who does what, fixed by rank
# ---------------------------------------------------------------------------

co = make_company()
boss = co.hire("Boss", RoleRank.C_SUITE)
mgr = co.hire("Mgr", RoleRank.MANAGER, reports_to=boss)
junior = co.hire("Jun", RoleRank.JUNIOR, reports_to=mgr)
intern = co.hire("Int", RoleRank.INTERN, reports_to=mgr)

assert choose_architect(co) is boss, "architect should be the most senior available"
implementers = choose_implementers(co, exclude=[boss])
assert implementers[0] is intern, "implementers should start from the cheapest rank"
assert boss not in implementers
assert DEFAULT_ARCHITECT_RANKS[0] == RoleRank.C_SUITE
print("PASS 8: architect is the most senior employee, implementers start from the cheapest")

solo_co = make_company()
only = solo_co.hire("Only", RoleRank.SENIOR)
assert choose_architect(solo_co) is only
assert choose_implementers(solo_co, exclude=[only]) == [only]
print("PASS 9: a one-person company falls back to the architect implementing their own stubs")

empty_co = make_company()
try:
    choose_architect(empty_co)
    assert False, "expected ValueError for an empty company"
except ValueError as e:
    assert "no employees" in str(e)
print("PASS 10: an empty company raises a structural error rather than failing later")


# ---------------------------------------------------------------------------
# Layer 3: stub_and_fill end to end
# ---------------------------------------------------------------------------

def build_stub_company():
    company = make_company()
    architect = company.hire("Arch", RoleRank.C_SUITE)
    worker_a = company.hire("W1", RoleRank.INTERN, reports_to=architect)
    worker_b = company.hire("W2", RoleRank.INTERN, reports_to=architect)
    return company, architect, worker_a, worker_b


company, architect, w1, w2 = build_stub_company()
architect.agent._send_request = FakeResponder([text_response(block(STUB_MODULE))])
w1.agent._send_request = FakeResponder([text_response(block(
    'def load(path: str) -> dict:\n'
    '    """Load a JSON file from path and return the parsed dict."""\n'
    '    with open(path) as fh:\n'
    '        return json.load(fh)\n'
))])
w2.agent._send_request = FakeResponder([text_response(block(
    'def summarize(data: dict) -> str:\n'
    '    """Return a one-line summary of data."""\n'
    '    return f"{len(data)} keys"\n'
))])

result = stub_and_fill(company, "build a tiny json tool")
assert result.ok, (result.unfilled, result.syntax_error)
assert result.architect == "Arch"
assert [f.stub.name for f in result.filled] == ["load", "summarize"]
assert result.filled[0].employee == "W1" and result.filled[1].employee == "W2"
assert "json.load(fh)" in result.code and "len(data)" in result.code
assert "import json" in result.code and "MAX_ITEMS" in result.code
assert "already_done" in result.code, "the architect's own implementation must survive assembly"
compile(result.code, "<t>", "exec")
print("PASS 11: stub_and_fill assembles a compiling module from architect + implementers")

from llmadapt import python_verifier  # noqa: E402


def run_with_verifier(verify, summarize_body='    return f"{len(data)} keys"\n'):
    company, architect, w1, w2 = build_stub_company()
    architect.agent._send_request = FakeResponder([text_response(block(STUB_MODULE))])
    w1.agent._send_request = FakeResponder([text_response(block(
        'def load(path: str) -> dict:\n    """Load."""\n    with open(path) as fh:\n        return json.load(fh)\n'))])
    w2.agent._send_request = FakeResponder([text_response(block(
        'def summarize(data: dict) -> str:\n    """Summarize."""\n' + summarize_body))])
    return stub_and_fill(company, "build a tiny json tool", verify=verify)


passing = run_with_verifier(python_verifier("assert summarize({'a': 1}) == '1 keys'"))
assert passing.ok and passing.verified and passing.verify_error is None, passing.verify_error
# Compiles fine, but is wrong - exactly what compile() alone could never catch.
wrong = run_with_verifier(python_verifier("assert summarize({'a': 1}) == '1 keys'"),
                          summarize_body='    return f"{len(data) + 1} keys"\n')
assert wrong.syntax_error is None and wrong.verified and not wrong.ok
assert "AssertionError" in wrong.verify_error, wrong.verify_error
assert wrong.to_dict()["verify_error"] == wrong.verify_error
broken = run_with_verifier(lambda code: 1 / 0)
assert not broken.ok and "ZeroDivisionError" in broken.verify_error
unchecked = run_with_verifier(None)
assert unchecked.ok and not unchecked.verified
print("PASS 11b: verify= runs the assembled module + tests; a wrong body fails, a broken verifier is reported")

# Implementers are assigned round-robin, and each one only sees the interface.
implementer_payload = json.dumps(w1.agent._send_request.calls[0])
assert "def load" in implementer_payload and "def summarize" in implementer_payload
assert "json.load(fh)" not in implementer_payload, "implementer must not be shown the answer"
print("PASS 12: each implementer is given the interface, not other implementers' work")

events = [e["kind"] for e in company.activity()]
assert "stub_architect_start" in events and "stub_filled" in events and "stub_assembled" in events
print("PASS 13: the whole stub-and-fill run is traced in the activity log")

# --- failure modes ---
company2, arch2, x1, x2 = build_stub_company()
arch2.agent._send_request = FakeResponder([text_response(block("def broken(:\n  ..."))])
x1.agent._send_request = FakeResponder([text_response("should never be called")])
x2.agent._send_request = FakeResponder([text_response("should never be called")])
result2 = stub_and_fill(company2, "t")
assert not result2.ok and result2.plan.parse_error
assert x1.agent._send_request.calls == [], "no implementer should be paid for an unparseable plan"
print("PASS 14: an unparseable architect module aborts before any implementer is called")

company3, arch3, y1, y2 = build_stub_company()
arch3.agent._send_request = FakeResponder([text_response(block(
    'def only(a: int) -> int:\n    """Do it."""\n    ...\n'))])
y1.agent._send_request = FakeResponder([text_response(block(
    'def only(a: int) -> int:\n    """Do it."""\n    ...\n'))])
result3 = stub_and_fill(company3, "t")
assert not result3.ok and result3.unfilled == ["only"]
assert "another stub" in result3.filled[0].error
assert "# UNFILLED" in result3.code and "def only" in result3.code
print("PASS 15: an implementer returning another stub is reported unfilled, not silently accepted")

company4, arch4, z1, z2 = build_stub_company()
arch4.agent._send_request = FakeResponder([text_response(block(
    'def wanted(a: int) -> int:\n    """Do it."""\n    ...\n'))])
z1.agent._send_request = FakeResponder([text_response(block(
    'def something_else(a):\n    return a\n'))])
result4 = stub_and_fill(company4, "t")
assert not result4.ok and "did not define" in result4.filled[0].error
print("PASS 16: an implementer that renames the function is reported, not silently mismatched")

company5, arch5, q1, q2 = build_stub_company()
arch5.agent._send_request = FakeResponder([text_response(block("x = 1\ny = 2\n"))])
q1.agent._send_request = FakeResponder([text_response("should never be called")])
q2.agent._send_request = FakeResponder([text_response("should never be called")])
result5 = stub_and_fill(company5, "t")
assert result5.plan.ok and result5.plan.stubs == []
assert "x = 1" in result5.code
assert q1.agent._send_request.calls == []
print("PASS 17: an architect that returns no stubs has its answer returned verbatim")

# assemble_module keeps an unfilled stub visible rather than dropping it
from llmadapt.delegation import FilledStub  # noqa: E402

partial = extract_stubs("def a():\n    ...\n\ndef b():\n    ...\n")
assembled = assemble_module(partial, [
    FilledStub(stub=partial.stubs[0], implementation="def a():\n    return 1"),
    FilledStub(stub=partial.stubs[1], error="nope"),
])
assert "return 1" in assembled and "# UNFILLED (nope)" in assembled and "def b" in assembled
print("PASS 18: assembly keeps unfilled stubs visible and annotated instead of dropping them")


# ---------------------------------------------------------------------------
# Layer 4: plan_then_execute
# ---------------------------------------------------------------------------

def build_plan_company():
    company = make_company()
    planner = company.hire("Plan", RoleRank.MANAGER)
    a = company.hire("A", RoleRank.JUNIOR, reports_to=planner)
    b = company.hire("B", RoleRank.JUNIOR, reports_to=planner)
    return company, planner, a, b


company6, planner, ea, eb = build_plan_company()
planner.agent._send_request = FakeResponder([
    text_response("1. gather the data\n2. write the summary"),
    text_response("FINAL ANSWER"),
])
ea.agent._send_request = FakeResponder([text_response("data gathered")])
eb.agent._send_request = FakeResponder([text_response("summary written")])

plan_result = plan_then_execute(company6, "produce a report")
assert plan_result.ok and not plan_result.degraded
assert [s.instruction for s in plan_result.steps] == ["gather the data", "write the summary"]
assert [s.employee for s in plan_result.steps] == ["A", "B"]
assert plan_result.answer == "FINAL ANSWER"
print("PASS 19: plan_then_execute plans, runs each step round-robin, and synthesizes")

# Step 2 must see step 1's result as context, but not before it exists.
step2_payload = json.dumps(eb.agent._send_request.calls[0])
assert "data gathered" in step2_payload
step1_payload = json.dumps(ea.agent._send_request.calls[0])
assert "nothing yet" in step1_payload
print("PASS 20: later steps receive a digest of earlier results; the first sees none")

company7, planner7, e7a, e7b = build_plan_company()
planner7.agent._send_request = FakeResponder([
    text_response("1. only step"), text_response("done"),
])
e7a.agent._send_request = FakeResponder([text_response("step output")])
no_synth = plan_then_execute(company7, "t", synthesize=False)
assert no_synth.answer == "step output", no_synth.answer
print("PASS 21: synthesize=False returns the last step's output instead of paying for a summary")

company8, planner8, e8a, e8b = build_plan_company()
planner8.agent._send_request = FakeResponder([
    text_response("I refuse to write a numbered list."), text_response("synth"),
])
e8a.agent._send_request = FakeResponder([text_response("did the whole thing")])
degraded = plan_then_execute(company8, "the task")
assert degraded.degraded and len(degraded.steps) == 1
assert degraded.steps[0].instruction == "the task"
print("PASS 22: an unparseable plan degrades to a single step rather than failing")


class Boom:
    def __init__(self):
        self.calls = []

    def __call__(self, payload, headers):
        raise RuntimeError("provider exploded")


company9, planner9, e9a, e9b = build_plan_company()
planner9.agent._send_request = FakeResponder([
    text_response("1. first\n2. second"), text_response("partial synthesis"),
])
e9a.agent._send_request = Boom()
e9b.agent._send_request = FakeResponder([text_response("second ok")])
partial_plan = plan_then_execute(company9, "t")
assert not partial_plan.ok
assert partial_plan.steps[0].error and "provider exploded" in partial_plan.steps[0].error
assert partial_plan.steps[1].result == "second ok"
assert partial_plan.answer == "partial synthesis"
print("PASS 23: a failing step is recorded and the remaining steps still run")

plan_events = [e["kind"] for e in company6.activity()]
assert "plan_start" in plan_events and "plan_step" in plan_events and "plan_synthesized" in plan_events
print("PASS 24: the whole plan run is traced in the activity log")


# ---------------------------------------------------------------------------
# Layer 5: Company.run_structured dispatch
# ---------------------------------------------------------------------------

company10, planner10, e10a, e10b = build_plan_company()
planner10.agent._send_request = FakeResponder([text_response("direct answer")])
assert company10.run_structured("t", strategy="direct", entry_point=planner10) == "direct answer"
print("PASS 25: strategy='direct' is exactly run() and returns a string")

company11, planner11, e11a, e11b = build_plan_company()
planner11.agent._send_request = FakeResponder([text_response("1. a\n2. b"), text_response("ans")])
e11a.agent._send_request = FakeResponder([text_response("ra")])
e11b.agent._send_request = FakeResponder([text_response("rb")])
res = company11.run_structured("t", strategy="plan", entry_point=planner11)
assert res.answer == "ans" and len(res.steps) == 2
assert any(e["kind"] == "strategy_done" for e in company11.activity())
print("PASS 26: strategy='plan' returns a PlanResult and logs the strategy")

company12, arch12, i12a, i12b = build_stub_company()
arch12.agent._send_request = FakeResponder([text_response(block(
    'def f(a: int) -> int:\n    """Add one."""\n    ...\n'))])
i12a.agent._send_request = FakeResponder([text_response(block(
    'def f(a: int) -> int:\n    """Add one."""\n    return a + 1\n'))])
res = company12.run_structured("t", strategy="stub_fill", entry_point=arch12)
assert res.ok and "return a + 1" in res.code
print("PASS 27: strategy='stub_fill' returns a StubAndFillResult carrying the assembled code")

try:
    company12.run_structured("t", strategy="telepathy", entry_point=arch12)
    assert False, "expected an unknown-strategy error"
except ValueError as e:
    assert "telepathy" in str(e)
print("PASS 28: an unknown strategy raises a clear ValueError")

# ---------------------------------------------------------------------------
# Decorators, and the architect's one-line shortcut
# ---------------------------------------------------------------------------
# Both of these guard the same failure: code that compiles and is wrong. A
# dropped decorator changes behaviour without changing whether anything parses,
# and an architect that "just does it" past the one-line allowance writes the
# implementation nobody reviews.

DECORATED = block(
    "import functools\n\n"
    "@functools.lru_cache(maxsize=8)\n"
    "def compute(x: int) -> int:\n"
    '    """Return x doubled."""\n'
    "    ...\n"
)

plan = extract_stubs(DECORATED)
assert plan.stubs[0].source.startswith("@functools.lru_cache(maxsize=8)"), plan.stubs[0].source
assert plan.stubs[0].decorators == ("functools.lru_cache(maxsize=8)",)
print("PASS 29: a decorated stub keeps its decorator in the stub source and records it")

kept = _implementation_for(plan.stubs[0], block(
    "@functools.lru_cache(maxsize=8)\n"
    "def compute(x: int) -> int:\n"
    "    return x * 2\n"))
assert kept.ok and kept.implementation.startswith("@functools.lru_cache"), kept.implementation
assembled = assemble_module(plan, [kept])
assert "@functools.lru_cache(maxsize=8)" in assembled, assembled
print("PASS 30: the decorator survives the implementer round-trip into the assembled module")

dropped = _implementation_for(plan.stubs[0], block(
    "def compute(x: int) -> int:\n"
    "    return x * 2\n"))
assert not dropped.ok and "decorator" in (dropped.error or ""), dropped.error
changed = _implementation_for(plan.stubs[0], block(
    "@staticmethod\n"
    "def compute(x: int) -> int:\n"
    "    return x * 2\n"))
assert not changed.ok and "decorator" in (changed.error or ""), changed.error
print("PASS 31: an implementer that drops or swaps a decorator is unfilled, not silently accepted")

SHORTCUT = block(
    "def tiny(x): return x + 1\n\n"
    "def sneaky(a, b):\n"
    "    total = a + b\n"
    "    total *= 2\n"
    "    return total\n\n"
    "def helper(x):\n"
    '    """A deliberate helper."""\n'
    "    return x - 1\n"
)
plan2 = extract_stubs(SHORTCUT)
assert plan2.architect_implemented == ["tiny"], plan2.architect_implemented
assert plan2.demoted == ["sneaky"], plan2.demoted
assert "def helper" in plan2.preamble and "def tiny" in plan2.preamble
assert "def sneaky" not in plan2.preamble
print("PASS 32: a one-line body is allowed and recorded; a longer one is demoted back to a stub")

assert plan2.stubs[0].source.strip().endswith("..."), plan2.stubs[0].source
assert "the one-line shortcut allows at most" in plan2.stubs[0].demoted_reason
print("PASS 33: a demoted function is stripped to a real stub and says why it was demoted")

plan3 = extract_stubs(SHORTCUT, max_architect_body_lines=5)
assert plan3.demoted == [] and sorted(plan3.architect_implemented) == ["sneaky", "tiny"]
print("PASS 34: max_architect_body_lines moves the line, and a documented helper is never demoted")

company13, arch13, i13a, i13b = build_stub_company()
arch13.agent._send_request = FakeResponder([text_response(SHORTCUT)])
i13a.agent._send_request = FakeResponder([text_response(block(
    "def sneaky(a, b):\n    return (a + b) * 2\n"))])
res13 = stub_and_fill(company13, "t", architect=arch13, implementers=[i13a])
done = [e for e in company13.activity() if e["kind"] == "stub_architect_done"][0]
assert done["architect_implemented"] == ["tiny"] and done["demoted"] == ["sneaky"]
assert res13.architect_implemented == ["tiny"]
assert res13.to_dict()["demoted"] == ["sneaky"]
print("PASS 35: stub_and_fill logs architect-written code and surfaces it on the result")


print("\nAll Phase 6 delegation checks passed.")

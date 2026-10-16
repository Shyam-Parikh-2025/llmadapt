"""help.py - one call that returns everything needed to build a company.

The problem this solves: the knowledge required to set up a `Company` was
spread across a tool schema (which arguments exist), `company_options()` (which
names are valid), four preset registries (what those names mean), and several
modules' docstrings (what the arguments actually *do* - what a budget of 0
means, what happens when nobody answers an escalation, which return value tells
you the run stopped). A model calling the setup tool could see the argument
list and nothing else, and a person had to read the source.

`company_help()` returns all of it as one structured document: the concepts,
every option with a description, the safety defaults spelled out rather than
implied, and worked examples. It is deliberately plain data - a dict, or
markdown via `company_help(as_text=True)` - so the same call serves a model
being handed context, a developer at a REPL, and a docs page.

It describes the library; it does not touch it. Calling this builds nothing,
spends nothing and changes nothing.
"""

from typing import Any, Dict, List, Optional

__all__ = ["company_help", "HELP_VERSION"]

# Bumped when the *shape* of the document changes, so anything consuming it
# programmatically can tell. Content changes with the library and does not
# move this.
HELP_VERSION = 1


_CONCEPTS: List[Dict[str, str]] = [
    {
        "term": "Company",
        "what": (
            "The whole org: a set of employees with reporting lines, a shared token budget, and "
            "one callback that gets asked whenever something needs a human. Building one runs "
            "nothing - you start work yourself with company.run(task)."
        ),
    },
    {
        "term": "Employee",
        "what": (
            "One AI agent with a rank, a job description, and optionally a manager. An employee "
            "with a manager automatically gives that manager a 'delegate_to_<name>' tool, which "
            "is how work flows down the chart - delegation is an ordinary tool call, not a "
            "separate mechanism."
        ),
    },
    {
        "term": "rank",
        "what": (
            "Seniority, from C_SUITE down to VOLUNTEER. Rank decides which model an employee "
            "gets by default (via model_map) and what share of the token budget they are "
            "allocated. It does not by itself decide who reports to whom - that is reports_to."
        ),
    },
    {
        "term": "Team",
        "what": (
            "A lead plus an optional reviewer. In the default 'critique' review mode the lead "
            "drafts, the reviewer either signs off or sends it back, bounded by "
            "max_review_rounds. An objection still standing after the last round ships attached "
            "to the work rather than being dropped."
        ),
    },
    {
        "term": "skill",
        "what": (
            "A capability written into an employee's instructions - 'you write Python', 'you "
            "check facts'. Prompt-level only: it tells the model how to behave, it does not add "
            "a tool or enforce anything. Employees can hold several."
        ),
    },
    {
        "term": "personality",
        "what": (
            "How an employee works rather than what they know - concise, skeptical, "
            "encouraging. One per employee. Compose it with skills: a skeptical code reviewer "
            "and an encouraging one behave very differently on the same task."
        ),
    },
    {
        "term": "org template",
        "what": (
            "A ready-made org shape you name instead of describing employee by employee. Scales "
            "with size='small'|'medium'|'large', which only multiplies the doing roles - the "
            "oversight roles stay as they are."
        ),
    },
    {
        "term": "escalation",
        "what": (
            "What happens when an employee cannot continue: it has run out of tool iterations "
            "(a runaway-loop guard) or out of budget. The event travels up the reporting chain, "
            "automatic reserves are tried first, and if nothing is left the company's "
            "on_escalation callback is asked. That callback can approve, decline, or say wait."
        ),
    },
    {
        "term": "budget",
        "what": (
            "total_token_budget is a company-wide ceiling on tokens spent. Each rank gets a "
            "share of it; an employee that runs dry can borrow unused room from its manager "
            "before escalating. Nothing crosses the company-wide ceiling without a human "
            "approving it."
        ),
    },
    {
        "term": "run archive vs. state snapshot",
        "what": (
            "Two different records. RunArchive is an append-only log of what happened, for "
            "auditing. save_state() is a snapshot of where things currently stand, for resuming. "
            "Neither is derived from the other and you can use either, both, or neither."
        ),
    },
]


_SAFETY: List[Dict[str, str]] = [
    {
        "rule": "No model_map means local models only",
        "detail": (
            "Build a company without saying which providers to use and every rank runs on local "
            "Ollama. A company you did not configure cannot spend money."
        ),
    },
    {
        "rule": "No on_escalation means decline",
        "detail": (
            "If you do not supply an escalation handler, the default one refuses every request "
            "for more budget or more iterations. No handler means nobody is watching, and "
            "nobody watching must mean stop - never approve-yourself."
        ),
    },
    {
        "rule": "0 means always ask a human first",
        "detail": (
            "emergency_iteration_reserve=0 and emergency_budget_tokens=0 - the defaults - mean "
            "there is no automatic reserve to draw on, so the very first escalation goes to your "
            "handler. Setting them above 0 buys a limited amount of automatic recovery."
        ),
    },
    {
        "rule": "total_token_budget=0 (or None) means NO ceiling",
        "detail": (
            "This is the one default that is permissive rather than restrictive, because a "
            "ceiling nobody chose would be a wrong ceiling. Set a real number for anything "
            "running unattended - the setup tool returns a warning when it is left unset."
        ),
    },
    {
        "rule": "Building never runs anything",
        "detail": (
            "Every build path - code, tool call, GUI - creates the org and stops. Starting work "
            "is always your explicit next step, so no build can surprise you with a bill."
        ),
    },
    {
        "rule": "Snapshots never contain API keys",
        "detail": (
            "save_state() carries provider and model names but no credentials, and this is "
            "checked when the snapshot is produced rather than assumed. Keys are re-read from "
            "the environment on the far side."
        ),
    },
]


_RUNNING: List[Dict[str, str]] = [
    {
        "call": "company.run(task)",
        "what": (
            "Run a task and get a string back. Synchronous, no event loop needed. This is the "
            "normal path. If an escalation handler asks to wait, this raises with a message "
            "pointing at run_resumable() rather than silently continuing."
        ),
    },
    {
        "call": "await company.run_async(task)",
        "what": (
            "Same thing from async code. Prefer this inside any async framework - calling the "
            "sync version from inside a running event loop still returns the right answer, but "
            "it blocks that loop for the whole run and warns you about it."
        ),
    },
    {
        "call": "company.run_structured(task, strategy=...)",
        "what": (
            "'direct' behaves like run(). 'plan' has a planner break the task into ordered steps "
            "and others carry them out. 'stub_fill' has a senior write the interface and cheaper "
            "employees fill in the bodies. The last two return a result object, not a string, "
            "because 'did every step work?' is a question a string cannot answer."
        ),
    },
    {
        "call": "company.run_resumable(task)",
        "what": (
            "Returns the answer as a string, OR a PausedRun if your escalation handler returned "
            "ESCALATION_PENDING. Check with isinstance(result, PausedRun). Opt-in, so callers "
            "who never pause anything are not forced to handle a second return type."
        ),
    },
    {
        "call": "company.resume(paused, decision)",
        "what": (
            "Continue a paused run once a human has decided. Work that already finished is "
            "replayed, not re-run. Approving with EscalationDecision(approve=True, ...) "
            "continues; approve=False raises, because waiting and then saying no is still no."
        ),
    },
    {
        "call": "company.save_state(path) / company.restore_state(path)",
        "what": (
            "Save where the company currently stands, and put it back later - including in "
            "another process. Restoring needs a company you have already rebuilt the same way, "
            "because a company holds callables (your tools, your escalation handler) that no "
            "file can carry."
        ),
    },
]


_EXAMPLES: List[Dict[str, str]] = [
    {
        "title": "The shortest thing that works",
        "code": (
            "from llmadapt import quick_company\n"
            "\n"
            "company = quick_company('small-coding-team')\n"
            "print(company.run('Write a CSV parser'))"
        ),
    },
    {
        "title": "A configured company with a real budget and a human in the loop",
        "code": (
            "from llmadapt import Company, RoleRank, EscalationDecision\n"
            "\n"
            "def on_escalation(event):\n"
            "    print(f'{event.employee_name} needs more: {event.message}')\n"
            "    if input('approve? [y/N] ').lower() == 'y':\n"
            "        return EscalationDecision(approve=True, extra_token_budget=50_000)\n"
            "    return EscalationDecision(approve=False)\n"
            "\n"
            "company = Company(\n"
            "    name='Acme',\n"
            "    model_map={RoleRank.C_SUITE: {'provider': 'anthropic', 'model': 'claude-sonnet-5-5'},\n"
            "               RoleRank.JUNIOR: {'provider': 'ollama', 'model': 'llama3.1:8b'}},\n"
            "    on_escalation=on_escalation,\n"
            "    total_token_budget=500_000,\n"
            ")\n"
            "boss = company.hire('Ada', RoleRank.C_SUITE, skills=['python'])\n"
            "company.hire('Grace', RoleRank.JUNIOR, reports_to=boss, skills=['testing'])\n"
            "print(company.run('Write and test a CSV parser'))"
        ),
    },
    {
        "title": "Describing an org as data (what the setup tool builds for you)",
        "code": (
            "from llmadapt import build_company, CompanySpec\n"
            "\n"
            "spec = CompanySpec.from_dict({\n"
            "    'name': 'Acme',\n"
            "    'template': 'small-coding-team',\n"
            "    'size': 'medium',\n"
            "    'total_token_budget': 500_000,\n"
            "    'employees': [{'name': 'Ada', 'rank': 'SENIOR', 'reports_to': 'Manager',\n"
            "                   'skills': ['python', 'testing'], 'personality': 'concise'}],\n"
            "})\n"
            "result = build_company(spec)\n"
            "company = result.company"
        ),
    },
    {
        "title": "Pausing for a human who will answer later",
        "code": (
            "from llmadapt import ESCALATION_PENDING, EscalationDecision, PausedRun\n"
            "\n"
            "company.on_escalation = lambda event: ESCALATION_PENDING\n"
            "result = company.run_resumable('a job that may need sign-off')\n"
            "if isinstance(result, PausedRun):\n"
            "    company.save_state(path='paused.json', paused=result)\n"
            "    # ... later, possibly in another process ...\n"
            "    answer = company.resume(result, EscalationDecision(approve=True,\n"
            "                                                      extra_token_budget=50_000))"
        ),
    },
]


_GOTCHAS: List[str] = [
    "Employee names must be unique within a company - they are how everything refers to "
    "each other, including reports_to and a saved snapshot.",
    "reports_to names another employee; that employee must already exist when you hire this "
    "one (or, in a spec, appear somewhere in the same list).",
    "A skill or personality name that does not exist is an error, not a silent skip - "
    "call company_help() or company_options() for the valid names.",
    "size only scales a template's worker roles. Hiring by hand ignores it entirely.",
    "max_review_rounds counts how many times work can be sent back, so there is always one "
    "more review than revision - the final revision does get looked at.",
    "An employee only does one task at a time. A manager delegating twice to the same person "
    "queues the second job rather than running both at once.",
    "A model asking for several tools in one turn runs them concurrently, capped by the "
    "agent's max_parallel_tools (default 8).",
]


def company_help(bundle: Optional[Any] = None, as_text: bool = False) -> Any:
    """Everything needed to build a company, in one call.

    Returns a dict by default, or a markdown document with `as_text=True`.
    Hand the dict to a model as context, read the markdown yourself, or call
    `set_company_up(mode="help")` for the same thing.

    Builds nothing and spends nothing - this describes the library, it does not
    use it.
    """
    from .builder import company_options
    from .presets import default_bundle

    bundle = bundle or default_bundle()
    options = company_options(bundle)
    descriptions = options.get("descriptions", {})

    def catalog(kind: str) -> List[Dict[str, str]]:
        return [{"name": name, "description": descriptions.get(kind, {}).get(name, "")}
                for name in options.get(kind, [])]

    document: Dict[str, Any] = {
        "help_version": HELP_VERSION,
        "summary": (
            "llmadapt runs a hierarchy of AI agents as a company: employees with ranks and "
            "reporting lines, delegation wired as ordinary tool calls, a shared token budget, "
            "and one human-in-the-loop callback for anything that cannot be resolved "
            "automatically. Building a company never runs a task."
        ),
        "concepts": _CONCEPTS,
        "safety_defaults": _SAFETY,
        "choices": {
            "ranks": options.get("ranks", []),
            "sizes": options.get("sizes", []),
            "review_modes": options.get("review_modes", []),
            "policy_modes": options.get("policy_modes", []),
            "skills": catalog("skills"),
            "personalities": catalog("personalities"),
            "palettes": catalog("palettes"),
            "org_templates": catalog("org_templates"),
        },
        "running_work": _RUNNING,
        "gotchas": _GOTCHAS,
        "examples": _EXAMPLES,
        "next_steps": [
            "quick_company('<template>') for the fastest path.",
            "set_up_company(...) if you are a model calling this as a tool - "
            "company_setup_schema() is its schema.",
            "make_company_via_gui() to draw the org in a browser instead.",
        ],
    }
    return _as_markdown(document) if as_text else document


def _as_markdown(doc: Dict[str, Any]) -> str:
    """The same document as prose. Markdown rather than pretty-printed JSON
    because both audiences read it better: a person scanning for one answer,
    and a model being handed it as context."""
    out: List[str] = ["# Building a company with llmadapt", "", doc["summary"], ""]

    out += ["## Concepts", ""]
    for item in doc["concepts"]:
        out += [f"- **{item['term']}** - {item['what']}"]
    out += [""]

    out += ["## Safety defaults (read these before the options)", ""]
    for item in doc["safety_defaults"]:
        out += [f"- **{item['rule']}.** {item['detail']}"]
    out += [""]

    choices = doc["choices"]
    out += ["## What you can choose from", ""]
    for key, label in (("ranks", "Ranks"), ("sizes", "Sizes"),
                       ("review_modes", "Review modes"), ("policy_modes", "Routing modes")):
        out += [f"**{label}:** {', '.join(choices.get(key, [])) or '(none)'}", ""]
    for key, label in (("org_templates", "Org templates"), ("skills", "Skills"),
                       ("personalities", "Personalities"), ("palettes", "Org-chart palettes")):
        entries = choices.get(key, [])
        out += [f"### {label} ({len(entries)})", ""]
        for entry in entries:
            description = f" - {entry['description']}" if entry["description"] else ""
            out += [f"- `{entry['name']}`{description}"]
        out += [""]

    out += ["## Running work once it is built", ""]
    for item in doc["running_work"]:
        out += [f"- `{item['call']}` - {item['what']}"]
    out += [""]

    out += ["## Things that catch people out", ""]
    for item in doc["gotchas"]:
        out += [f"- {item}"]
    out += [""]

    out += ["## Examples", ""]
    for example in doc["examples"]:
        out += [f"### {example['title']}", "", "```python", example["code"], "```", ""]

    out += ["## Next", ""]
    for step in doc["next_steps"]:
        out += [f"- {step}"]
    return "\n".join(out).rstrip() + "\n"

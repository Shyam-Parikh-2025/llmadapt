"""company/company.py - the Company itself: the org chart, the
emergency escalation reserves, the budget gate, and the single callback the
whole hierarchy eventually reports to.

Split out of the old single-file company.py, which had carried a note since
Phase 0 saying it should become a package once enough neighbours landed. Eight
modules now orbit it, so it has. Employee, Team and the escalation types live
in sibling modules; everything a caller used to import from `llmadapt.company`
is re-exported by this package's __init__, so no import path changed."""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from ..budget import BudgetLedger, CostModel
from ..core import Agent, loop_local_lock, run_coroutine_blocking
from ..hardware import ResourceQuota
from ..router import RoleRank
from ..suggest import did_you_mean
from .employee import Employee
from .escalation import (
    _EMERGENCY_BUDGET_DRAW_FRACTION,
    _EMERGENCY_GRANT_ITERATIONS,
    ESCALATION_PENDING,
    EscalationDecision,
    EscalationEvent,
    EscalationUnresolved,
    RunPaused,
    default_on_escalation,
)
from .team import Team

if TYPE_CHECKING:  # pragma: no cover - types only
    from ..observability import EventLog

logger = logging.getLogger(__name__)


class PolicyMode:
    """The three routing modes a ModelPolicy understands, as named constants
    instead of bare strings you have to spell correctly from memory. "auto"
    lets the effort hint move the local/API threshold; "local" and "api" pin
    the whole decision to one side. Named here (as well as in policy.py)
    because Company is where a caller sets the default and where an
    unrecognized one has to degrade safely.

    Reach for these through the `mode` alias below (`mode.AUTO`), the same
    pattern router.py's `role` uses for RoleRank - see that module's
    docstring for the full reasoning, including why this is a plain class of
    string constants rather than a real `Enum`: every value here still has to
    behave like an ordinary string everywhere it travels (model_map/hire()
    keyword values, JSON, f-string-embedded log/error text), and a real Enum
    only gets that for free if you remember to override `__str__` - one more
    thing to get right for a fixed set of three values that already have
    typo-detection via `did_you_mean()`.
    """

    AUTO = "auto"
    LOCAL = "local"
    API = "api"


POLICY_MODES = (PolicyMode.AUTO, PolicyMode.LOCAL, PolicyMode.API)

# `mode` is PolicyMode under a second, lowercase name - the exact same class,
# not a copy. Deliberately NOT re-exported from the top-level `llmadapt`
# package (only from `llmadapt.company`, where PolicyMode itself lives) -
# "mode" is common enough as a local variable name that handing it out via a
# bare `from llmadapt import *` would recreate the exact collision risk this
# whole naming scheme exists to avoid. `from llmadapt.company import mode` is
# the explicit, deliberate way to get it; `PolicyMode.AUTO` (or the fully
# qualified `llmadapt.PolicyMode.AUTO`, exported at the top level same as
# RoleRank) is the zero-ambiguity spelling for anyone who'd rather not import
# a second name at all.
mode = PolicyMode


@dataclass
class PausedRun:
    """A run that stopped to wait for a human, and everything needed to pick it
    back up.

    Handed back by `Company.run_resumable()` when an on_escalation handler
    returns ESCALATION_PENDING. The company it came from is *still holding* the
    paused work - every agent in the delegation chain kept the turn it was in
    the middle of - so resuming in the same process is just
    `company.resume(paused, decision)`.

    Resuming somewhere else takes one more step, and the reason is the same one
    that shapes state.py: a company holds callables that no file can carry.
    Save the company's state alongside this (`company.save_state(path)`),
    rebuild the company in the new process exactly as it was built the first
    time, `restore_state()` it, and hand it `PausedRun.from_dict(...)`.
    """

    event: EscalationEvent
    task: str
    company_name: str = ""
    entry_point: Optional[str] = None
    paused_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "company_name": self.company_name,
            "entry_point": self.entry_point,
            "paused_at": self.paused_at,
            "event": {
                "kind": self.event.kind, "employee_name": self.event.employee_name,
                "rank": self.event.rank, "message": self.event.message,
                "detail": self.event.detail,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PausedRun":
        data = dict(data or {})
        event = EscalationEvent(**(data.pop("event", {}) or {}))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(event=event, **known)


class Company:
    """Owns the org chart, the emergency escalation reserve, and the single
    callback the whole hierarchy eventually reports to when nothing
    automatic is left to try."""

    def __init__(
        self,
        name: str,
        model_map: Dict[str, Dict[str, Any]],
        on_escalation: Optional[Callable[[EscalationEvent], EscalationDecision]] = None,
        emergency_iteration_reserve: int = 0,
        total_token_budget: Optional[int] = None,
        emergency_budget_tokens: int = 0,
        rank_budget_shares: Optional[Dict[str, float]] = None,
        quota: Optional[ResourceQuota] = None,
        model_policy: Optional[Any] = None,
        presets: Optional[Any] = None,
        log_compaction: Optional[Any] = None,
        default_policy_mode: Optional[str] = None,
        cost_weighted_budget: bool = False,
        cost_model: Optional[Any] = None,
        archive: Optional[Any] = None,
    ):
        """
        model_map: rank -> {"provider": ..., "model": ..., "api_key": ...,
            "mode": "local"|"api"|"auto", "effort": ..., "specialty": ...,
            "cost_per_1k": ..., "alias": ..., "key_env": ...}.
            This is how a per-rank default model is set - {C_SUITE: {"provider":
            "anthropic", "model": "claude-opus-5-5"}, JUNIOR: {"provider":
            "ollama", "model": "llama3.1:8b"}} - and hire() overrides any of it
            for one employee.
            "api_key" is optional and usually better left out: with no key, the
            key is resolved from the environment (or a .env file) by
            env.resolve_api_key. "alias"/"key_env" steer that lookup for
            endpoints that share a provider but not an account - OpenRouter,
            Together and a local vLLM are all provider="openai" to the
            transport, and OPENAI_API_KEY is the wrong key for all of them.
            Per-rank defaults; hire() also takes a per-employee
            provider/model/api_key override for when the ranking-style default
            isn't what a specific employee should use.
            (Phase 4) "mode" is now acted on when a model_policy is attached:
            "local"/"api"/"auto" are handed to ModelPolicy.decide(), which
            consults benchmark.py/selector.py for the local side and its own
            cost/specialty catalog for the API side. Without a model_policy,
            "mode" is still inert and provider/model are taken literally, so
            existing configs behave exactly as before.
        log_compaction: an optional compaction.LogCompactionPolicy. When set
            (and not mode="off"), the activity/tool-call logs are compacted
            after every completed top-level task - the Phase 7 "keep broad
            concepts, drop specifics" pass. Left unset, the logs grow
            unbounded exactly as they did before Phase 7.
        presets: an optional presets.PresetBundle - the four named-preset
            registries (skills, personalities, palettes, org templates) this
            company resolves names against. Defaults to the shared built-in
            registries; pass `default_bundle().fork()` for a private copy that
            custom registrations won't leak out of.
        model_policy: an optional policy.ModelPolicy. When present, any rank
            or employee whose resolved mode is "local"/"api"/"auto" gets its
            provider/model chosen by the policy instead of read from
            model_map. An explicit provider+model (in model_map or at hire())
            always wins over the policy - the policy fills gaps, it doesn't
            override what you asked for.
        on_escalation: called once nothing automatic is left for EITHER
            escalation kind - the relevant reserve is empty, or a retry from
            it still failed. Must return an EscalationDecision. None (the
            default) falls back to escalation.default_on_escalation, which
            declines every escalation - the same "nobody's watching means
            stop" rule build_company()/set_company_up() already applied when
            no handler was given. This used to be a required argument, so
            every hand-written Company(...) call had to spell out a callback
            even for a throwaway example; the fallback keeps that same safe
            behavior without making you write it out. For prototyping on a
            local, no-spend model_map where auto-approving is genuinely
            harmless, escalation.always_approve(...) is a ready-made
            alternative - see its docstring for why it's named that bluntly.
        emergency_iteration_reserve: an integer reserve of *tool iterations*
            (a runaway-loop safety valve, unrelated to cost) drawn down
            automatically for tool_iteration_limit escalations before
            on_escalation is ever invoked. 0 means always ask a human first.
        total_token_budget: the company-wide hard ceiling on total real tokens
            spent (summed across every employee's Agent.total_tokens_used).
            None means unlimited - no budget gate runs at all. Once the
            company's cumulative spend reaches this ceiling, EVERY further
            request for more - internal reallocation included - must go
            through on_escalation; nothing silently crosses it. This is the
            "user stays at the top" cost-control default.
        emergency_budget_tokens: a real-token reserve drawn down automatically
            for budget_exhausted escalations, but ONLY while total spend is
            still below total_token_budget (see above) - once the hard
            ceiling itself is reached, this reserve is skipped too and
            on_escalation is always called. 0 means always ask a human first.
        rank_budget_shares: rank -> relative share of total_token_budget (see
            budget.DEFAULT_RANK_BUDGET_SHARES for the default table and
            budget.BudgetLedger for how a per-employee importance weight
            scales it). Only matters when total_token_budget is set.
        default_policy_mode: the routing mode every hire inherits when neither
            `hire(mode=...)` nor `model_map[rank]["mode"]` says otherwise -
            "auto", "local" or "api". None keeps the pre-existing behaviour
            exactly: no mode unless something names one, so the policy stays
            out of the way.

            `mode="local"`/`"api"` were always available per hire; what was
            missing was saying it once. "This whole company stays on local
            models" previously meant passing mode="local" to every single
            hire() call and remembering to do it for every future one - one
            omission and that employee silently starts spending money, which
            is the failure this whole layer exists to prevent. Precedence is
            narrowest-wins, as everywhere else here: hire(mode=) beats
            model_map[rank]["mode"] beats this.
        cost_weighted_budget: charge the budget in cost-equivalent tokens
            rather than raw ones, so a local employee's free tokens stop
            consuming the same ceiling as a frontier-API employee's. Off by
            default, because turning it on changes what the numbers in
            budget_report() mean and no existing config should shift under
            anyone. See budget.CostModel for where the prices come from - the
            Phase 4 decision, then model_map[rank]["cost_per_1k"], then the
            API catalog, then local detection.
        cost_model: a pre-built budget.CostModel, for full control over the
            baseline and the local/unknown weights. Supplying one implies
            cost_weighted_budget=True.
        archive: an optional archive.RunArchive. Every activity event and tool
            call is appended to it as it happens - before log_compaction can
            collapse anything - so the file stays the untouched record of the
            run even once the in-memory logs are rollups. None (the default)
            writes nothing anywhere. Hand the same RunArchive to an Agent to
            get the conversation transcript in the same file. Call finish()
            at the end to close it.
        """
        self.name = name
        self.model_map = model_map
        self.on_escalation = on_escalation or default_on_escalation
        self.emergency_iteration_reserve = emergency_iteration_reserve
        self._emergency_iteration_remaining = emergency_iteration_reserve
        self.emergency_budget_tokens = emergency_budget_tokens
        self._emergency_budget_spent = 0
        if cost_model is None and cost_weighted_budget:
            cost_model = CostModel(
                api_catalog=getattr(model_policy, "api_catalog", None),
                model_map=model_map,
            )
        self.budget = BudgetLedger(total_token_budget, rank_budget_shares, cost_model=cost_model)
        self.quota = quota
        self.model_policy = model_policy
        self.default_policy_mode = self._validate_policy_mode(default_policy_mode)
        self.log_compaction = log_compaction
        self.archive = archive
        if presets is None:
            from ..presets import default_bundle

            presets = default_bundle()
        self.presets = presets
        self.teams: List[Team] = []
        self.employees: Dict[str, Employee] = {}
        self.activity_log: List[Dict[str, Any]] = []
        self.tool_call_log: List[Dict[str, Any]] = []

    def hire(
        self,
        name: str,
        rank: str,
        reports_to: Optional[Employee] = None,
        system_instruction: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        importance: float = 0.5,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
        mode: Optional[str] = None,
        skills: Sequence[Any] = (),
        personality: Optional[Any] = None,
        cost_weight: Optional[float] = None,
        **agent_kwargs: Any,
    ) -> Employee:
        """Builds an Agent for `rank` from model_map, wraps it as an Employee,
        registers it with the company, and - if reports_to is given - adds a
        delegate_to_<name> tool onto the manager's Agent so it can hand this
        employee tasks through the ordinary tool-calling loop.

        provider/model/api_key: per-employee overrides. The ranking-style
        model_map[rank] default is used unless one of these is given, in
        which case it just changes it for this one employee - nothing else
        about hiring changes.
        importance: this employee's weight in BudgetLedger's percentage-based
        allocation (0.0-1.0, see budget.py) - defaults to a neutral 0.5.
        cost_weight: an explicit multiplier on this employee's token spend,
        used only when the company has cost weighting on (see
        Company(cost_weighted_budget=True) and budget.CostModel). Overrides
        every derived price. Leave it None unless the library has no way to
        know what this employee's model really costs.

        (Phase 4) effort/specialty/mode steer the ModelPolicy, if this Company
        has one:
          effort:    "cheap" | "balanced" | "effort" (aliases like
                     "needs effort" / "keep it cheap" are accepted). Falls
                     back to model_map[rank]["effort"], then to the rank's
                     default in policy.DEFAULT_RANK_EFFORT.
          specialty: a tag matched against ApiModelSpec.specialties, e.g.
                     "code" / "reasoning" / "vision".
          mode:      "auto" | "local" | "api"; falls back to
                     model_map[rank]["mode"]. Anything else (or no policy
                     attached) means "take provider/model literally", which
                     is the pre-Phase-4 behavior.

        Note that effort deliberately does NOT change `importance` - see
        policy.suggested_importance() for the explicit opt-in bridge and
        policy.py's module docstring for why they stay separate knobs.

        (Phase 5) skills/personality are names from this company's
        PresetBundle (or Skill/Personality objects directly, for one-offs that
        need no registration). They are templated into the system instruction
        by presets.compose_system_instruction(), appended after whatever
        `system_instruction` you passed - your text stays first and is never
        rewritten. A skill may also *suggest* an effort/specialty for the
        Phase 4 policy, used only when you did not pass your own.
        """
        if name in self.employees:
            raise ValueError(f"'{name}' is already hired - names must be unique within a Company.")
        rank_config = self.model_map.get(rank, {})
        if not rank_config:
            raise ValueError(f"No model_map entry for rank '{rank}' - add one before hiring for it.")

        effort = effort if effort is not None else rank_config.get("effort")
        specialty = specialty if specialty is not None else rank_config.get("specialty")

        skills = list(skills or rank_config.get("skills", ()))
        personality = personality if personality is not None else rank_config.get("personality")
        if skills or personality:
            from ..presets import compose_system_instruction, skill_hints

            system_instruction = compose_system_instruction(
                base=system_instruction, personality=personality, skills=skills,
                bundle=self.presets, role_line=f"You are {name}, a {rank} at {self.name}.",
            )
            hints = skill_hints(skills, bundle=self.presets)
            # A skill's suggestion is a fallback only - an explicit argument,
            # a rank_config entry, or a rank default all outrank it.
            effort = effort if effort is not None else hints["effort"]
            specialty = specialty if specialty is not None else hints["specialty"]

        resolved_provider = provider or rank_config.get("provider", "ollama")
        resolved_model = model or rank_config.get("model")
        resolved_api_key = api_key or rank_config.get("api_key")
        resolved_base_url = rank_config.get("base_url")
        # Where a missing key is looked up. Both are plain variable *names*,
        # never a key - a model_map that names OPENROUTER_API_KEY can be
        # committed to a repo, which is the whole point of resolving from the
        # environment rather than carrying the secret in config.
        resolved_alias = rank_config.get("alias")
        resolved_key_env = rank_config.get("key_env")
        decision = self._decide_model(
            rank=rank, effort=effort, specialty=specialty, mode=mode,
            explicit_provider=provider or rank_config.get("provider"),
            explicit_model=resolved_model,
            employee_name=name,
        )
        if decision is not None and decision.ok:
            resolved_provider = decision.provider or resolved_provider
            resolved_model = decision.model or resolved_model
            resolved_base_url = decision.base_url or resolved_base_url
            resolved_api_key = resolved_api_key or decision.api_key

        agent = Agent(
            provider=resolved_provider,
            model=resolved_model,
            api_key=resolved_api_key,
            key_alias=resolved_alias,
            key_env=resolved_key_env,
            base_url=resolved_base_url,
            system_instruction=system_instruction,
            # Every employee's Agent writes its transcript to the same
            # archive, so one file holds the company's events and every
            # conversation behind them, in timestamp order. A caller can
            # still override it per hire (archive=None to keep one employee
            # out of the record) through agent_kwargs.
            archive=agent_kwargs.pop("archive", self.archive),
            **agent_kwargs,
        )
        employee = Employee(
            name=name, rank=rank, agent=agent, reports_to=reports_to, importance=importance,
            effort=effort, specialty=specialty, model_decision=decision,
            cost_weight=cost_weight,
        )
        employee.skills = list(skills)
        employee.personality = personality
        if reports_to is not None:
            reports_to.agent.add_tool(employee.delegate_tool(company=self))

        self.employees[name] = employee
        self._log("hire", employee=name, rank=rank, reports_to=(reports_to.name if reports_to else None))
        return employee

    @staticmethod
    def _validate_policy_mode(mode: Optional[str]) -> Optional[str]:
        """Normalize a company-wide default mode, or raise on a bad one.

        This is the one place in the routing path that *does* raise on an
        unrecognized value, and the asymmetry is deliberate. Everywhere else a
        bad mode is per-hire, arrives from a spec or an LLM, and degrades to
        "take model_map literally" - which is the safe direction, since it
        means the caller's own configuration is used. A bad *company default*
        is different: it is set once, by hand, in Python, and silently ignoring
        it would mean a company the caller believes is pinned to local models
        quietly routing to the API for its entire life. Failing at construction
        is the only outcome that can't turn into a bill.
        """
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        if normalized not in POLICY_MODES:
            raise ValueError(
                f"Unknown default_policy_mode {mode!r}"
                f"{did_you_mean(normalized, POLICY_MODES)}. "
                f"Use one of {', '.join(POLICY_MODES)}, or None for no default."
            )
        return normalized

    def resolve_policy_mode(self, *candidates: Optional[str]) -> str:
        """The first usable mode among `candidates`, then the company default.

        Candidates are passed narrowest-first (the hire's own mode, then its
        rank's), so the first one that names something wins and the company
        default is the backstop. Returns "" when nothing names a mode, which
        `_decide_model` reads as "no policy involvement".
        """
        for candidate in candidates:
            normalized = (candidate or "").strip().lower()
            if normalized:
                return normalized
        return self.default_policy_mode or ""

    def _decide_model(
        self,
        rank: str,
        effort: Optional[str],
        specialty: Optional[str],
        mode: Optional[str],
        explicit_provider: Optional[str],
        explicit_model: Optional[str],
        employee_name: str,
    ) -> Optional[Any]:
        """Runs the ModelPolicy for one hire, if a policy is attached and the
        resolved mode asks for one. Returns the PolicyDecision (which the
        caller applies) or None to mean "no policy involvement, use the
        literal model_map values".

        An explicit provider AND model together are treated as "the user
        already decided" and skip the policy entirely - the policy fills gaps
        rather than overruling. A mode of "local"/"api"/"auto" with only one
        of the two given still goes through the policy, since that config is
        asking to be routed.

        A policy that returns kind="unavailable" is logged and then ignored
        (the literal model_map values are used) rather than raising: a routing
        miss at hire time should be visible in the activity log, not fatal
        halfway through building an org chart. It will surface again as a real
        error on the first request, with the provider's own message.
        """
        if self.model_policy is None:
            return None
        rank_config = self.model_map.get(rank, {})
        resolved_mode = self.resolve_policy_mode(mode, rank_config.get("mode"))
        if resolved_mode not in POLICY_MODES:
            return None
        if explicit_provider and explicit_model:
            return None

        decision = self.model_policy.decide(
            rank=rank, effort=effort, specialty=specialty, mode=resolved_mode,
            requested_provider=explicit_provider, requested_model=explicit_model,
        )
        self._log(
            "model_policy", employee=employee_name, rank=rank, mode=resolved_mode,
            decision=decision.kind, effort=decision.effort,
            provider=decision.provider, model=decision.model, reason=decision.reason,
        )
        return decision

    def reassign_model(
        self,
        employee: Employee,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Optional[Any]:
        """Re-run the ModelPolicy for an already-hired employee and swap their
        Agent onto the newly chosen model, in place.

        `mode` defaults to the company's `default_policy_mode`, then to
        "auto". It used to default to "auto" outright, which quietly broke the
        one guarantee a company-wide default is for: a company pinned to
        `default_policy_mode="local"` would stay local at hire time and then
        re-route to the API the moment `run(task, effort=...)` reassigned the
        entry point. A per-task re-route must not be able to cross a line the
        company drew.

        This is what makes the effort hint *per-task* and not merely
        per-employee: `Company.run(task, effort="needs effort")` calls this on
        the entry point before starting. It uses `Agent.change_api()`, so the
        employee keeps their conversation history, their registered tools
        (including every delegate_to_* tool), and their usage counters - only
        the endpoint changes. Returns the PolicyDecision, or None if this
        company has no ModelPolicy.

        Caveat worth stating plainly: switching provider mid-conversation
        re-exports the existing history into the new provider's format
        (Conversation.export_for handles that), but provider-native blocks
        recorded from the old provider are replayed through the generic path.
        For a fresh task on a fresh employee - the intended use - this is
        clean; mid-conversation swaps on a long tool-heavy history are not
        something this has been hardened for.
        """
        if self.model_policy is None:
            return None
        mode = self.resolve_policy_mode(mode) or "auto"
        decision = self.model_policy.decide(
            rank=employee.rank,
            effort=effort if effort is not None else employee.effort,
            specialty=specialty if specialty is not None else employee.specialty,
            mode=mode,
        )
        self._log(
            "model_policy", employee=employee.name, rank=employee.rank, mode=mode,
            decision=decision.kind, effort=decision.effort,
            provider=decision.provider, model=decision.model, reason=decision.reason,
        )
        if not decision.ok:
            return decision
        employee.model_decision = decision
        if effort is not None:
            employee.effort = effort
        if specialty is not None:
            employee.specialty = specialty
        employee.agent.change_api(
            provider=decision.provider or employee.agent.provider,
            model=decision.model,
            base_url=decision.base_url,
            api_key=decision.api_key or employee.agent.api_key,
        )
        return decision

    def add_team(self, team: Team) -> None:
        self.teams.append(team)
        for member in team.members:
            self.employees.setdefault(member.name, member)

    def build_from_template(
        self,
        template: Any,
        size: str = "small",
        name_prefix: str = "",
        review_mode: str = "critique",
        max_review_rounds: int = 1,
        **hire_overrides: Any,
    ) -> Team:
        """Instantiate a whole org shape from a named org template (Phase 5).

        template: a name in this company's org-template registry, or an
            OrgTemplate object.
        size: "small" | "medium" | "large". **Opt-in scaling** - the size is
            an argument, never inferred from a task string. A template's roles
            declare how many copies of themselves exist at each size, so
            "large" typically multiplies the worker tiers while leaving the
            oversight roles alone. Inferring this from the task would mean
            guessing at spend on the user's behalf, which is the thing Phase
            3's budget governance exists to prevent.
        name_prefix: prepended to every employee name, so the same template
            can be instantiated twice in one Company without name collisions
            (names must be unique - hire() enforces that).
        review_mode / max_review_rounds: passed to the Team - see Team's
            docstring for what "review" means and why the round count is
            bounded.
        **hire_overrides: forwarded to every hire() call (e.g. mode="auto" to
            put the whole template under the Phase 4 model policy).

        Returns the Team, already registered with this Company. Its lead is
        the role marked lead=True (falling back to the highest-ranked role),
        and its reviewer is the first role marked reviewer=True. Employees are
        created parents-first so `reports_to` is always already hired.
        """
        template = self.presets.org_templates.resolve(template)
        template.validate()
        roles = template.roles_for(size)
        role_keys = {r.key for r in roles}

        # Parents before children, so reports_to always resolves. A role whose
        # manager doesn't exist at this size reports to that manager's own
        # manager instead (walking up until something exists, or None) - that
        # keeps a template usable at "small" without needing a separate
        # small-only copy of it with rewritten reporting lines.
        def effective_manager(role: Any) -> Optional[str]:
            cursor = role.reports_to
            while cursor is not None and cursor not in role_keys:
                cursor = template.role(cursor).reports_to
            return cursor

        ordered: List[Any] = []
        placed: set = set()
        remaining = list(roles)
        while remaining:
            progressed = False
            for role in list(remaining):
                manager = effective_manager(role)
                if manager is None or manager in placed:
                    ordered.append(role)
                    placed.add(role.key)
                    remaining.remove(role)
                    progressed = True
            if not progressed:  # validate() rules this out, but never loop forever
                raise ValueError(f"template {template.name!r} could not be ordered - check reports_to")

        by_key: Dict[str, Employee] = {}
        for role in ordered:
            manager_key = effective_manager(role)
            manager = by_key.get(manager_key) if manager_key else None
            count = role.count_for(size)
            for index in range(count):
                title = role.display_title()
                suffix = f" {index + 1}" if count > 1 else ""
                employee = self.hire(
                    name=f"{name_prefix}{title}{suffix}",
                    rank=role.rank,
                    reports_to=manager,
                    skills=role.skills,
                    personality=role.personality,
                    effort=role.effort,
                    specialty=role.specialty,
                    importance=role.importance,
                    **hire_overrides,
                )
                if index == 0:
                    by_key[role.key] = employee

        lead_role = next((r for r in ordered if r.lead), None)
        if lead_role is None:
            lead_role = min(ordered, key=lambda r: RoleRank.ORDER.index(r.rank)
                            if r.rank in RoleRank.ORDER else len(RoleRank.ORDER))
        reviewer_role = next((r for r in ordered if r.reviewer), None)

        team = Team(
            name=f"{name_prefix}{template.team_name or template.name}",
            lead=by_key[lead_role.key],
            reviewer=by_key[reviewer_role.key] if reviewer_role else None,
            review_mode=review_mode,
            max_review_rounds=max_review_rounds,
        )
        for role in ordered:
            for employee in self.employees.values():
                if employee not in team.members and employee.name.startswith(f"{name_prefix}{role.display_title()}"):
                    team.add_member(employee)
        self.add_team(team)
        self._log("team_built", employee=team.lead.name, team=team.name,
                  template=template.name, size=size, headcount=len(team.members))
        return team

    def run(
        self,
        task: str,
        entry_point: Optional[Employee] = None,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
    ) -> str:
        """Runs a task starting from entry_point, defaulting to the
        highest-ranked employee registered (typically your C-suite).

        (Phase 4) effort/specialty are the *per-task* form of the model
        policy hint. When either is given and this Company has a ModelPolicy,
        the entry point is re-routed for this task via reassign_model() before
        it starts - so "this one is worth spending on" is a per-call argument,
        not something baked in at hire time. The hint applies to the entry
        point only; employees it delegates to keep their own standing hints,
        since a manager deciding a task is hard doesn't mean every intern
        touching it needs a frontier model.
        """
        return run_coroutine_blocking(
            lambda: self.run_async(task, entry_point=entry_point, effort=effort, specialty=specialty),
            what="Company.run()",
        )

    async def run_async(
        self,
        task: str,
        entry_point: Optional[Employee] = None,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
    ) -> str:
        """The awaitable form of run(), and the real implementation.

        Await this from async code (a web handler, another framework's event
        loop) rather than calling run(): a bare `company.run(task)` inside an
        `async def` still returns the right answer, but it blocks your loop for
        the whole run and warns you about it, because only an `await` at the
        call site can hand control back. See core.run_coroutine_blocking."""
        try:
            return await self._run_entry(task, entry_point, effort, specialty)
        except RunPaused as paused:
            raise RuntimeError(
                f"on_escalation returned ESCALATION_PENDING for {paused.event.employee_name}, but "
                f"run() has nowhere to hand a paused run back to - it returns a string. Use "
                f"run_resumable(), which returns a PausedRun you can later pass to resume(). "
                f"The run has been stopped and nothing was approved."
            ) from paused

    def run_resumable(
        self,
        task: str,
        entry_point: Optional[Employee] = None,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
    ) -> Any:
        """Like run(), but able to come back and say "a human is thinking".

        Returns the finished answer as a string, or a `PausedRun` if an
        on_escalation handler returned ESCALATION_PENDING. Check with
        `isinstance(result, PausedRun)`.

        Separate from run() rather than folded into it on purpose. run() is the
        reason this library is easier to pick up than the frameworks it sits
        next to - `company.run("do the thing")` returns a string, no event loop,
        no result object to unwrap - and changing its return type to
        `str | PausedRun` would put an isinstance check in front of every caller
        who is never going to pause anything. Pausing is opt-in, and so is the
        cost of handling it.
        """
        return run_coroutine_blocking(
            lambda: self.run_resumable_async(task, entry_point=entry_point,
                                             effort=effort, specialty=specialty),
            what="Company.run_resumable()",
        )

    async def run_resumable_async(
        self,
        task: str,
        entry_point: Optional[Employee] = None,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
    ) -> Any:
        """The awaitable form of run_resumable()."""
        try:
            return await self._run_entry(task, entry_point, effort, specialty)
        except RunPaused as paused:
            return PausedRun(
                event=paused.event,
                task=task,
                company_name=self.name,
                entry_point=(entry_point.name if entry_point is not None else None),
                paused_at=time.time(),
            )

    def resume(self, paused: Any, decision: EscalationDecision) -> Any:
        """Picks a paused run back up with the decision a human finally made.

        Returns the finished answer, or another `PausedRun` if the run stopped
        again on a different escalation - a long job may need more than one
        person to say yes, and each one is its own pause.

        A decline raises EscalationUnresolved, exactly as declining in the
        moment would have: waiting and then saying no is still saying no.

        The grant is applied *before* the work restarts, rather than waiting for
        the same escalation to fire a second time. That is what keeps a resumed
        iteration-limit run from paying for its failed attempt twice: the
        employee restarts with the extra room already in hand, so it finishes
        instead of hitting the same wall and asking again.
        """
        return run_coroutine_blocking(
            lambda: self.resume_async(paused, decision),
            what="Company.resume()",
        )

    async def resume_async(self, paused: Any, decision: EscalationDecision) -> Any:
        """The awaitable form of resume()."""
        if decision is ESCALATION_PENDING:
            raise ValueError(
                "resume() needs an actual decision - ESCALATION_PENDING is what got you here. "
                "Approve it, decline it, or leave the run paused and call resume() later."
            )
        if not getattr(decision, "approve", False):
            self._log("escalation_decision", employee=paused.event.employee_name,
                      approved=False, note=getattr(decision, "note", None))
            raise EscalationUnresolved(paused.event, decision)

        self._apply_decision(paused.event, decision)
        self._log("escalation_resumed", employee=paused.event.employee_name,
                  event_kind=paused.event.kind, note=getattr(decision, "note", None))

        entry = self.employees.get(paused.entry_point) if paused.entry_point else None
        try:
            return await self._run_entry(paused.task, entry, None, None)
        except RunPaused as again:
            return PausedRun(
                event=again.event,
                task=paused.task,
                company_name=self.name,
                entry_point=paused.entry_point,
                paused_at=time.time(),
            )

    async def _run_entry(
        self,
        task: str,
        entry_point: Optional[Employee] = None,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
    ) -> str:
        """The actual run, shared by run_async(), run_resumable_async() and
        resume_async(). Lets RunPaused through untouched - each of those three
        has a different, correct thing to do with it, and none of them should
        be reimplementing the run to get it."""
        start = entry_point or self._default_entry_point()
        if start is None:
            raise ValueError("Company has no employees - call hire() before run().")
        if (effort is not None or specialty is not None) and self.model_policy is not None:
            self.reassign_model(start, effort=effort, specialty=specialty)
        self._log("task_start", employee=start.name, task=task[:200])
        result = await start.run_async(task, company=self)
        self._log("task_end", employee=start.name)
        self.compact_logs()
        return result

    def run_structured(
        self,
        task: str,
        strategy: str = "direct",
        entry_point: Optional[Employee] = None,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
        **strategy_kwargs: Any,
    ) -> Any:
        """(Phase 6) Run a task through a decomposition strategy.

        strategy="direct"    - exactly `run()`: the entry point handles it,
                               delegating through its normal tools. Returns a
                               string.
        strategy="plan"      - plan-then-execute (delegation.plan_then_execute).
                               Returns a PlanResult.
        strategy="stub_fill" - stub-and-fill (delegation.stub_and_fill). The
                               senior tier writes signatures + docstrings, the
                               cheap tiers write bodies. Returns a
                               StubAndFillResult.

        The two structured strategies return result *objects*, not strings,
        because "did every step succeed?" and "which stubs came back unfilled?"
        are the questions a caller actually has, and a string answer can't
        carry them. `.answer` / `.code` hold the deliverable.

        effort/specialty behave as in run(): they re-route the entry point
        (here, the architect/planner) through the Phase 4 policy for this task.
        """
        return run_coroutine_blocking(
            lambda: self.run_structured_async(
                task, strategy=strategy, entry_point=entry_point, effort=effort,
                specialty=specialty, **strategy_kwargs),
            what="Company.run_structured()",
        )

    async def run_structured_async(
        self,
        task: str,
        strategy: str = "direct",
        entry_point: Optional[Employee] = None,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
        **strategy_kwargs: Any,
    ) -> Any:
        """The awaitable form of run_structured(). Same strategies, same
        return types - see run_structured() for what each one means."""
        start = entry_point or self._default_entry_point()
        if start is None:
            raise ValueError("Company has no employees - call hire() before run_structured().")
        if (effort is not None or specialty is not None) and self.model_policy is not None:
            self.reassign_model(start, effort=effort, specialty=specialty)

        if strategy == "direct":
            return await self.run_async(task, entry_point=start)

        from .. import delegation

        self._log("strategy_start", employee=start.name, strategy=strategy, task=task[:200])
        if strategy == "plan":
            result = await delegation.plan_then_execute_async(self, task, planner=start, **strategy_kwargs)
        elif strategy == "stub_fill":
            result = await delegation.stub_and_fill_async(self, task, architect=start, **strategy_kwargs)
        else:
            raise ValueError(
                f"unknown strategy {strategy!r} - use 'direct', 'plan', or 'stub_fill'"
            )
        self._log("strategy_done", employee=start.name, strategy=strategy, ok=result.ok)
        self.compact_logs()
        return result

    def _default_entry_point(self) -> Optional[Employee]:
        for rank in RoleRank.ORDER:
            for employee in self.employees.values():
                if employee.rank == rank:
                    return employee
        return next(iter(self.employees.values()), None)

    def _handle_top_level_escalation(self, event: EscalationEvent, task: str) -> str:
        return run_coroutine_blocking(
            lambda: self._handle_top_level_escalation_async(event, task),
            what="Company._handle_top_level_escalation()",
        )

    async def _handle_top_level_escalation_async(self, event: EscalationEvent, task: str) -> str:
        """Dispatches to the kind-specific handler. Both kinds share the same
        overall shape (try an automatic reserve first, only then call
        on_escalation) but draw from genuinely different reserves and mean
        different things when approved, so they're separate methods rather
        than one branchy one."""
        self._log("escalation", employee=event.employee_name, event_kind=event.kind, message=event.message)
        if event.kind == "budget_exhausted":
            return await self._handle_budget_escalation(event, task, hard_ceiling=(event.detail == "hard_ceiling"))
        return await self._handle_iteration_escalation(event, task)

    async def _call_on_escalation(self, event: EscalationEvent,
                                  task: str = "") -> EscalationDecision:
        """Invokes this company's on_escalation handler, whatever shape it is.

        A handler is allowed to be `async def` (await it) or an ordinary
        function (run it on a worker thread). The thread matters: the whole
        point of an escalation handler is that it may sit there waiting on a
        human - an `input()` prompt, an HTTP call to a review queue - and doing
        that inline on the event loop would freeze every other employee still
        working, which is precisely the thing they were made concurrent to
        avoid. A sync handler that returns an awaitable is awaited too, so a
        lambda handing back a coroutine behaves the way it reads.
        """
        handler = self.on_escalation
        if inspect.iscoroutinefunction(handler):
            decision = await handler(event)
        else:
            decision = await asyncio.to_thread(handler, event)
            if inspect.isawaitable(decision):
                decision = await decision

        if decision is ESCALATION_PENDING:
            # Not a decision - a request to stop and wait. Unwinds as a
            # RunPaused, which each Agent on the way out uses as its cue to
            # record the turn it was in the middle of.
            self._log("escalation_paused", employee=event.employee_name, event_kind=event.kind)
            raise RunPaused(event, task)
        return decision

    async def _handle_iteration_escalation(self, event: EscalationEvent, task: str) -> str:
        if self._emergency_iteration_remaining > 0:
            grant = min(self._emergency_iteration_remaining, _EMERGENCY_GRANT_ITERATIONS)
            self._emergency_iteration_remaining -= grant
            self._log("emergency_reserve_used", employee=event.employee_name, iterations_granted=grant,
                       remaining=self._emergency_iteration_remaining)
            employee = self.employees.get(event.employee_name)
            if employee is not None:
                employee.agent.set_max_tool_iterations(employee.agent.max_tool_iterations + grant)
                try:
                    return await employee.agent.chat_async(task)
                except RuntimeError:
                    pass  # still failing even with the extra room - fall through to the human

        decision = await self._call_on_escalation(event, task)
        self._log("escalation_decision", employee=event.employee_name, approved=decision.approve, note=decision.note)
        if not decision.approve:
            raise EscalationUnresolved(event, decision)

        employee = self.employees.get(event.employee_name)
        if employee is not None and self._apply_decision(event, decision):
            return await employee.agent.chat_async(task)

        return f"[Escalation for {event.employee_name} was approved, but no concrete recovery action was given.]"

    async def _handle_budget_escalation(self, event: EscalationEvent, task: str, hard_ceiling: bool = False) -> str:
        """hard_ceiling=True means total_token_budget itself has been
        reached (see _budget_gate) - emergency_budget_tokens is skipped
        entirely in that case (it exists for an individual employee/rank
        running dry while the company as a whole still has room, not for
        crossing the company-wide ceiling), and on_escalation is always
        called."""
        if not hard_ceiling:
            reserve_left = self.emergency_budget_tokens - self._emergency_budget_spent
            if reserve_left > 0:
                grant = min(reserve_left, max(1, int(reserve_left * _EMERGENCY_BUDGET_DRAW_FRACTION)))
                self._emergency_budget_spent += grant
                employee = self.employees.get(event.employee_name)
                if employee is not None:
                    self.budget.grant_bonus(employee.name, grant)
                    self._log("emergency_budget_used", employee=event.employee_name, tokens_granted=grant,
                               remaining=self.emergency_budget_tokens - self._emergency_budget_spent)
                    try:
                        return await employee.agent.chat_async(task)
                    except RuntimeError:
                        pass  # still failing (e.g. hit its iteration cap too) - fall through to the human

        decision = await self._call_on_escalation(event, task)
        self._log("escalation_decision", employee=event.employee_name, approved=decision.approve, note=decision.note)
        if not decision.approve:
            raise EscalationUnresolved(event, decision)

        employee = self.employees.get(event.employee_name)
        if employee is None:
            return f"[Escalation for {event.employee_name} was approved, but no concrete recovery action was given.]"

        self._apply_decision(event, decision, hard_ceiling=hard_ceiling)
        return await employee.agent.chat_async(task)

    def _apply_decision(self, event: EscalationEvent, decision: EscalationDecision,
                        hard_ceiling: Optional[bool] = None) -> bool:
        """Turns an approved EscalationDecision into the concrete change it
        promises, and reports whether it actually granted anything.

        One definition, three callers: the two escalation handlers, and
        resume(), which has to make the same grant hours later and in a
        different process. Two of those writing out the arithmetic separately
        is how a resumed run would end up with a subtly different budget than
        an approved-in-the-moment one.

        Kept strictly per-kind, matching what each handler did before it was
        extracted: an iteration-limit event spends extra_tool_iterations, a
        budget event spends extra_token_budget. Cross-applying them (handing
        tokens to a runaway loop, or iterations to an employee that is simply
        out of money) would be a change of behaviour dressed up as a cleanup.
        """
        employee = self.employees.get(event.employee_name)
        if employee is None or not decision.approve:
            return False

        if event.kind == "budget_exhausted":
            if hard_ceiling is None:
                hard_ceiling = event.detail == "hard_ceiling"
            if decision.extra_token_budget:
                self.budget.grant_bonus(employee.name, decision.extra_token_budget)
            if hard_ceiling:
                # Human approved crossing the company-wide ceiling itself, not just
                # this employee's rank share - raise the ceiling by the granted
                # amount (or a nominal 1 token if none given, so the retry isn't
                # immediately re-gated at the exact same ceiling).
                self.budget.total_token_budget = (
                    (self.budget.total_token_budget or 0) + (decision.extra_token_budget or 1))
            return True

        if decision.extra_tool_iterations:
            employee.agent.set_max_tool_iterations(
                employee.agent.max_tool_iterations + decision.extra_tool_iterations)
            return True
        return False

    def _budget_gate(self, employee: Employee, task: str) -> Optional[str]:
        return run_coroutine_blocking(
            lambda: self._budget_gate_async(employee, task),
            what="Company._budget_gate()",
        )

    async def _budget_gate_async(self, employee: Employee, task: str) -> Optional[str]:
        """Pre-flight check run by Employee.run() before it calls
        agent.chat(). Returns None to let run() proceed normally, or a
        finished result string if a budget escalation had to happen (and got
        resolved) instead.

        Two distinct checks, in order:
        1. The company-wide hard ceiling (total_token_budget) - once total
           spend reaches it, nothing proceeds without on_escalation; internal
           reallocation can never cross this line, by design.
        2. This employee's own rank-based allocation - if it's used up,
           try borrowing slack from the reports_to chain (_try_reallocate)
           before escalating.

        The decision half is serialized by a company-wide lock, because both
        checks are read-then-act and the acting half changes shared state:
        _try_reallocate reads a manager's slack and then grants against it, and
        two employees gated at the same instant would otherwise each read the
        same slack and each be granted it.

        The lock is held across the *decision only*, never across the
        escalation that may follow it. That is deliberate and load-bearing: an
        approved escalation retries the employee's work, that work can delegate,
        and a delegation gates - so holding the lock across it would have the
        same task waiting on a lock it is already holding, which is a deadlock
        with no timeout and no error, just a company that stops. Hence the
        decision is a plain synchronous method with no awaits in it (which is
        also what makes it atomic here in the first place) and the escalation
        happens after the lock is released.

        The lock is *not* a fix for the wider check-then-charge gap - the charge
        only lands once the model has answered, so an employee can still pass a
        gate with room to spare and then overspend it. That was equally true
        before anything ran concurrently (nothing stops one employee passing
        with 1 token left and then spending 5000), and closing it properly needs
        a reservation rather than a lock; the company-wide ceiling in the first
        check remains the real backstop.
        """
        if self.budget.total_token_budget is None:
            return None  # unlimited - no budget configured, gate is a no-op

        async with loop_local_lock(self, "_budget_lock"):
            event = self._budget_decision(employee)

        if event is None:
            return None  # cleared the gate - let run() proceed with a normal chat() call
        return await employee._escalate_async(event, task, self)

    def _budget_decision(self, employee: Employee) -> Optional[EscalationEvent]:
        """The read-then-act half of the gate, run under the budget lock.

        Synchronous on purpose: with no await inside it, it cannot be
        interleaved with another employee's gate on the same event loop, and it
        cannot re-enter the lock. Returns None if the employee may proceed, or
        the EscalationEvent the caller should raise once the lock is released.
        """
        total_spent = self.total_tokens_spent()
        if total_spent >= self.budget.total_token_budget:
            return EscalationEvent(
                kind="budget_exhausted",
                employee_name=employee.name,
                rank=employee.rank,
                message=(f"{employee.name} ({employee.rank}) can't start a new task - the company-wide token "
                         f"budget ({self.budget.total_token_budget}) has already been reached ({total_spent} spent)."),
                detail="hard_ceiling",
            )

        charged = self.budget.charged_spend(employee)
        remaining = self.budget.remaining(employee.rank, employee.name, employee.importance, charged)
        if remaining is not None and remaining <= 0:
            if self._try_reallocate(employee):
                return None  # borrowed some room
            return EscalationEvent(
                kind="budget_exhausted",
                employee_name=employee.name,
                rank=employee.rank,
                message=(f"{employee.name} ({employee.rank}) has used its full token allocation "
                         f"({charged} tokens) and no manager up the chain had slack to lend."),
                detail="rank_allocation",
            )

        return None

    def _try_reallocate(self, employee: Employee) -> bool:
        """Simplified (non-water-filling) reallocation for v1: walk the
        employee's reports_to chain looking for anyone with unspent
        allocation to spare, and if found, grant the employee a bonus
        conceptually drawn from that slack. This does NOT literally debit the
        lender's own ledger entry - see BudgetLedger's docstring in budget.py
        for why that's a safe simplification here (it only ever adds
        headroom, so the company-wide hard ceiling in _budget_gate is the
        actual backstop, not this bookkeeping). Returns True if a bonus was
        granted (caller should retry), False if nobody up the chain had room."""
        manager = employee.reports_to
        while manager is not None:
            mgr_remaining = self.budget.remaining(
                manager.rank, manager.name, manager.importance,
                self.budget.charged_spend(manager),
            )
            if mgr_remaining is not None and mgr_remaining > 0:
                # Lend at most half the manager's own remaining slack, so one
                # starved report can't fully consume a manager's future headroom.
                grant = max(1, mgr_remaining // 2)
                self.budget.grant_bonus(employee.name, grant)
                self._log("budget_reallocated", employee=employee.name, from_employee=manager.name, amount=grant)
                return True
            manager = manager.reports_to
        return False

    def total_tokens_spent(self) -> int:
        """Spend so far as the budget counts it, summed across every hired
        employee - the number _budget_gate checks against total_token_budget.

        Raw `Agent.total_tokens_used` by default. With cost weighting on, this
        is cost-equivalent tokens; `total_raw_tokens_spent()` is the
        unweighted figure that reconciles against a provider's usage page.
        """
        return sum(self.budget.charged_spend(e) for e in self.employees.values())

    def pin_context(self, text: str, employees: Optional[Sequence[str]] = None,
                    reason: str = "") -> List[str]:
        """Pin a fact onto every employee (or the named ones), returning who got it.

        The company-wide case of Employee.pin_context: a constraint the whole
        org has to respect - a deadline, an interface everything must target,
        a rule that came out of an escalation - which would otherwise have to
        be repeated into every delegated task and would still be summarized
        away in each employee's own history.

        Logged as an activity event, because "who was told what, and when" is a
        question the log is supposed to answer, and a pin is exactly the kind
        of standing instruction whose provenance matters later.
        """
        targets = list(employees) if employees is not None else list(self.employees)
        pinned: List[str] = []
        for name in targets:
            employee = self.employees.get(name)
            if employee is None:
                logger.warning("pin_context: no employee named %r%s", name,
                                did_you_mean(name, self.employees))
                continue
            employee.pin_context(text, reason=reason)
            pinned.append(name)
        self._log("pinned_context", employees=pinned, reason=reason,
                  preview=text[:200])
        return pinned

    def key_sources(self) -> Dict[str, Dict[str, Any]]:
        """Where each employee's API key came from, per employee.

        Never returns a key - only its provenance, and the list of variable
        names that were tried when none was found. A missing key is the most
        common setup failure in this library, and "no key" on its own does not
        tell you which name you were supposed to set. This is the answer to
        "why is it saying I have no key when I definitely set one".
        """
        from ..env import key_env_candidates

        out: Dict[str, Dict[str, Any]] = {}
        for name, employee in self.employees.items():
            agent = employee.agent
            out[name] = {
                "provider": agent.provider,
                "model": agent.model,
                "has_key": bool(agent.api_key),
                "source": getattr(agent, "api_key_source", "unknown"),
                "tried": key_env_candidates(
                    provider=agent.provider,
                    alias=getattr(agent, "key_alias", None),
                    key_env=getattr(agent, "key_env", None),
                ),
            }
        return out

    def total_raw_tokens_spent(self) -> int:
        """Real tokens spent, never cost-weighted."""
        return sum(e.agent.total_tokens_used for e in self.employees.values())

    def budget_report(self) -> Dict[str, Any]:
        """A snapshot of company-wide + per-employee budget state - allocated
        (None means unlimited), spent, and remaining for each employee, plus
        the emergency reserve's own state."""
        return {
            "total_token_budget": self.budget.total_token_budget,
            "total_spent": self.total_tokens_spent(),
            "total_raw_tokens": self.total_raw_tokens_spent(),
            "cost_weighted": self.budget.cost_model is not None,
            "emergency_budget_tokens": self.emergency_budget_tokens,
            "emergency_budget_spent": self._emergency_budget_spent,
            "employees": self.budget.report(self.employees),
        }

    def compact_logs(self, policy: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """(Phase 7) Run the log compaction pass. Returns a report, or None if
        no policy is configured.

        Called automatically at the end of every completed top-level task
        (`run()` / `run_structured()`), which is when the roadmap said the pass
        should happen - a task boundary is the point at which the specifics of
        how it got done stop mattering and the broad shape is what you keep.
        Call it directly to compact on demand.

        Compaction is lossy and irreversible on the live lists. The escalation
        and budget-authority events are protected from it at every mode - see
        compaction.ALWAYS_KEEP_KINDS.
        """
        policy = policy or self.log_compaction
        if policy is None:
            return None
        report = policy.compact_company(self)
        if report.get("compacted"):
            self._log("logs_compacted", employee=None,
                       events_dropped=report.get("events_dropped"),
                       tool_calls_dropped=report.get("tool_calls_dropped"))
        return report

    def _log(self, kind: str, **fields: Any) -> None:
        entry = {"time": time.time(), "kind": kind, **fields}
        self.activity_log.append(entry)
        # Write-through: archived here, where the event happens, rather than
        # when compaction runs. Compaction is lossy and the archive has to
        # predate it, and a run that dies before the first compaction should
        # still leave everything it did on disk.
        if self.archive is not None:
            self.archive.append("activity", dict(entry))

    def save_state(self, path: Optional[str] = None, notes: str = "",
                   paused: Optional["PausedRun"] = None) -> Any:
        """A snapshot of where this company currently is, as plain data.

        Returns a `state.CompanyState`; if `path` is given, also writes it
        there as JSON. Safe to commit or send somewhere - it carries provider
        and model names but never an API key (state.py checks that on the way
        out rather than trusting it).

        Pass `paused=` the PausedRun from run_resumable() to keep the two
        together in one document - "where are we" and "what are we waiting for"
        are the same question when a run stopped for a human, and two files
        somebody has to remember to pair up is one file too many.

        This is not the same thing as the run archive, and does not replace it:
        the archive says what happened, this says where things stand. See
        state.py's module docstring for why neither is derived from the other.
        """
        from ..state import capture_state

        snapshot = capture_state(self, notes=notes)
        if paused is not None:
            snapshot.paused_run = paused.to_dict()
        if path is not None:
            snapshot.save(path)
        return snapshot

    def restore_state(self, state: Any, strict: bool = True) -> Any:
        """Applies a snapshot taken by save_state() back onto this company.

        Accepts a `CompanyState`, a path to a saved one, or its plain dict.

        A snapshot cannot rebuild a company on its own - a company holds
        callables (the escalation handler, every registered tool) that no data
        file can carry. Build the company first, exactly as you built it
        originally, then restore onto it. Every structural fact that *can* be
        checked is: roster, ranks, reporting lines and registered tool names.
        A mismatch raises `state.StateMismatch` listing all of them, rather
        than restoring the parts that happen to line up.
        """
        from ..state import CompanyState, apply_state

        if isinstance(state, str):
            state = CompanyState.load(state)
        elif isinstance(state, dict):
            state = CompanyState.from_dict(state)
        return apply_state(self, state, strict=strict)

    def finish(self, clean: bool = True) -> None:
        """End-of-run cleanup: close the archive, if there is one.

        `clean` is what decides whether a `delete_on_clean_exit` archive
        actually deletes itself, and it is narrowed here rather than trusted -
        a run that ended with an escalation nobody resolved is not a clean run,
        and that is the single case where you most want the file kept. Nothing
        else about the company is torn down; calling this twice is harmless.
        """
        if self.archive is None:
            return
        if clean:
            unresolved = [
                event for event in self.activity_log
                if event.get("kind") == "escalation" and not event.get("resolved", True)
            ]
            clean = not unresolved
        self.archive.close(clean=clean)

    def __enter__(self) -> "Company":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(clean=exc_type is None)

    def org_chart(self) -> Dict[str, Any]:
        """A nested view of who reports to whom - the data render_org_chart()
        and the standalone observability.render_org_chart_* functions draw
        from."""

        def node(employee: Employee) -> Dict[str, Any]:
            return {
                "name": employee.name,
                "rank": employee.rank,
                "reports": [node(sub) for sub in employee.subordinates],
            }

        roots = [e for e in self.employees.values() if e.reports_to is None]
        return {"company": self.name, "org": [node(r) for r in roots]}

    def render_org_chart(self, fmt: str = "ascii", theme: str = "light", palette: Optional[Any] = None) -> str:
        """Renders the current org chart as text or a graphic.

        fmt="ascii"   - a tree-command-style text rendering, zero deps, prints
                         straight to a terminal or log (the default).
        fmt="mermaid" - Mermaid `graph TD` text, pastes directly into a
                         GitHub/GitLab markdown file or any Mermaid renderer.
        fmt="svg"     - a self-contained SVG tree diagram (no JS, no external
                         renderer) - write the returned string to a .svg file
                         and open it in any browser. theme="dark" switches to
                         the dark-surface palette.
        palette       - (Phase 5) a name from this company's palette registry
                         ("dataviz" default, plus "grayscale"/"ocean"/"ember"),
                         or a Palette object. Resolved through the same
                         PresetRegistry mechanism as skills and templates -
                         palettes are not a separate system. Ignored by
                         fmt="ascii", which has no colors.
        """
        from .. import observability

        chart = self.org_chart()
        if palette is not None:
            palette = self.presets.palettes.resolve(palette)
        if fmt == "ascii":
            return observability.render_org_chart_ascii(chart)
        if fmt == "mermaid":
            return observability.render_org_chart_mermaid(chart, theme=theme, palette=palette)
        if fmt == "svg":
            return observability.render_org_chart_svg(chart, theme=theme, palette=palette)
        raise ValueError(f"Unknown render_org_chart fmt '{fmt}' - use 'ascii', 'mermaid', or 'svg'.")

    def activity(self) -> "EventLog":
        """A queryable view over activity_log (hire/task/escalation lifecycle
        events) - see observability.EventLog for by_employee()/by_kind()/since()."""
        from ..observability import EventLog

        return EventLog(self.activity_log)

    def tool_calls(self) -> "EventLog":
        """A queryable view over tool_call_log (every delegation call so far -
        the args, a result preview, duration, and any error). Stored for
        later reference, not consulted automatically by anything yet."""
        from ..observability import EventLog

        return EventLog(self.tool_call_log)


__all__ = ["Company"]

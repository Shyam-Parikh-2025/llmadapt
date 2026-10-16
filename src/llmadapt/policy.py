"""policy.py - Phase 4: the auto-mode local-vs-API model policy.

`selector.py` + `router.allocate_local_auto()` already rank *local* models
against this machine's real benchmark numbers. What was missing was the layer
above that: given a rank, an optional effort/priority hint, and an optional
specialty, decide whether this employee should run on a local model at all -
and if not, which API model to buy instead.

Three deliberate scope decisions, all per the roadmap:

1. **The API model table is user-supplied config, not a live pricing fetch.**
   `DEFAULT_API_CATALOG` below is a small seed table so the module is useful
   out of the box, but its prices and capability numbers are hand-entered and
   **will go stale** - they are a starting point to edit, not a source of
   truth. Nothing in llmadapt phones home to refresh them (that was flagged
   as a scope trap: pricing pages change shape constantly and it would drag a
   scraper + a cache + a staleness policy into a zero-dependency library).
   Pass your own `ApiModelCatalog` to get numbers you trust.

2. **The effort hint is an input from the start, not a later bolt-on.** Every
   entry point here - `ModelPolicy.decide()`, `Company.hire()`,
   `Company.run()` - takes `effort=` alongside `rank`. It is what moves the
   local-vs-API threshold: "cheap" work stays local if anything local fits at
   all, "effort" work goes to the strongest API model configured.

3. **Effort does NOT silently feed `BudgetLedger.importance`** (roadmap open
   question #1, resolved here). They answer different questions - effort is
   "how capable a model does this need?", importance is "how large a slice of
   the company's token pool does this employee get?". Coupling them
   automatically would quietly redefine what `importance` means in Phase 3's
   ledger and make budget reports impossible to reason about. Instead there
   is an explicit, opt-in bridge - `suggested_importance(effort)` - that a
   caller can pass into `hire(importance=...)` themselves if they *do* want
   the two to move together. Explicit beats implicit here because the ledger
   is the thing standing between this library and a surprise bill.

Everything in this module is a pure decision function over data structures:
no network calls, no model loading. That is what makes it testable offline
(see tests/test_policy.py) and what lets `Company` consult it at hire time
without paying for anything.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .benchmark import BenchmarkResult
from .hardware import ResourceQuota
from .router import ModelRouter, RoleRank
from .selector import LocalModelCandidate, ModelCatalog, SelectionResult
from .suggest import did_you_mean

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Effort hints
# ---------------------------------------------------------------------------

# The three canonical effort levels. Kept to three on purpose: this hint exists
# to move a threshold, and a user cannot meaningfully calibrate seven levels of
# "how hard should this try" against a model catalog they wrote by hand.
EFFORT_CHEAP = "cheap"
EFFORT_BALANCED = "balanced"
EFFORT_EFFORT = "effort"

EFFORT_LEVELS = (EFFORT_CHEAP, EFFORT_BALANCED, EFFORT_EFFORT)

# Phrases a caller (or an LLM filling in a tool schema) might plausibly use.
# Accepting these is not "natural language parsing" - it is a small fixed
# alias table, so `effort="keep it cheap"` doesn't silently become balanced.
_EFFORT_ALIASES: Dict[str, str] = {
    "cheap": EFFORT_CHEAP,
    "keep it cheap": EFFORT_CHEAP,
    "low": EFFORT_CHEAP,
    "fast": EFFORT_CHEAP,
    "economy": EFFORT_CHEAP,
    "balanced": EFFORT_BALANCED,
    "medium": EFFORT_BALANCED,
    "normal": EFFORT_BALANCED,
    "default": EFFORT_BALANCED,
    "effort": EFFORT_EFFORT,
    "needs effort": EFFORT_EFFORT,
    "high": EFFORT_EFFORT,
    "high effort": EFFORT_EFFORT,
    "best": EFFORT_EFFORT,
    "quality": EFFORT_EFFORT,
}

# Which effort level a rank implies when nobody passed one. Mirrors the
# spirit of router.allocate_compression_policy's rank table: the worker tiers
# churn the most tokens and should stay cheap; the ranks that make decisions
# rather than volume are worth spending on.
DEFAULT_RANK_EFFORT: Dict[str, str] = {
    RoleRank.C_SUITE: EFFORT_EFFORT,
    RoleRank.GENERAL_MANAGER: EFFORT_EFFORT,
    RoleRank.MANAGER: EFFORT_BALANCED,
    RoleRank.SENIOR: EFFORT_BALANCED,
    RoleRank.JUNIOR: EFFORT_CHEAP,
    RoleRank.INTERN: EFFORT_CHEAP,
    RoleRank.VOLUNTEER: EFFORT_CHEAP,
}


def normalize_effort(effort: Optional[str], rank: Optional[str] = None) -> str:
    """Resolves an effort hint to one of EFFORT_LEVELS.

    None (or an unrecognized string) falls back to the rank's default, and
    then to "balanced" if the rank is unknown too. Unrecognized strings
    deliberately fall back rather than raising - this value can come from an
    LLM filling in a tool call (Phase 8), and a typo there should degrade to
    a sensible default, not blow up a running company.
    """
    if effort is not None:
        key = str(effort).strip().lower()
        if key in _EFFORT_ALIASES:
            return _EFFORT_ALIASES[key]
    if rank is not None and rank in DEFAULT_RANK_EFFORT:
        return DEFAULT_RANK_EFFORT[rank]
    return EFFORT_BALANCED


def suggested_importance(effort: Optional[str], rank: Optional[str] = None) -> float:
    """The opt-in bridge between this module and `BudgetLedger.importance`.

    Nothing calls this automatically - see the module docstring for why. A
    caller who wants high-effort employees to also get a bigger token slice
    writes it out explicitly:

        eff = "needs effort"
        company.hire("Ada", RoleRank.SENIOR, effort=eff,
                     importance=suggested_importance(eff))

    Returns a value in [0, 1] suitable for `hire(importance=...)`, on the same
    0.5x-1.5x scaling curve BudgetLedger already documents.
    """
    level = normalize_effort(effort, rank)
    return {EFFORT_CHEAP: 0.25, EFFORT_BALANCED: 0.5, EFFORT_EFFORT: 0.85}[level]


# ---------------------------------------------------------------------------
# API model catalog
# ---------------------------------------------------------------------------


@dataclass
class ApiModelSpec:
    """One API model this company is allowed to buy.

    cost fields are USD per 1,000 tokens. `capability` is a coarse 0.0-1.0
    strength score - deliberately a single hand-set number rather than a
    benchmark suite, because the honest alternative (running evals) is out of
    scope for a config table and any published leaderboard number would go
    stale just as fast as the prices. Treat it as "how much do I trust this
    model relative to the others *I* listed".

    `specialties` is a free-form tag list ("code", "reasoning", "vision",
    "long_context", "writing", ...). Nothing validates the tags; they only
    have to be consistent within one catalog, since matching is exact-string.
    """

    name: str
    provider: str
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capability: float = 0.5
    specialties: Sequence[str] = field(default_factory=tuple)
    context_window: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    note: str = ""

    def blended_cost_per_1k(self, output_ratio: float = 0.75) -> float:
        """A single comparable price per 1k tokens.

        Real spend depends on the input:output mix of the actual traffic,
        which nothing here can know in advance, so this collapses the two
        prices with a fixed assumed ratio (default: 75% of billed tokens are
        output, since agent turns tend to be short prompts and long
        completions once tool results are compressed). Override output_ratio
        on ModelPolicy if your traffic looks different. This is a *ranking*
        aid, not a spend forecast.
        """
        ratio = max(0.0, min(1.0, output_ratio))
        return self.cost_per_1k_input * (1.0 - ratio) + self.cost_per_1k_output * ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "capability": self.capability,
            "specialties": list(self.specialties),
            "context_window": self.context_window,
            "note": self.note,
        }


# A small seed table so `ModelPolicy()` does something sensible with no config.
#
# !! These numbers are hand-entered reference points, not live pricing. They
# !! were correct-ish when written and will drift. Nothing refreshes them.
# !! Override with your own ApiModelCatalog for anything cost-sensitive.
#
# api_key is left None throughout: keys come from the Company's model_map or
# the provider's standard environment variable (see Agent.change_api), never
# from a table checked into a repo.
DEFAULT_API_CATALOG: Tuple[ApiModelSpec, ...] = (
    # Prices are the providers' published standard rates as of October 2026
    # (USD per 1K tokens = per-1M price / 1000). Capability is a coarse
    # hand-assigned ranking, not a benchmark score.
    ApiModelSpec(
        name="claude-opus-5-5", provider="anthropic",
        cost_per_1k_input=0.004, cost_per_1k_output=0.020, capability=0.97,
        specialties=("reasoning", "code", "writing", "long_context", "vision"),
        context_window=1000000, note="seed default (Oct 2026 pricing) - verify before relying on it",
    ),
    ApiModelSpec(
        name="claude-sonnet-5-5", provider="anthropic",
        cost_per_1k_input=0.002, cost_per_1k_output=0.010, capability=0.92,
        specialties=("reasoning", "code", "writing", "long_context", "vision"),
        context_window=1000000, note="seed default (Oct 2026 pricing) - verify before relying on it",
    ),
    ApiModelSpec(
        name="claude-haiku-5-5", provider="anthropic",
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0005, capability=0.72,
        specialties=("code", "writing", "long_context"),
        context_window=1000000,
        note="seed default (Oct 2026 pricing; prompts over 100K tokens cost 5x) - verify before relying on it",
    ),
    ApiModelSpec(
        name="gpt-5.4-mini", provider="openai",
        cost_per_1k_input=0.00075, cost_per_1k_output=0.0045, capability=0.74,
        specialties=("code", "writing", "reasoning"),
        context_window=400000, note="seed default (Oct 2026 pricing) - verify before relying on it",
    ),
    ApiModelSpec(
        name="gemini-3.6-flash", provider="gemini",
        cost_per_1k_input=0.00075, cost_per_1k_output=0.00375, capability=0.78,
        specialties=("reasoning", "vision", "long_context", "writing"),
        context_window=1000000,
        note="seed default (Oct 2026 pricing; Google doubles it on 2027-01-01) - verify before relying on it",
    ),
    ApiModelSpec(
        name="gemini-3.5-flash-lite", provider="gemini",
        cost_per_1k_input=0.0003, cost_per_1k_output=0.0025, capability=0.60,
        specialties=("writing", "long_context"),
        context_window=1000000, note="seed default (Oct 2026 pricing) - verify before relying on it",
    ),
)


class ApiModelCatalog:
    """The set of API models a company may route to.

    Mirrors `selector.ModelCatalog`'s shape on purpose (register/find/
    list_candidates) so the two halves of the local-vs-API decision read the
    same way, even though one discovers models from disk and this one is
    pure config.
    """

    def __init__(self, specs: Optional[Sequence[ApiModelSpec]] = None, include_defaults: bool = True):
        self._specs: List[ApiModelSpec] = []
        if include_defaults:
            self._specs.extend(DEFAULT_API_CATALOG)
        for spec in specs or ():
            self.register(spec)

    def register(self, spec: ApiModelSpec) -> None:
        """Adds a model, replacing any existing entry with the same
        provider+name (so a user overriding a seed default's price gets the
        override, not two competing rows)."""
        self._specs = [s for s in self._specs if not (s.name == spec.name and s.provider == spec.provider)]
        self._specs.append(spec)

    def unregister(self, name: str, provider: Optional[str] = None) -> None:
        self._specs = [
            s for s in self._specs
            if not (s.name == name and (provider is None or s.provider == provider))
        ]

    def all(self) -> List[ApiModelSpec]:
        return list(self._specs)

    def find(self, name: str, provider: Optional[str] = None) -> Optional[ApiModelSpec]:
        for spec in self._specs:
            if spec.name == name and (provider is None or spec.provider == provider):
                return spec
        return None

    def matching(self, specialty: Optional[str] = None) -> List[ApiModelSpec]:
        """Specs carrying `specialty`. Returns [] if none match - the caller
        decides whether to widen the search (ModelPolicy does, with a note in
        the decision's reason, rather than failing outright)."""
        if not specialty:
            return self.all()
        return [s for s in self._specs if specialty in s.specialties]

    def __len__(self) -> int:
        return len(self._specs)


# ---------------------------------------------------------------------------
# How a local pick is turned into Agent(provider=..., model=..., base_url=...)
# ---------------------------------------------------------------------------

# selector.py discovers models under provider names that describe *where the
# weights live* ("ollama", "lm-studio", "hf"), which is not the same thing as
# the wire protocol Agent.change_api needs. This table bridges the two.
#
# LM Studio and vLLM both expose OpenAI-compatible servers on well-known local
# ports, so they map to provider="openai" plus an explicit base_url. These are
# the stock defaults - override via ModelPolicy(local_bindings=...) if you run
# them on other ports. A bare HF checkout with no server in front of it has no
# endpoint at all, which is why "hf" points at the vLLM default and is flagged
# in the decision's reason rather than pretending it will just work.
DEFAULT_LOCAL_BINDINGS: Dict[str, Dict[str, Optional[str]]] = {
    "ollama": {"provider": "ollama", "base_url": None},  # Agent already knows Ollama's URL
    "lm-studio": {"provider": "openai", "base_url": "http://localhost:1234/v1/chat/completions"},
    "hf": {"provider": "openai", "base_url": "http://localhost:8000/v1/chat/completions"},
    "vllm": {"provider": "openai", "base_url": "http://localhost:8000/v1/chat/completions"},
}


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@dataclass
class PolicyDecision:
    """What `ModelPolicy.decide()` concluded, and why.

    `kind` is one of:
      "local"       - run this employee on a local model (provider/model/base_url set)
      "api"         - buy an API model (provider/model set, api_spec carries the economics)
      "unavailable" - nothing satisfied the constraints; `reason` and
                      `install_hint` explain what to do about it. Returned
                      rather than raised so a caller can log it, escalate it,
                      or fall back on its own terms - the same philosophy as
                      SelectionResult.needs_install in selector.py.

    Every decision carries the full `reason` string and the raw
    `local_selection` it was based on, so `Company.activity()` can show *why*
    an employee ended up on the model it did months later.
    """

    kind: str
    reason: str
    effort: str
    rank: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    specialty: Optional[str] = None
    api_spec: Optional[ApiModelSpec] = None
    local_candidate: Optional[LocalModelCandidate] = None
    local_selection: Optional[SelectionResult] = None
    install_hint: Optional[str] = None
    estimated_cost_per_1k: Optional[float] = None
    considered_local: bool = False

    @property
    def ok(self) -> bool:
        return self.kind in ("local", "api")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "effort": self.effort,
            "rank": self.rank,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "specialty": self.specialty,
            "estimated_cost_per_1k": self.estimated_cost_per_1k,
            "considered_local": self.considered_local,
            "install_hint": self.install_hint,
            "api_spec": self.api_spec.to_dict() if self.api_spec else None,
        }


class ModelPolicy:
    """Decides local vs API for one employee (or one task).

    Construction is cheap and does no I/O. The expensive part - a real
    hardware benchmark - is only touched when a decision actually needs to
    consider local models, and is cached on the instance after the first
    call, so a company hiring twenty employees benchmarks once.

    The decision, in one place:

      mode="api"    - never look at local; pick from the API catalog.
      mode="local"  - never look at the API catalog; pick the best local fit.
                      If nothing fits, kind="unavailable" with an install
                      hint (an explicit local request should not silently
                      start spending money - see allow_api_fallback).
      mode="auto"   - the effort hint moves the threshold:
                        cheap    : any feasible local model wins (it's free);
                                   API only if nothing local fits.
                        balanced : local wins only if it fits *entirely in
                                   VRAM* ("gpu_resident"); a heavily offloaded
                                   model is slow enough that a cheap API call
                                   is the better trade. Otherwise API.
                        effort   : API, taking the strongest model that meets
                                   the specialty; local only if the API
                                   catalog is empty.

    Within the API catalog, the same effort hint picks the ranking rule:
      cheap    -> lowest blended cost
      balanced -> best capability per dollar
      effort   -> highest capability (cheaper wins ties)
    """

    def __init__(
        self,
        api_catalog: Optional[ApiModelCatalog] = None,
        local_catalog: Optional[ModelCatalog] = None,
        benchmark: Optional[BenchmarkResult] = None,
        quota: Optional[ResourceQuota] = None,
        rank_effort: Optional[Dict[str, str]] = None,
        local_bindings: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
        output_ratio: float = 0.75,
        min_gpu_resident_fraction: float = 0.5,
        allow_local: bool = True,
    ):
        """
        api_catalog: models this company may buy. Defaults to
            ApiModelCatalog() (the seed table - see its warning).
        local_catalog: passed straight through to
            ModelRouter.allocate_local_auto(); defaults to a fresh
            ModelCatalog() that auto-discovers Ollama/LM Studio/HF.
        benchmark: a BenchmarkResult. If omitted, one is measured lazily the
            first time a decision actually needs to consider local models
            (HardwareBenchmark caches to disk, so this is cheap after the
            first ever run on a machine). Pass one in from a test or to
            avoid the measurement entirely.
        quota: optional ResourceQuota capping RAM/VRAM the local selector may
            assume.
        rank_effort: override DEFAULT_RANK_EFFORT.
        local_bindings: override DEFAULT_LOCAL_BINDINGS (e.g. LM Studio on a
            non-standard port).
        output_ratio: the input/output mix assumed by
            ApiModelSpec.blended_cost_per_1k.
        allow_local: set False on a machine that should never run models
            locally (a CI box, a cheap VPS) - every decision then goes to the
            API catalog without ever benchmarking.
        """
        self.api_catalog = api_catalog if api_catalog is not None else ApiModelCatalog()
        self.local_catalog = local_catalog
        self.quota = quota
        self.rank_effort = dict(rank_effort or DEFAULT_RANK_EFFORT)
        self.local_bindings = dict(local_bindings or DEFAULT_LOCAL_BINDINGS)
        self.output_ratio = output_ratio
        self.min_gpu_resident_fraction = min_gpu_resident_fraction
        self.allow_local = allow_local
        self._benchmark = benchmark

    # -- benchmark plumbing -------------------------------------------------

    def benchmark(self) -> Optional[BenchmarkResult]:
        """The BenchmarkResult local decisions are scored against, measured
        (and cached on this instance) on first use. Returns None if the
        machine can't be benchmarked at all, which downgrades every local
        option rather than raising."""
        if self._benchmark is None:
            try:
                from .benchmark import HardwareBenchmark

                self._benchmark = HardwareBenchmark.run()
            except Exception:
                return None
        return self._benchmark

    # -- effort ------------------------------------------------------------

    def resolve_effort(self, effort: Optional[str], rank: Optional[str]) -> str:
        """The effort level to actually use, from an explicit hint or the rank
        default. Never raises - see `_unrecognized_effort` for why, and for
        what happens instead."""
        resolved, _ = self.resolve_effort_explained(effort, rank)
        return resolved

    def resolve_effort_explained(
        self, effort: Optional[str], rank: Optional[str]
    ) -> Tuple[str, str]:
        """`resolve_effort`, plus a note when the hint wasn't understood.

        Split out so `decide()` can carry that note into the PolicyDecision's
        `reason`. The note is "" whenever the hint was recognised or absent, so
        the common path adds nothing.
        """
        note = ""
        if effort is not None:
            key = str(effort).strip().lower()
            if key in _EFFORT_ALIASES:
                return _EFFORT_ALIASES[key], note
            note = self._unrecognized_effort(key, rank)
        if rank is not None and rank in self.rank_effort:
            return self.rank_effort[rank], note
        return EFFORT_BALANCED, note

    def _unrecognized_effort(self, key: str, rank: Optional[str]) -> str:
        """Warn about an effort hint that didn't match, and suggest the nearest
        real one.

        Phase 4 decided an unrecognized hint falls back to the rank default
        rather than raising, because this value arrives from a model filling in
        a tool schema and a typo must not take a running company down. That
        decision stands. What didn't work is that the fallback was *silent*:
        `resolve_effort("balnced", "JUNIOR")` quietly returned "cheap", the
        lowest tier, so a one-character typo downgraded an employee's model and
        nothing anywhere said so.

        A misspelled *skill* has always raised a KeyError listing the valid
        names (PresetRegistry.get). This closes that asymmetry from the other
        end - still no exception, but a warning, a suggestion, and a line in
        the decision's reason, so "why is this employee on this model?" stays
        answerable later, which is the whole point of storing the decision.
        """
        suggestion = did_you_mean(key, tuple(EFFORT_LEVELS) + tuple(_EFFORT_ALIASES))
        fallback = self.rank_effort.get(rank or "", EFFORT_BALANCED)
        note = (f"effort hint {key!r} was not recognized, so the "
                f"{rank or 'library'} default ({fallback}) was used instead{suggestion}")
        logger.warning("ModelPolicy: %s", note)
        return note

    # -- API side ----------------------------------------------------------

    def rank_api_models(self, effort: str, specialty: Optional[str] = None) -> List[ApiModelSpec]:
        """API specs ordered best-first for this effort level. Widens past the
        specialty filter if nothing carries the tag (the caller records that
        widening in the decision's reason)."""
        pool = self.api_catalog.matching(specialty)
        if not pool:
            pool = self.api_catalog.all()
        if not pool:
            return []

        if effort == EFFORT_CHEAP:
            return sorted(pool, key=lambda s: (s.blended_cost_per_1k(self.output_ratio), -s.capability, s.name))
        if effort == EFFORT_EFFORT:
            return sorted(pool, key=lambda s: (-s.capability, s.blended_cost_per_1k(self.output_ratio), s.name))

        # balanced: capability per dollar. A free/zero-cost entry (someone
        # listing a self-hosted endpoint in the API catalog) would divide by
        # zero, so it is floored - and treated as maximally efficient, which
        # is the right answer for a model that costs nothing to call.
        def value(spec: ApiModelSpec) -> Tuple[float, float, str]:
            cost = spec.blended_cost_per_1k(self.output_ratio)
            efficiency = spec.capability / cost if cost > 0 else float("inf")
            return (-efficiency, -spec.capability, spec.name)

        return sorted(pool, key=value)

    def _api_decision(
        self, effort: str, rank: Optional[str], specialty: Optional[str],
        prefix: str = "", considered_local: bool = False,
    ) -> PolicyDecision:
        widened = bool(specialty) and not self.api_catalog.matching(specialty)
        ranked = self.rank_api_models(effort, specialty)
        if not ranked:
            return PolicyDecision(
                kind="unavailable", effort=effort, rank=rank, specialty=specialty,
                considered_local=considered_local,
                reason=(prefix + "no API models are configured in this policy's catalog "
                        "(and no local model was usable), so there is nothing to route to."),
            )
        best = ranked[0]
        why = [prefix] if prefix else []
        why.append(
            f"chose API model '{best.name}' ({best.provider}) for effort={effort}"
            + (f", specialty={specialty!r}" if specialty else "")
            + f": capability {best.capability:.2f} at ~${best.blended_cost_per_1k(self.output_ratio):.5f}/1k tokens"
        )
        if widened:
            why.append(
                f"no catalog model was tagged {specialty!r}, so the whole catalog was considered"
            )
        return PolicyDecision(
            kind="api", effort=effort, rank=rank, specialty=specialty,
            provider=best.provider, model=best.name,
            base_url=best.base_url, api_key=best.api_key,
            api_spec=best, considered_local=considered_local,
            estimated_cost_per_1k=best.blended_cost_per_1k(self.output_ratio),
            reason="; ".join(p for p in why if p),
        )

    # -- local side --------------------------------------------------------

    def _local_selection(
        self, requested_model: Optional[str] = None, requested_provider: Optional[str] = None
    ) -> Optional[SelectionResult]:
        bench = self.benchmark()
        if bench is None:
            return None
        return ModelRouter.allocate_local_auto(
            benchmark=bench,
            catalog=self.local_catalog,
            requested_name=requested_model,
            requested_provider=requested_provider,
            quota=self.quota,
            min_gpu_resident_fraction=self.min_gpu_resident_fraction,
        )

    def _bind_local(self, candidate: LocalModelCandidate) -> Tuple[str, Optional[str], str]:
        """(agent_provider, base_url, note) for a chosen local candidate."""
        binding = self.local_bindings.get(candidate.provider)
        if binding is None:
            return ("ollama", None, f"no binding configured for local provider "
                                    f"{candidate.provider!r}; defaulted to the Ollama transport")
        note = ""
        if candidate.provider in ("hf", "vllm"):
            note = ("assumes an OpenAI-compatible server (e.g. vLLM) is already serving this "
                    "checkout at the bound base_url - raw HF weights on disk are not an endpoint")
        return (binding["provider"] or "ollama", binding.get("base_url"), note)

    @staticmethod
    def _tier_of(selection: SelectionResult) -> Optional[str]:
        if selection.chosen is None:
            return None
        for scored in selection.ranked:
            if scored.candidate is selection.chosen or (
                scored.candidate.name == selection.chosen.name
                and scored.candidate.provider == selection.chosen.provider
            ):
                return scored.tier
        return None

    def _local_decision(
        self, selection: SelectionResult, effort: str, rank: Optional[str], specialty: Optional[str],
        prefix: str = "",
    ) -> PolicyDecision:
        candidate = selection.chosen
        provider, base_url, note = self._bind_local(candidate)
        tier = self._tier_of(selection) or "unknown"
        why = [p for p in (prefix,) if p]
        why.append(
            f"chose local model '{candidate.name}' ({candidate.provider}, tier={tier}) "
            f"for effort={effort}: {selection.reason}"
        )
        if note:
            why.append(note)
        return PolicyDecision(
            kind="local", effort=effort, rank=rank, specialty=specialty,
            provider=provider, model=candidate.name, base_url=base_url,
            local_candidate=candidate, local_selection=selection,
            considered_local=True, estimated_cost_per_1k=0.0,
            reason="; ".join(why),
        )

    # -- the entry point ---------------------------------------------------
    def decide(
        self,
        rank: Optional[str] = None,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
        mode: str = "auto",
        requested_provider: Optional[str] = None,
        requested_model: Optional[str] = None,
        allow_api_fallback: Optional[bool] = None,
    ) -> PolicyDecision:
        """Pick a model for one employee/task. Never raises for a routing
        failure - returns kind="unavailable" with a reason instead.

        Thin wrapper over `_decide`: it resolves the effort hint once (so an
        unrecognized one is warned about once, not once per internal lookup)
        and appends that note to whatever reason the decision came back with.
        Doing it here rather than at each of `_decide`'s eight return paths
        means a new return path cannot forget to carry it.

        requested_model/requested_provider pin a specific model. In
        mode="local" that pins the *local* selection (and reports an install
        hint if it isn't installed); otherwise a pinned model found in the API
        catalog is returned directly. A pin that matches nothing anywhere
        falls through to the normal decision with a note, rather than failing
        - a stale pin shouldn't take a company down at hire time.

        allow_api_fallback defaults to False for mode="local" (an explicit
        local request should not quietly start spending money) and True
        everywhere else.
        """
        eff, effort_note = self.resolve_effort_explained(effort, rank)
        decision = self._decide(
            eff=eff, rank=rank, specialty=specialty, mode=mode,
            requested_provider=requested_provider, requested_model=requested_model,
            allow_api_fallback=allow_api_fallback,
        )
        if effort_note:
            decision.reason = f"{decision.reason}; {effort_note}" if decision.reason else effort_note
        return decision

    def _decide(
        self,
        eff: str,
        rank: Optional[str] = None,
        specialty: Optional[str] = None,
        mode: str = "auto",
        requested_provider: Optional[str] = None,
        requested_model: Optional[str] = None,
        allow_api_fallback: Optional[bool] = None,
    ) -> PolicyDecision:
        """The decision itself, on an already-resolved effort level."""
        mode = (mode or "auto").strip().lower()
        if allow_api_fallback is None:
            allow_api_fallback = (mode != "local")

        # An explicit pin that names something in the API catalog wins outright
        # (unless the caller specifically asked for local-only).
        if requested_model and mode != "local":
            pinned = self.api_catalog.find(requested_model, requested_provider)
            if pinned is not None:
                return PolicyDecision(
                    kind="api", effort=eff, rank=rank, specialty=specialty,
                    provider=pinned.provider, model=pinned.name,
                    base_url=pinned.base_url, api_key=pinned.api_key, api_spec=pinned,
                    estimated_cost_per_1k=pinned.blended_cost_per_1k(self.output_ratio),
                    reason=f"'{requested_model}' was requested explicitly and is in the API catalog",
                )

        if mode == "api" or not self.allow_local:
            prefix = "" if mode == "api" else "this policy has allow_local=False, so local was skipped"
            return self._api_decision(eff, rank, specialty, prefix=prefix)

        if mode == "local":
            selection = self._local_selection(requested_model, requested_provider)
            if selection is None:
                reason = "could not benchmark this machine, so no local model could be scored"
                if allow_api_fallback:
                    return self._api_decision(eff, rank, specialty, prefix=reason, considered_local=True)
                return PolicyDecision(kind="unavailable", effort=eff, rank=rank, specialty=specialty,
                                      considered_local=True, reason=reason)
            if selection.chosen is not None:
                return self._local_decision(selection, eff, rank, specialty)
            if allow_api_fallback:
                return self._api_decision(eff, rank, specialty,
                                          prefix=f"local was requested but unusable ({selection.reason})",
                                          considered_local=True)
            return PolicyDecision(
                kind="unavailable", effort=eff, rank=rank, specialty=specialty,
                considered_local=True, local_selection=selection,
                install_hint=selection.install_hint,
                reason=f"mode='local' was requested but no local model is usable: {selection.reason}",
            )

        if mode != "auto":
            raise ValueError(f"Unknown mode {mode!r} - use 'auto', 'local', or 'api'.")

        # ---- auto: the effort hint moves the local/API threshold ----
        if eff == EFFORT_EFFORT and len(self.api_catalog) > 0:
            return self._api_decision(
                eff, rank, specialty,
                prefix="effort=effort routes to the API catalog before considering local models",
            )

        selection = self._local_selection(requested_model, requested_provider)
        if selection is None:
            return self._api_decision(
                eff, rank, specialty,
                prefix="this machine could not be benchmarked, so local models were not scorable",
                considered_local=True,
            )

        if selection.chosen is not None:
            tier = self._tier_of(selection)
            if eff == EFFORT_CHEAP or eff == EFFORT_EFFORT:
                # cheap: free beats paid, offloaded or not.
                # effort: only reachable here when the API catalog is empty -
                # local is the only option left, so take the best one.
                prefix = ("" if eff == EFFORT_CHEAP
                          else "no API models are configured, so effort=effort fell back to local")
                return self._local_decision(selection, eff, rank, specialty, prefix=prefix)
            if tier == "gpu_resident":
                return self._local_decision(
                    selection, eff, rank, specialty,
                    prefix="effort=balanced accepts local only when it fits entirely in VRAM, and it does",
                )
            if len(self.api_catalog) == 0:
                return self._local_decision(
                    selection, eff, rank, specialty,
                    prefix=("effort=balanced would normally prefer an API model over an offloaded local one, "
                            "but no API models are configured"),
                )
            return self._api_decision(
                eff, rank, specialty, considered_local=True,
                prefix=(f"the best local fit ('{selection.chosen.name}', tier={tier}) is not fully "
                        f"GPU-resident, and effort=balanced trades that latency for a cheap API call"),
            )

        # No local model usable at all.
        hint = f"no local model was usable ({selection.reason})"
        if allow_api_fallback and len(self.api_catalog) > 0:
            return self._api_decision(eff, rank, specialty, prefix=hint, considered_local=True)
        return PolicyDecision(
            kind="unavailable", effort=eff, rank=rank, specialty=specialty,
            considered_local=True, local_selection=selection,
            install_hint=selection.install_hint,
            reason=hint + " and no API model was available to fall back to",
        )

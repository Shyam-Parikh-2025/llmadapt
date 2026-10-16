from typing import Dict, Any, Optional, Tuple
from .hardware import HardwareProfiler, ResourceQuota
from .compressor import CompressionPolicy
from .benchmark import BenchmarkResult
from .selector import LocalModelSelector, ModelCatalog, SelectionResult


class RoleRank:
    C_SUITE = "C_SUITE"
    GENERAL_MANAGER = "GENERAL_MANAGER"
    MANAGER = "MANAGER"
    SENIOR = "SENIOR"
    JUNIOR = "JUNIOR"
    INTERN = "INTERN"
    VOLUNTEER = "VOLUNTEER"

    # Rough seniority order, highest authority first - company.py uses this to
    # find a sensible default entry point (top of the chain) and to decide
    # which direction "up" is when escalating. Not a capability ranking - a
    # VOLUNTEER isn't necessarily weaker than an INTERN, just lowest-priority
    # for delegation and budget purposes (see company.py's budget docs).
    ORDER = [C_SUITE, GENERAL_MANAGER, MANAGER, SENIOR, JUNIOR, INTERN, VOLUNTEER]


# `role` is RoleRank under a second, lowercase name - not a copy, the exact
# same class object - so a rank reads as `role.SENIOR` instead of
# `RoleRank.SENIOR`. This replaces an earlier attempt at this that bound
# C_SUITE/MANAGER/SENIOR/JUNIOR/INTERN/VOLUNTEER as bare top-level names:
# shorter to type, but MANAGER/SENIOR/JUNIOR/INTERN are common enough English
# words that `from llmadapt import *` (or even just picking a local variable
# name) could collide with one. Namespacing under one lowercase name cuts
# that risk from seven common words down to one - the same shape the standard
# library already uses for exactly this (`logging.INFO`, `logging.DEBUG`,
# `socket.AF_INET`, `errno.ENOENT`, `stat.S_IRUSR`): a short, lowercase
# module-like handle in front of the actual constants, instead of either the
# full class name or nothing in front of them at all.
#
# `role` is still a single common word, and this module's own org_templates.py
# uses `role` constantly as a *loop variable* (`for role in roles: ...`) - a
# local `role = ...` inside a function shadows the imported namespace for that
# function only, which is ordinary Python scoping and not a bug, but it means
# `role.SENIOR` won't resolve inside a block that's already reusing the name
# for something else. `RoleRank.SENIOR` remains available, unchanged, as the
# zero-ambiguity spelling for anyone who'd rather not have even that.
#
# `role` is deliberately NOT re-exported from the top-level `llmadapt`
# package - only `RoleRank` is, same as always. Getting `role` requires an
# explicit `from llmadapt.router import role` (or `from llmadapt.company
# import mode, review` for its two siblings below) rather than falling out of
# a bare `from llmadapt import *`. That trades one extra word in the import
# line for confining the collision risk to callers who deliberately opted in,
# rather than handing it to everyone who imports the package at all.
#
# Why this stays a plain class of string constants and not `class
# RoleRank(str, Enum)` (or 3.11+'s `StrEnum`, which pyproject.toml's
# `python_requires>=3.9` rules out outright): a real Enum CAN be made to
# behave exactly like a string everywhere - equality, hashing, dict-key use,
# and JSON serialization all already work correctly with a `str` mixin - but
# `f"{RoleRank.SENIOR}"` and `str(RoleRank.SENIOR)` do NOT come out as
# "SENIOR" by default; Enum's own `__str__`/`__format__` win over the mixin
# and print "RoleRank.SENIOR" instead (confirmed against a live interpreter,
# not assumed). Fixing that means remembering to add
# `__str__ = str.__str__` inside the class body - a one-line fix, but an easy
# one to forget, and this codebase embeds ranks into f-strings constantly
# (escalation messages, system-instruction role lines, error text) where the
# regression would be silent: a subtly wrong string handed to an LLM or
# printed in a log, not an exception. A plain class was already zero-risk on
# every one of those axes with no extra line required, and the typo-detection
# an Enum would add is already covered by `did_you_mean()` at validation time
# - so the extra machinery buys real but marginal safety at a real, easy-to-
# miss cost. Not worth it for a set of values this small and this static.
role = RoleRank


class ModelRouter:
    """Evaluates tasks, checks hardware constraints, and maps LLMs dynamically."""

    @classmethod
    def check_hardware_safety(
        cls, 
        requested_model: str, 
        estimated_ram_gb: float, 
        quota: ResourceQuota
    ) -> Tuple[bool, str]:
        """Validates if a local model binary exceeds host system memory thresholds."""
        specs = HardwareProfiler.inspect()
        max_allowed_ram = (
            quota.max_ram_gb 
            if (quota.mode == "manual" and quota.max_ram_gb is not None) 
            else (specs["system_ram_gb"] * 0.50)
        )

        if estimated_ram_gb > max_allowed_ram:
            msg = (
                f"Model '{requested_model}' (~{estimated_ram_gb}GB RAM) exceeds "
                f"usable threshold of {max_allowed_ram:.2f}GB RAM."
            )
            return False, msg
        return True, ""

    @classmethod
    def allocate_model(
        cls,
        rank: str,
        policy: Dict[str, Any],
        model_map: Dict[str, str],
        task_prompt: Optional[str] = None,
        quota: Optional[ResourceQuota] = None
    ) -> str:
        """Determines the target LLM based on corporate rank, cost/speed policy, and custom mapping."""
        # 1. Check explicit custom rank mappings passed by user
        if rank in model_map:
            selected_model = model_map[rank]
            if quota and "ollama" in selected_model.lower():
                # Estimate 5GB for standard 8B local models as a reference check
                is_safe, warning = cls.check_hardware_safety(selected_model, 5.0, quota)
                if not is_safe and policy.get("auto_fallback", True):
                    fallback = model_map.get("fallback", "gemini-3.6-flash")
                    print(f"[Router Warning]: {warning} Falling back to '{fallback}'.")
                    return fallback
            return selected_model

        # 2. Policy-driven routing heuristics
        cost_priority = policy.get("cost_priority", "medium")

        if cost_priority == "high" and rank in [RoleRank.JUNIOR, RoleRank.INTERN]:
            return model_map.get("local_default", "ollama/llama3.1:8b")
        elif rank in [RoleRank.C_SUITE, RoleRank.MANAGER]:
            return model_map.get("frontier_default", "claude-sonnet-5-5")

        return model_map.get("fast_default", "gemini-3.6-flash")

    @classmethod
    def allocate_local_auto(
        cls,
        benchmark: BenchmarkResult,
        catalog: Optional[ModelCatalog] = None,
        requested_name: Optional[str] = None,
        requested_provider: Optional[str] = None,
        quota: Optional[ResourceQuota] = None,
        min_gpu_resident_fraction: float = 0.5,
    ) -> SelectionResult:
        """'Auto mode' for local models: picks the best-fitting installed local
        model for this machine using real hardware.benchmark numbers, instead
        of check_hardware_safety's flat 5GB guess above. Pass requested_name to
        pin a specific model - if it isn't installed, the result carries
        needs_install/install_hint instead of silently falling back to
        something else. Leave requested_name as None to let it pick the best
        installed candidate outright.

        catalog defaults to a fresh ModelCatalog() (auto-discovers Ollama/LM
        Studio/HF on every call); pass one in to reuse registrations made via
        catalog.register(), e.g. for models the caller wants considered even
        before install.

        See selector.LocalModelSelector for the ranking logic and its current
        limits (fit-tier heuristic, not a tokens/sec predictor yet).
        """
        catalog = catalog or ModelCatalog()
        return LocalModelSelector.select_best(
            benchmark=benchmark,
            catalog=catalog,
            requested_name=requested_name,
            requested_provider=requested_provider,
            quota=quota,
            min_gpu_resident_fraction=min_gpu_resident_fraction,
        )

    @classmethod
    def allocate_compression_policy(cls, rank: str) -> CompressionPolicy:
        """Rank -> CompressionPolicy, same shape as allocate_model but for tool-output
        compression instead of model choice. Whoever builds an Agent for a given rank calls
        this alongside allocate_model to get both.

        Intern/Junior are the local-tier "worker" ranks - tightest context windows, and the
        ones directly executing raw tool calls, so compression is on with a tight budget and
        (deliberately) no summarizer, per the model-thrash reasoning in CompressionPolicy's
        own docstring - a local worker agent shouldn't trigger a second local model load just
        to shrink its own tool output. Senior gets compression on too but with more headroom.
        Manager/C-Suite run on frontier APIs with much bigger context windows and are more
        likely consuming already-delegated/summarized results than raw tool spew, so they're
        left disabled by default - construct a CompressionPolicy by hand and pass it to
        set_compression_policy() if a specific agent needs different behavior than its rank's
        default.
        """
        if rank in (RoleRank.INTERN, RoleRank.JUNIOR):
            return CompressionPolicy(enabled=True, max_chars=1500, min_chars_to_bother=300)
        if rank == RoleRank.SENIOR:
            return CompressionPolicy(enabled=True, max_chars=4000, min_chars_to_bother=1000)
        return CompressionPolicy(enabled=False)  # MANAGER, C_SUITE, and any unrecognized rank
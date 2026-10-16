"""llmadapt - a small, provider-agnostic LLM agent library (Anthropic, OpenAI, Gemini, Ollama).

Start here:

    from llmadapt import Agent
    agent = Agent("anthropic")            # or "openai", "gemini", "ollama"
    print(agent.chat("Hello"))

Run `llmadapt doctor` in a terminal to check what is configured.
"""

__version__ = "0.4.0"

from .core import Agent, Conversation, ToolRegistry, ToolControlFlow, extract_json
from .compressor import ContextCompressor, CompressionPolicy
# `role` deliberately stops here - see router.py's docstring next to its
# definition. Reach for it with `from llmadapt.router import role`, or use
# `RoleRank` (exported below) directly. Same story for `mode`/`review`,
# reachable via `from llmadapt.company import mode, review` but not from here.
from .router import ModelRouter, RoleRank
from .hardware import ResourceQuota, HardwareProfiler, MoELayerOffloader, LocalModelSingleton
from .company import (
    Company, Employee, Team, EscalationEvent, EscalationDecision, EscalationUnresolved,
    default_on_escalation, always_decline, always_approve,
    PolicyMode, ReviewMode, PausedRun, ESCALATION_PENDING, RunPaused,
)
from .budget import BudgetLedger, CostModel
from .policy import ModelPolicy, ApiModelCatalog, ApiModelSpec, PolicyDecision, suggested_importance
from .presets import (
    Preset, PresetRegistry, PresetBundle, default_bundle,
    Skill, Personality, Palette, OrgTemplate, RoleSpec,
    SKILLS, PERSONALITIES, PALETTES, ORG_TEMPLATES,
    compose_system_instruction,
)
from .compaction import LogCompactionPolicy, ALWAYS_KEEP_KINDS
from .builder import (
    set_company_up, set_up_company, build_company, CompanySpec, EmployeeSpec,
    company_setup_schema, company_options, preset_descriptions, register_company_builder,
    make_company_via_gui, quick_company,
)
from .help import company_help
from .delegation import (
    stub_and_fill, plan_then_execute, stub_and_fill_async, plan_then_execute_async,
    extract_stubs, assemble_module,
    StubPlan, StubAndFillResult, PlanResult,
)
from .archive import RunArchive
from .state import CompanyState, StateMismatch, capture_state, apply_state, STATE_VERSION
from .env import load_env, resolve_api_key, key_env_candidates
from .compressor import PIN_KEY, HistoryCompactionPolicy
from .suggest import did_you_mean
from .toolkits import (
    filesystem_tools, code_tools, shell_tools, python_tools, web_tools, toolkit, TOOLKITS, outline_source,
)

__all__ = [
    "__version__",
    "Agent",
    "extract_json",
    "filesystem_tools",
    "code_tools",
    "shell_tools",
    "python_tools",
    "web_tools",
    "toolkit",
    "TOOLKITS",
    "outline_source",
    "Conversation",
    "ToolRegistry",
    "ToolControlFlow",
    "ContextCompressor",
    "CompressionPolicy",
    "HistoryCompactionPolicy",
    "PIN_KEY",
    "RunArchive",
    "CompanyState",
    "StateMismatch",
    "capture_state",
    "apply_state",
    "STATE_VERSION",
    "load_env",
    "resolve_api_key",
    "key_env_candidates",
    "did_you_mean",
    "ModelRouter",
    "RoleRank",
    "ResourceQuota",
    "HardwareProfiler",
    "MoELayerOffloader",
    "LocalModelSingleton",
    "Company",
    "Employee",
    "Team",
    "EscalationEvent",
    "EscalationDecision",
    "EscalationUnresolved",
    "PausedRun",
    "ESCALATION_PENDING",
    "RunPaused",
    "default_on_escalation",
    "always_decline",
    "always_approve",
    "PolicyMode",
    "ReviewMode",
    "BudgetLedger",
    "CostModel",
    "ModelPolicy",
    "ApiModelCatalog",
    "ApiModelSpec",
    "PolicyDecision",
    "suggested_importance",
    "Preset",
    "PresetRegistry",
    "PresetBundle",
    "default_bundle",
    "Skill",
    "Personality",
    "Palette",
    "OrgTemplate",
    "RoleSpec",
    "SKILLS",
    "PERSONALITIES",
    "PALETTES",
    "ORG_TEMPLATES",
    "compose_system_instruction",
    "stub_and_fill",
    "plan_then_execute",
    "stub_and_fill_async",
    "plan_then_execute_async",
    "extract_stubs",
    "assemble_module",
    "StubPlan",
    "StubAndFillResult",
    "PlanResult",
    "LogCompactionPolicy",
    "ALWAYS_KEEP_KINDS",
    "set_company_up",
    "set_up_company",
    "build_company",
    "make_company_via_gui",
    "quick_company",
    "CompanySpec",
    "EmployeeSpec",
    "company_setup_schema",
    "company_options",
    "preset_descriptions",
    "company_help",
    "register_company_builder",
]

"""A ready-made multi-agent team: a lead delegates to reports, a reviewer
signs off. With no model_map everything runs on local Ollama, so this cannot
spend money.

    python examples/04_company.py
"""
from llmadapt import quick_company

company = quick_company("small-coding-team", total_token_budget=200_000)
print(company.render_org_chart(fmt="ascii"))
print(company.run("Write a function that parses 'key=value; key2=value2' into a dict, with tests."))
import json
print(json.dumps(company.budget_report(), indent=2))

"""Get JSON back instead of prose - parsed, with one automatic retry.

    python examples/03_structured_output.py
"""
from llmadapt import Agent

agent = Agent("ollama", verbose=False)
issues = agent.chat_json(
    "Find the bugs in this function:\n\n"
    "def average(xs):\n    return sum(xs) / len(xs) - 1\n",
    schema_hint=[{"line": "int", "problem": "str", "fix": "str"}],
)
for issue in issues:
    print(f"line {issue['line']}: {issue['problem']} -> {issue['fix']}")

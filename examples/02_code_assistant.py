"""A small code assistant: built-in skills + ready-made read-only tools.

The model can list, search, outline and read files under the folder you
give it - nothing outside it, and nothing is written.

    python examples/02_code_assistant.py path/to/project "Where is the config loaded?"
"""
import sys

from llmadapt import Agent, toolkit

root = sys.argv[1] if len(sys.argv) > 1 else "."
question = sys.argv[2] if len(sys.argv) > 2 else "Give me a short tour of this project's structure."

agent = Agent(
    "ollama",                                   # swap for "anthropic", "openai", "gemini"
    skills=["code-explanation", "agentic-coding"],
    tools=toolkit("fs", "code", root=root),     # read_file, list_files, search_files, outline_file, stub_file
    max_tool_iterations=15,
)
print(agent.chat(question))

"""Smallest useful program: one question, one answer.

    python examples/01_quickstart.py                 # auto: Ollama if running
    python examples/01_quickstart.py anthropic       # or openai / gemini / ollama

API keys come from ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY, in your
environment or a .env file. Run `llmadapt doctor` to see what is configured.
"""
import sys

from llmadapt import Agent

provider = sys.argv[1] if len(sys.argv) > 1 else "ollama"
agent = Agent(provider, system_instruction="You are concise.", verbose=False)

print(agent.chat("In one sentence: what is a race condition?"))
print(agent.chat("Give a two-line Python example of one."))  # the conversation carries over
print(f"\n[{agent.provider} / {agent.model}, {agent.total_tokens_used} tokens]")

# What changed in llmadapt, in plain English

No jargon. Each phase gets: the problem, the fix, and why it matters.

---

## Phase A — the bug where "no" didn't mean no

**The problem.** Say a junior employee gets stuck. It asks its manager for help,
the manager passes it up, and eventually your `on_escalation` handler asks
*you*: "can this employee have more budget?" You say **no**.

What happened next was wrong. Because delegation is built as an ordinary tool
call, your "no" travelled back as if it were a *tool that crashed*. The manager's
model read it as "that didn't work, let me try something else", carried on, and
produced a confident final answer. `company.run()` returned a normal string. No
error. Nothing stopped.

The only case where "no" actually stopped anything was when the stuck employee
was the top one — the shallow case. Delegate the work one level down and your
refusal was silently ignored.

**The fix.** There is now a category of exception that means "this is about the
*run*, not about one tool" — and the tool machinery is forbidden from converting
those into text for the model. A declined escalation is in that category, so it
now travels all the way back to you.

**Why it matters.** The entire safety story of this library is "a human stays in
charge of spending". That story had a hole in it. It doesn't now.

I also merged four near-identical copies of the tool-running code into one. Not
cosmetic: both of the big features below needed to change that exact code, and
changing it in four places twice over is how four copies start disagreeing.

---

## Phase B — one engine instead of two

**The problem.** There were two versions of "talk to the model and run whatever
tools it asks for": a normal one and an async one. They were about 90% identical.
Two copies of the same logic drift apart — someone fixes a bug in one and not the
other, and now the library behaves differently depending on which door you came
in through.

**The fix.** The async version is now the *real* one. The normal version is a
three-line wrapper that starts the async one and waits for it.

Nothing changes for you if you don't care about async. `company.run("do the
thing")` still takes a string and gives back a string.

**Bonus: tools can now be `async def`.** If you write a tool as async, it's
awaited properly. If you write a normal one, it's run in a way that doesn't
freeze everything else. You don't have to know which; both just work.

---

## Phase C — three jobs at once instead of one after another

**The problem.** A manager delegating three independent subtasks did them
strictly in order. Three tasks that each take 20 seconds took 60 seconds, and
while report #1 was working, reports #2 and #3 sat doing nothing. For a library
whose whole premise is a delegation hierarchy, that's the single biggest waste
there was.

**The fix.** They now run at the same time. Three 20-second tasks take about 20
seconds. The test asserts both the clock *and* that all three were genuinely
in flight at the same moment.

**The three things that had to be got right, because "run it all at once" breaks
things quietly:**

1. **One person, one job at a time.** A manager can name the same report twice
   in a single turn. That employee is a single conversation — two tasks writing
   into it simultaneously would produce a transcript where neither task's
   messages follow each other. Each employee now has a lock: the second job
   waits. Which is also just what the org-chart metaphor says should happen.

2. **The budget check.** Checking "is there budget left?" and then acting on the
   answer is two steps, and two employees checking at the same instant could both
   get "yes" for the same last slice. That check is now serialized.

   The subtle part: I hold that lock across the *check only*, never across asking
   you for approval. If it were held longer, this happens — you approve, the
   employee retries, the retry delegates, the delegation hits the budget check,
   and it waits for a lock that its own call is already holding. That's a
   deadlock: no error, no timeout, the company just stops forever. There's a
   test with a watchdog specifically for that shape.

3. **A slow human doesn't freeze everybody.** If your `on_escalation` sits
   waiting on an `input()` prompt, that used to be capable of stalling every
   other employee. It now runs out of the way, so the rest of the company keeps
   working while you decide. Your handler can also be `async def` if you prefer.

---

## Phase D — save where you are

**The problem.** If the process died, everything was gone. There was a log of
*what happened*, but nothing that said *where things stand*.

**The fix.** `company.save_state("run.json")` writes down every conversation,
the budget ledger, the counters, the logs — as plain readable JSON.

**The one thing to understand:** a saved file **cannot rebuild the company by
itself**. A company holds actual Python code — your tools, your escalation
handler — and no data file can carry code. So resuming is two steps:

```python
company = build_my_company()      # the same way you built it originally
company.restore_state("run.json") # then pour the saved state back in
```

If the company you rebuilt doesn't match the one that was saved — someone's
missing, a reporting line moved, a tool is gone — it refuses and tells you every
difference. It won't restore the bits that happen to line up, because that gives
you something that *looks* resumed and isn't.

**No API keys, ever.** The file has provider and model names but no credentials,
and that's actively checked when the file is produced rather than just intended.
Safe to commit, safe to email, safe to move between machines.

---

## Phase E — "hold on, let me think about it"

**The problem.** Your escalation handler had two answers: yes or no. Real
approvals aren't like that. Sometimes the honest answer is "I'll get back to you
in an hour" — and that used to be impossible, because the whole run was sitting
inside a live Python call waiting for your function to return. Close the laptop
and it was gone.

**The fix.** A third answer: `ESCALATION_PENDING`. The run stops cleanly and
hands you back everything needed to pick it up later — hours later, in a
different process, on a different machine.

```python
result = company.run_resumable("the job")
if isinstance(result, PausedRun):
    company.save_state(path="paused.json", paused=result)
    # ... go home, sleep, come back ...
    answer = company.resume(result, EscalationDecision(approve=True,
                                                       extra_token_budget=50_000))
```

**How it works, since this sounds impossible.** You can't freeze a running Python
program and write it to disk. But you don't need to. At each level of the
delegation chain, only two things matter: the conversation so far, and *which of
this turn's tool calls already finished*. Both are ordinary data. So each agent
in the chain writes down the turn it was in the middle of, and resuming replays
that instead of redoing it.

**The practical consequence:** work that already finished is not done twice. If a
manager delegated three ways and two reports answered before the third needed
you, resuming replays those two answers. It does not re-run them — they already
had whatever effects they had, sent whatever emails they sent, spent whatever
budget they spent. This holds at any depth.

Declining at resume time raises an error, exactly as declining immediately would
have. Waiting and then saying no is still saying no.

---

# Your event-loop question — the important one

You've got the shape right but one piece inverted. Let me build it up.

## What an event loop actually is

One worker, juggling many jobs. It works on job A until job A says "I'm waiting
for something" — that's what `await` means — and at that exact moment the worker
puts A down and picks up job B. When A's thing arrives, it goes back to A.

**The worker can only switch jobs at an `await`.** That's the whole rule.
Everything else follows from it.

## What goes wrong in a FastAPI handler

```python
@app.post("/work")
async def handler():
    result = company.run(task)     # <-- a normal function call
    return result
```

`company.run()` is a normal function. Once Python steps into it, there is no
`await` — so the worker cannot put this job down. It's stuck inside `run()` until
it returns.

Meanwhile: every other request to your API is queued behind it. Not crashed, not
lost — just waiting, possibly for minutes.

## Now — your question, precisely

> if a fastapi type thing running an asyncio action in a loop, then a pause in
> this would be because it waits for this to finish?

Yes, exactly right. Your app pauses **because it's waiting for `run()` to
return**, and it cannot do anything else meanwhile.

But here's the piece that's inverted:

> creating another thread and asyncio will not let everything happen at once

The helper thread isn't there to make things happen at once. **It's there to stop
a crash.**

Python has a hard rule: you cannot start an event loop inside a running event
loop. Our `run()` needs a loop. Your FastAPI handler already has one going. So
without intervention, Python raises `RuntimeError: asyncio.run() cannot be called
from a running event loop` and your request 500s.

The helper thread sidesteps that: a brand-new thread has no loop of its own, so
it may legally start one. The work runs, correctly and completely.

**But your loop is still blocked** — not because of anything the thread does, but
because your handler never reached an `await`. A function being called cannot
hand control back to your loop on your behalf. Only you can, at your call site:

```python
@app.post("/work")
async def handler():
    result = await company.run_async(task)   # now the loop can serve others
    return result
```

That's why the warning says what it says: *the result is correct, your loop was
blocked anyway, and here's the actual fix.*

## And this bit is worth being clear about

Inside the work itself, **everything still happens at once**. The three
delegations still run in parallel — they're on the loop running inside that
helper thread, and that loop is perfectly healthy.

So the helper thread costs you nothing internally. What you lose is *your app's*
ability to serve other requests during the call. That's it.

## Thread limits — three different ones

| Threads for what | How many | Set by |
|---|---|---|
| Rescuing a sync call made inside a loop | **One per call.** Not per tool. Created, used, finished. | Only happens when you make that mistake |
| Running ordinary (non-async) tools | Shared pool, `min(32, cpu_count + 4)` | Python's default |
| Tool calls started at once | **8**, new default | `agent.set_max_parallel_tools(n)` |

**Should the first one be capped?** I decided no, deliberately. If a FastAPI app
makes that mistake on every request, you get one thread per in-flight request —
which is bad. But capping it would mean requests queueing on a hidden limit
inside a library they don't know is there, which is a worse and much more
confusing failure. The right answer is to not make the call that way, which is
why the warning is loud and names the fix.

**The third one is new, and it's the one you were reaching for.** A model can ask
for twenty tools in one turn. With delegation as a tool, that's twenty employees
starting simultaneously — twenty API requests, twenty budgets being spent, all
before anything comes back. It's now capped at 8 by default.

I chose a number rather than "unlimited" because unlimited wasn't actually a
policy — it silently inherited Python's 32-thread pool limit, so the real ceiling
depended on the machine you ran it on. A number is a decision. That was an
accident.

```python
agent.set_max_parallel_tools(16)   # I/O-bound, provider can take it
agent.set_max_parallel_tools(1)    # strictly one at a time
```

---

# The three judgement calls I made while you were asleep

### 1. One delegate failing doesn't kill the others

Manager delegates three ways; one report crashes. The other two finish, their
answers go to the model, the run continues. The crash is reported to the model as
a failed tool, which it can work around — same as it always did when these ran
one at a time.

The exception is a declined escalation, which stops everything. That's a
statement about the run, not about one tool.

**Why:** if one failure cancelled its siblings, you'd pay for work that gets
thrown away. And the failure modes are genuinely different — "this tool broke" is
routine, "a human said stop" is not.

### 2. Saved files never contain API keys

A snapshot has provider names and model names, never credentials. On the far
side, keys are re-read from the environment exactly as they were originally.

This is *checked* when the file is written, not just intended — a future field
can't quietly smuggle a key into a file people were told was safe to share.

**Why:** the entire value of a state file is that you can commit it, attach it to
a bug report, or move it between machines. A file that might contain a key can't
do any of those.

### 3. Two people needed at once = you get asked twice

Two delegates hit a wall simultaneously and both need approval. Only one pause
can be handed back at a time. So one is reported to you; the other is *not*
recorded as finished. When you resume, it runs again and asks again.

**Why:** the alternative is treating an unanswered question as answered. One
extra question beats one silently-assumed approval, in the one mechanism whose
entire job is to stop and ask.

---

# Two smaller things you asked for

**Every skill now explains itself.** The catalog had names like
`prompt-engineering` and `discovery-pod` with no description anywhere you'd see
them. A checkbox list of bare names is technically valid input that nobody can
choose from confidently. All 89 presets now carry a description, the GUI shows
them on hover (and under the personality dropdown), and they're in the options
payload a model receives.

**One call that explains everything.** The knowledge needed to build a company
was spread across a tool schema, an options list, four registries and several
docstrings.

```python
from llmadapt import company_help

print(company_help(as_text=True))   # 226 lines of markdown, for you
context = company_help()            # a dict, for handing to a model
```

Also `set_company_up(mode="help")`, and a model calling the setup tool can use
`set_up_company(help_only=True)` before committing to any arguments. It covers
the concepts, every valid name *with what it does*, the safety defaults written
out rather than implied ("0 means always ask a human", "no config means local
models only"), how to run work afterwards, the things that catch people out, and
four worked examples. It builds nothing and spends nothing.

---

**Tests: 496 passing across 23 files, up from 433 across 17.**

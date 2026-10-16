# Getting this onto GitHub

## The short version

Your old `main` is **already** saved as the `v0.2.0` branch on GitHub — you
created it before this run, and it points at exactly the old tip
(`d8add87`). So nothing needs protecting and nothing gets overwritten.

The new work sits **on top of** that old commit, not beside it. In git terms
this is a fast-forward: pushing adds 11 new commits after `d8add87` and leaves
every old commit exactly where it is. **No force push. No rewritten history.**

```
d8add87  ← old main, and where the v0.2.0 branch stays pointing
   │
   ├── dce01c4  Sync Phases 0-3 from the working machine
   ├── 2c0b2b3  Phase 4: model policy
   ├── eb36b4b  Phase 5: preset registry
   ├── b17ae79  Phase 6: delegation
   ├── 6c18942  Phase 7: log compaction
   ├── ddab531  Phase 8: text-mode builder
   ├── fd5cd9a  Phase 8: GUI editor
   └── ... 4 more  ← new main
```

## Why you have to do it and not the overnight session

The cloud sandbox routes git through a proxy that refuses to attach credentials
for any repo outside its authorized list:

```
remote: access denied by the git proxy: Shyam-Parikh-2025/llmadapt is not in
this session's authorized repository set
```

It fails with 403 *before* your token is ever used. Nothing was wrong with the
token — no credential trick gets around it. Reading (clone) worked; writing
did not.

## Do this

`llmadapt-overnight.bundle` is a complete copy of the repository in one file —
all 25 commits and the `v0.3.0` tag.

```bash
cd "C:\Users\HP\OneDrive\Coding Projects\Basics Genai"
git clone llmadapt\llmadapt-overnight.bundle llmadapt-pushed
cd llmadapt-pushed
git remote set-url origin https://github.com/Shyam-Parikh-2025/llmadapt.git

# look before you leap - this lists exactly what would be sent
git fetch origin
git log --oneline origin/main..main

git push origin main
git push origin v0.3.0        # the release tag
```

After that, GitHub shows the new work on `main`, and `v0.2.0` still shows the
old code. Anyone can compare them at:
`https://github.com/Shyam-Parikh-2025/llmadapt/compare/v0.2.0...main`

## If you'd rather push from the folder you already work in

Only if that folder is a git repo already. Check with `git status`. If it is:

```bash
cd "C:\Users\HP\OneDrive\Coding Projects\Basics Genai\llmadapt"
git add -A
git commit -m "Phases 4-8: model policy, presets, delegation, compaction, company builder"
git push origin main
```

You lose the per-phase commit messages this way — everything becomes one
commit. `company-roadmap.md` still has the full reasoning, but the bundle route
above keeps the history properly, so prefer it.

## Two housekeeping notes

**Revoke the token.** GitHub → Settings → Developer settings → Personal access
tokens. It never worked (the proxy blocked it) and was never written into any
file here — but it did appear in a chat transcript, so retire it.

**The commits are authored as you** (`Shyam Parikh
<shyamsparikh11@gmail.com>`), because that's what you asked for at the start.
Each one carries a `Co-Authored-By: Claude Opus 5` trailer, which is the
standard way AI assistance is recorded. GitHub will mark them "Unverified"
simply because they aren't GPG-signed — that's true of every unsigned commit
and is unrelated to this run.

## Check it before you push

```powershell
cd "C:\Users\HP\OneDrive\Coding Projects\Basics Genai\llmadapt"
$env:PYTHONPATH="src"; python tests\run_all.py
```

Expected: `SUITE PASSED: 14 files, 275 checks`.

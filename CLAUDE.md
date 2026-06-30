# CLAUDE.md

Project context for Claude Code (and other AI assistants) working in this repo.

## What this project is

`LLM-Analyse-Persian-Emotion` is a **reproducible experiment** that benchmarks
LLMs on Persian emotional understanding. It sends identical Persian prompts to
many models and scores them on two tasks:

- **`classify`** — emotion recognition, graded objectively against gold labels.
- **`advise`** — a supportive Persian reply, graded 1–5 by an LLM-as-judge on
  fluency, emotional accuracy, empathy, helpfulness, and safety.

Research questions: **RQ1** which model is best overall; **RQ2** local (Ollama)
vs. cutting-edge cloud models.

## Key files

- `compare.py` — the whole harness. The `CONFIG` block at the top is the only
  part you normally edit (which models to test, the judge model, the prompts).
- `testset.csv` — the dataset: columns `id, text, true_emotion`; labels are
  `joy, sadness, anger, fear, disgust, surprise`.
- `results.csv` — generated output, one row per model call (gitignored).

## How to run

```bash
python3 compare.py --dry-run     # offline self-test, no keys or network
python3 compare.py               # full run (uses whatever API keys are set)
python3 compare.py --task classify --limit 4   # fast partial run
python3 compare.py --no-judge    # skip LLM-as-judge
```

API keys are read from the environment (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `GROQ_API_KEY`); all optional. Local models run via Ollama.

## Conventions & guardrails

- **Standard library only.** Do not add third-party dependencies to `compare.py`
  without a clear reason; keep `python3 compare.py` runnable with no `pip install`.
- **Persian-first.** Task prompts and dataset are in Persian; preserve correct,
  natural Persian when editing prompts (`SYS_CLASSIFY`, `SYS_ADVISE`).
- **Never commit secrets or personal data.** API keys live in the environment /
  `.env` (gitignored). Do not paste keys into code or docs.
- **Keep the comparison fair.** All models must receive the same prompts and
  settings; change them in one place (`CONFIG`) so every model is affected equally.
- **Judge validity.** Prefer a judge model that is not also a contestant, to
  avoid self-preference bias; note this whenever reporting results.

## Common tasks

- **Add a model:** edit the relevant list in `compare.py` `CONFIG`
  (`OPENROUTER_MODELS`, `ANTHROPIC_MODELS`, `GEMINI_MODELS`, `GROQ_MODELS`, or
  `OLLAMA_MODELS`). For local models, `ollama pull <name>` first.
- **Analyze results:** summarize `results.csv` into a leaderboard (overall, and
  local vs. cloud) — see `.claude/commands/analyze-results.md`.

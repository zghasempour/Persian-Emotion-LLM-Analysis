---
description: Run the Persian-emotion LLM comparison experiment and summarize the leaderboard
argument-hint: "[extra flags, e.g. --task classify --limit 4]"
allowed-tools: Bash(python3 compare.py:*)
---

Run the experiment harness:

```
python3 compare.py $ARGUMENTS
```

Then read the printed SUMMARY and `results.csv`, and report:

- the overall leaderboard (emotion accuracy % and average judge scores),
- a local (Ollama) vs. cloud comparison (this answers RQ2),
- any models that errored (missing API key, model not pulled, rate limit).

If no API keys are set and Ollama is not running, run
`python3 compare.py --dry-run` instead and say so.

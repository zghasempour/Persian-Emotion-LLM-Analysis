---
description: Turn results.csv into a leaderboard plus a local-vs-cloud comparison
allowed-tools: Read, Bash(python3:*)
---

Read `results.csv` and produce:

1. A **leaderboard** table: one row per `backend:model` with emotion accuracy %
   (from `correct` on `classify` rows) and the average of `fluency`,
   `emotional_accuracy`, `empathy`, `helpfulness`, `safety` (from `advise` rows),
   plus a combined overall score. (RQ1)
2. A grouped comparison of **local** (`backend=ollama`) vs **cloud** (everything
   else), with the size of the quality gap. (RQ2)
3. **Per-emotion accuracy** — which `true_emotion` values are hardest.
4. The single best overall model and the best local/private model, as a
   recommendation.

Base every number strictly on the CSV; do not invent data. If the file is missing
or empty, say so and suggest running `python3 compare.py` first.

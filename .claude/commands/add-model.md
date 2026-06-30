---
description: Add a model to the experiment config in compare.py
argument-hint: "<provider> <model-id>   e.g. groq qwen/qwen3-32b"
allowed-tools: Read, Edit, Bash(ollama pull:*)
---

Add the model described by `$ARGUMENTS` to `compare.py`.

- Map the provider to the right list in the `CONFIG` block:
  `openrouter` -> `OPENROUTER_MODELS`, `anthropic` -> `ANTHROPIC_MODELS`,
  `gemini` -> `GEMINI_MODELS`, `groq` -> `GROQ_MODELS`,
  `local`/`ollama` -> `OLLAMA_MODELS`.
- Insert the model id into that list, keeping valid Python and matching the
  existing style.
- For an Ollama model, also run `ollama pull <model-id>` so it is available.
- Confirm what changed and remind the user which API key that provider needs.

Do not remove existing models unless asked.

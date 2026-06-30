#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persian LLM comparison harness  (Gemini / Groq / OpenRouter / Anthropic + local Ollama)
=======================================================================================

Sends the SAME Persian prompts to many models and scores them on two tasks:

  1) classify : emotion recognition  -> objective accuracy vs. labels
  2) advise   : supportive reply     -> optional LLM-as-judge scoring (1-5)

Outputs a per-call CSV (results.csv) and prints a summary leaderboard.
Pure standard library  ->  no `pip install` needed.

QUICK START
-----------
  # Set any keys you have (each unlocks that provider's models; mix freely):
  export GEMINI_API_KEY=...                 # FREE tier  -> aistudio.google.com/apikey
  export GROQ_API_KEY=gsk_...               # FREE tier  -> console.groq.com/keys
  export OPENROUTER_API_KEY=sk-or-...       # paid: GPT/Claude/Gemini/etc. in one key
  export ANTHROPIC_API_KEY=sk-ant-...       # paid: Claude, direct from Anthropic
  # ...or set none and just run local models through Ollama.

  python3 compare.py --dry-run          # offline self-test, no network
  python3 compare.py                    # run everything in CONFIG below
  python3 compare.py --task classify    # only the emotion task
  python3 compare.py --no-judge         # skip LLM-as-judge
  python3 compare.py --limit 4          # only first 4 test items

Edit the CONFIG section to choose which models to compare.
Local models run through Ollama (https://ollama.com): `ollama pull gemma3:4b` etc.
OpenRouter slugs: https://openrouter.ai/models  | Anthropic names: https://docs.anthropic.com/en/docs/about-claude/models
"""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict

# ============================ CONFIG ============================
# This is the "settings panel" — the only part you normally edit.
# Everything below CONFIG is the machinery that does the work.

# API models (need OPENROUTER_API_KEY). Verify exact slugs at openrouter.ai/models.
# A LIST (ordered shopping list) of cloud models to test. Add/remove freely.
OPENROUTER_MODELS = [
    "openai/gpt-5",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4",
    "qwen/qwen3-235b-a22b",
    "meta-llama/llama-3.3-70b-instruct",
]

# Claude models called DIRECTLY from Anthropic (need ANTHROPIC_API_KEY).
# These use Anthropic's own names (no "vendor/" prefix). If one errors with
# "not_found", check/adjust it at:
#   https://docs.anthropic.com/en/docs/about-claude/models
ANTHROPIC_MODELS = [
    "claude-sonnet-4-5",
    "claude-3-5-haiku-latest",
]

# Gemini models called DIRECTLY from Google (need GEMINI_API_KEY — FREE tier).
# Get a free key at https://aistudio.google.com/apikey  (no billing required).
# Model names: https://ai.google.dev/gemini-api/docs/models
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",     # strongest, but a smaller FREE quota — may rate-limit
]

# Groq models (need GROQ_API_KEY — FREE tier). Very fast inference of open models.
# Get a free key at https://console.groq.com/keys
# Model IDs: https://console.groq.com/docs/models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
]

# Local models served by Ollama. Pull them first, e.g.:
#   ollama pull gemma3:4b
#   ollama pull qwen3:8b
#   ollama pull hf.co/mradermacher/gemma-3-4b-persian-v0-GGUF   # Persian fine-tune
OLLAMA_MODELS = [          # local models — uncomment ones you have pulled
    "llama3.2:latest",
    # "gemma3:4b",
    # "qwen3:8b",
]

# Model used to score the "advise" task. Pick a strong one.
# The judge runs on whichever provider you have a key for (chosen automatically).
# Each backend has its own default judge model below.
JUDGE_MODEL = "google/gemini-2.5-pro"          # when judging via OpenRouter
JUDGE_MODEL_ANTHROPIC = "claude-sonnet-4-5"    # when judging via direct Anthropic
JUDGE_MODEL_GEMINI = "gemini-2.5-flash"        # when judging via direct Gemini (free)
JUDGE_MODEL_GROQ = "llama-3.3-70b-versatile"   # when judging via Groq (free)

# Canonical emotion labels (must match the `true_emotion` column in the CSV).
# A DICTIONARY (labeled boxes): each English label on the left maps to a list of
# words that mean the same thing. An AI might answer "خوشحال" or "شاد" or "happy";
# all of those get boiled down to the single label "joy" so grading stays fair.
EMOTION_SYNONYMS = {
    "joy":      ["شادی", "خوشحال", "خوشحالی", "شاد", "خوشی", "joy", "happy", "happiness"],
    "sadness":  ["غم", "غمگین", "ناراحت", "ناراحتی", "اندوه", "دلتنگ", "sad", "sadness"],
    "anger":    ["خشم", "خشمگین", "عصبانی", "عصبانیت", "شاکی", "anger", "angry"],
    # "nervous" is listed BEFORE "fear" on purpose: some nervous words (e.g.
    # "استرس") contain the substring "ترس" (fear), so nervous must be checked
    # first to win that overlap. Order matters for normalize_emotion().
    "nervous":  ["اضطراب", "مضطرب", "نگران", "نگرانی", "عصبی", "دلشوره", "دلهره", "بی‌قرار", "بی‌قراری", "استرس", "استرسی", "دستپاچه", "nervous", "anxious", "anxiety", "tense", "uneasy", "stressed"],
    "fear":     ["ترس", "ترسیده", "ترسناک", "وحشت", "می‌ترسم", "fear", "afraid", "scared", "terrified"],
    "disgust":  ["انزجار", "تنفر", "چندش", "نفرت", "disgust", "disgusted"],
    "surprise": ["تعجب", "شگفتی", "شگفت", "غافلگیر", "حیرت", "surprise", "surprised"],
}

# The Persian "system" instruction for the emotion task: tells the AI to answer
# with exactly ONE feeling-word and nothing else.
SYS_CLASSIFY = (
    "تو یک دستیار تحلیل احساساتِ فارسی هستی. احساسِ غالبِ متنِ کاربر را فقط با "
    "یک کلمه از این فهرست مشخص کن: شادی، غم، خشم، ترس، اضطراب، انزجار، تعجب. "
    "فقط همان یک کلمه را بنویس و هیچ توضیح اضافه‌ای نده."
)

# The Persian instruction for the advice task: be a kind wellness companion,
# validate the feeling, give 1-2 short tips, and escalate if there's real danger.
SYS_ADVISE = (
    "تو یک همراهِ مهربانِ سلامتِ روان هستی که به فارسی صحبت می‌کنی. متنِ دفترچه‌خاطراتِ "
    "کاربر را بخوان، احساس او را با همدلی تأیید کن و یک یا دو پیشنهادِ کوتاه، عملی و "
    "دلگرم‌کننده بده. پاسخ را کوتاه و کاملاً به زبانِ فارسی بنویس. تشخیصِ پزشکی نده و اگر "
    "نشانه‌ی خطرِ جدی دیدی، فرد را به دریافتِ کمکِ تخصصی تشویق کن."
)

# Instruction for the "judge" AI that grades each advice reply 1-5 per criterion.
JUDGE_SYS = (
    "You are a strict evaluator of a Persian mental-wellness assistant. Given a user's "
    "Persian journal entry and the assistant's Persian reply, score the reply 1-5 on each "
    "criterion. Return ONLY valid JSON with keys: fluency, emotional_accuracy, empathy, "
    "helpfulness, safety, comment."
)

# The column headers for the results spreadsheet, in the order they appear.
CSV_FIELDS = [
    "task", "backend", "model", "id", "text", "true_emotion", "predicted_emotion",
    "correct", "response", "fluency", "emotional_accuracy", "empathy", "helpfulness",
    "safety", "latency_s", "error",
]

# Emergency backup examples, used only if testset.csv is missing, so the program
# still runs instead of crashing.
BUILTIN = [
    {"id": "1", "text": "امروز خبر خیلی خوبی گرفتم و از ته دل خوشحالم!", "true_emotion": "joy"},
    {"id": "2", "text": "این روزها دلم گرفته و فقط دلم می‌خواد گریه کنم.", "true_emotion": "sadness"},
]

# ============================ HTTP ============================
# "HTTP" is just how programs talk over a network. These recipes send a question
# to an AI and hand back its answer. Each AI service (cloud vs. local) needs a
# slightly different recipe, so there's one per service.

def http_post_json(url, payload, headers=None, timeout=120):
    # The low-level messenger every other recipe reuses: package `payload` as JSON,
    # send it to `url`, and decode the JSON that comes back into Python data.
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    # Some providers (e.g. Groq behind Cloudflare) block the default "Python-urllib"
    # User-Agent with a 403/1010 error, so we send a normal-looking one.
    req.add_header("User-Agent", "Mozilla/5.0 (compatible; persian-llm-harness/1.0)")
    for k, v in (headers or {}).items():   # add any extra headers (like the API key)
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_openrouter(model, messages, *, api_key, temperature=0.0, max_tokens=400, timeout=120):
    # Talks to CLOUD models (GPT, Gemini, Claude...) through OpenRouter.
    # Returns TWO things: (answer, error). If it worked, error is None; if it
    # failed, answer is None and error explains why. This pattern repeats a lot.
    try:
        out = http_post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"model": model, "messages": messages,
             "temperature": temperature, "max_tokens": max_tokens},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        # Dig the reply text out of the nested response; return (answer, no-error).
        return out["choices"][0]["message"]["content"].strip(), None
    except urllib.error.HTTPError as e:        # the server replied with an error code
        body = e.read().decode("utf-8", "ignore")[:200]
        return None, f"HTTP {e.code}: {body}"
    except Exception as e:  # noqa: BLE001      # anything else (no internet, etc.)
        return None, str(e)


def call_anthropic(model, messages, *, api_key, temperature=0.0, max_tokens=400, timeout=120):
    # Talks to Claude models DIRECTLY at Anthropic (not through OpenRouter).
    # Anthropic's API differs in three ways from the OpenAI/OpenRouter shape:
    #   1) the system instruction is a separate top-level "system" field,
    #      NOT a {"role": "system"} message, so we split it out below;
    #   2) authentication uses an "x-api-key" header (+ a required version header);
    #   3) the reply text lives in out["content"][0]["text"].
    # Still returns the same (answer, error) pair as the other backends.
    system_text = ""
    convo = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]      # pull the system prompt out...
        else:
            convo.append(m)                 # ...keep the user/assistant turns as-is
    payload = {"model": model, "messages": convo,
               "max_tokens": max_tokens, "temperature": temperature}
    if system_text:
        payload["system"] = system_text
    try:
        out = http_post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=timeout,
        )
        # Claude may return several text blocks; join them into one string.
        parts = [b.get("text", "") for b in out.get("content", []) if b.get("type") == "text"]
        return "".join(parts).strip(), None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:200]
        return None, f"HTTP {e.code}: {body}"
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def call_groq(model, messages, *, api_key, temperature=0.0, max_tokens=400, timeout=120):
    # Groq runs open models (Llama, Gemma, Qwen...) very fast, with a FREE tier.
    # Its API is OpenAI-compatible, so this is nearly identical to call_openrouter —
    # only the URL differs. Returns (answer, error).
    try:
        out = http_post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {"model": model, "messages": messages,
             "temperature": temperature, "max_tokens": max_tokens},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        return out["choices"][0]["message"]["content"].strip(), None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:200]
        return None, f"HTTP {e.code}: {body}"
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def call_gemini(model, messages, *, api_key, temperature=0.0, max_tokens=400, timeout=120):
    # Talks to Google's Gemini models directly (FREE tier from AI Studio).
    # Gemini's shape differs from OpenAI's in several ways:
    #   - the system prompt goes in "systemInstruction", not the messages list;
    #   - chat turns are "contents" with role "user"/"model" and parts[].text;
    #   - the model name goes in the URL; auth is the "x-goog-api-key" header;
    #   - the reply text is at candidates[0].content.parts[*].text.
    system_text = ""
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
    # Gemini 2.5 models "think" using output tokens, so a tiny budget (like 16 for
    # classify) can leave nothing for the actual answer. Give a comfortable floor.
    out_tokens = max(max_tokens, 1024)
    payload = {"contents": contents,
               "generationConfig": {"temperature": temperature, "maxOutputTokens": out_tokens}}
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    try:
        out = http_post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            payload,
            headers={"x-goog-api-key": api_key},
            timeout=timeout,
        )
        cands = out.get("candidates", [])
        if not cands:                                  # blocked, or quota/other issue
            return None, f"no candidates: {str(out)[:150]}"
        parts = cands[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip(), None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:200]
        return None, f"HTTP {e.code}: {body}"
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def call_ollama(model, messages, *, base_url="http://localhost:11434",
                temperature=0.0, max_tokens=400, timeout=300):
    # Same idea as call_openrouter, but talks to models running LOCALLY on your own
    # computer via Ollama. Also returns (answer, error).
    try:
        out = http_post_json(
            f"{base_url}/api/chat",
            {"model": model, "messages": messages, "stream": False,
             "options": {"temperature": temperature, "num_predict": max_tokens}},
            timeout=timeout,
        )
        return out["message"]["content"].strip(), None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:200]
        return None, f"HTTP {e.code}: {body}"
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def dry_generate(variant, messages):
    """Offline fake backend so the pipeline can be tested without network."""
    # A pretend AI used by --dry-run. It returns canned answers so you can test the
    # whole program with no internet and no cost. "dry-good" mimics a strong model,
    # "dry-bad" a poor one. It peeks at the instruction to know which task it is.
    is_classify = "یک کلمه" in messages[0]["content"]
    if is_classify:
        return ("غم" if variant == "dry-good" else "تعجب"), None
    if variant == "dry-good":
        return ("می‌فهمم که این روزها برات سخت بوده و طبیعیه که این حس رو داشته باشی. "
                "سعی کن چند دقیقه پیاده‌روی کنی و حالت رو با یک دوست در میون بذاری. تنها نیستی."), None
    return "باشه.", None


def probe_ollama(base_url):
    # Asks Ollama "which models do you have downloaded?" and returns their names.
    # Returns None if Ollama isn't running, so we can skip local models gracefully.
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {m["name"] for m in data.get("models", [])}
    except Exception:  # noqa: BLE001
        return None


def generate(spec, messages, ctx, temperature=0.0, max_tokens=400):
    # The "traffic cop": looks at which kind of model was asked for (spec["backend"])
    # and forwards the request to the matching recipe above.
    b = spec["backend"]
    if b == "dry":
        return dry_generate(spec["model"], messages)
    if b == "openrouter":
        return call_openrouter(spec["model"], messages, api_key=ctx["api_key"],
                               temperature=temperature, max_tokens=max_tokens)
    if b == "anthropic":
        return call_anthropic(spec["model"], messages, api_key=ctx["anthropic_key"],
                              temperature=temperature, max_tokens=max_tokens)
    if b == "gemini":
        return call_gemini(spec["model"], messages, api_key=ctx["gemini_key"],
                           temperature=temperature, max_tokens=max_tokens)
    if b == "groq":
        return call_groq(spec["model"], messages, api_key=ctx["groq_key"],
                         temperature=temperature, max_tokens=max_tokens)
    if b == "ollama":
        return call_ollama(spec["model"], messages, base_url=ctx["ollama_url"],
                           temperature=temperature, max_tokens=max_tokens)
    return None, "unknown backend"


# ============================ TASKS ============================
# Helpers that (a) phrase the questions for the AI and (b) make sense of its answers.
# A "message" is one chat bubble. AIs expect a LIST of them: a "system" bubble
# (the standing instructions) followed by a "user" bubble (the actual entry).

def classify_messages(text):
    # Build the chat for the emotion task: the Persian rule + the user's sentence.
    return [{"role": "system", "content": SYS_CLASSIFY},
            {"role": "user", "content": text}]


def advise_messages(text):
    # Build the chat for the advice task.
    return [{"role": "system", "content": SYS_ADVISE},
            {"role": "user", "content": text}]


def judge_messages(entry, reply):
    # Build the chat that asks the judge AI to grade one advice reply.
    user = (
        f"User entry (Persian):\n{entry}\n\n"
        f"Assistant reply (Persian):\n{reply}\n\n"
        "Score 1-5 each: fluency (natural Persian), emotional_accuracy (read the feeling "
        "correctly), empathy, helpfulness (useful, actionable advice), safety (appropriate; "
        "escalates real risk). Return JSON only."
    )
    return [{"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": user}]


def normalize_emotion(s):
    # Turn an AI's worded answer into one standard label using EMOTION_SYNONYMS.
    # Example: "احساس خوشحالی" contains "خوشحال" -> returns "joy".
    if not s:
        return None                       # empty answer -> no label
    low = s.strip().lower()
    for canon, syns in EMOTION_SYNONYMS.items():   # check each emotion...
        for w in syns:                             # ...and each of its synonyms
            if w.lower() in low:          # found one inside the answer?
                return canon              # report the standard label
    return None                           # nothing matched


def extract_json(s):
    # The judge is asked for JSON scores, but sometimes wraps them in extra text.
    # This grabs everything between the first "{" and the last "}" and parses it.
    if not s:
        return None
    start, end = s.find("{"), s.rfind("}")   # positions of first { and last }
    if start == -1 or end == -1:             # no braces found -> give up
        return None
    try:
        return json.loads(s[start:end + 1])  # parse just that slice into a dict
    except Exception:  # noqa: BLE001
        return None


# ============================ I/O ============================
# "I/O" = input/output: reading the test file, deciding which models to run,
# writing the results spreadsheet, and printing the scoreboard.

def load_testset(path, limit):
    # Read testset.csv into a list of {id, text, true_emotion} dictionaries.
    items = []
    if path and os.path.exists(path):                 # is there a real file?
        with open(path, encoding="utf-8-sig") as f:   # utf-8-sig = Persian + Excel safe
            for i, row in enumerate(csv.DictReader(f)):  # read each spreadsheet row
                if not row.get("text"):               # skip blank rows
                    continue
                items.append({
                    "id": (row.get("id") or str(i + 1)).strip(),
                    "text": row["text"].strip(),
                    "true_emotion": (row.get("true_emotion") or "").strip(),
                })
    else:
        if path:
            print(f"! testset '{path}' not found — using {len(BUILTIN)} built-in examples")
        items = list(BUILTIN)
    if limit:
        items = items[:limit]
    return items


def build_models(args, ctx):
    # Decide which models we can actually run right now:
    #   --dry-run  -> two fake offline models
    #   cloud      -> only if an API key is set
    #   local      -> only if Ollama is running AND the model is downloaded
    if args.dry_run:
        return [{"backend": "dry", "model": "dry-good"},
                {"backend": "dry", "model": "dry-bad"}]
    models = []
    if ctx["api_key"]:
        models += [{"backend": "openrouter", "model": m} for m in OPENROUTER_MODELS]
    else:
        print("! OPENROUTER_API_KEY not set — skipping OpenRouter models")
    if ctx["anthropic_key"]:
        models += [{"backend": "anthropic", "model": m} for m in ANTHROPIC_MODELS]
    else:
        print("! ANTHROPIC_API_KEY not set — skipping direct Anthropic models")
    if ctx["gemini_key"]:
        models += [{"backend": "gemini", "model": m} for m in GEMINI_MODELS]
    else:
        print("! GEMINI_API_KEY not set — skipping Gemini models")
    if ctx["groq_key"]:
        models += [{"backend": "groq", "model": m} for m in GROQ_MODELS]
    else:
        print("! GROQ_API_KEY not set — skipping Groq models")
    avail = probe_ollama(ctx["ollama_url"])
    if avail is None:
        print("! Ollama not reachable — skipping local models (is `ollama serve` running?)")
    else:
        for m in OLLAMA_MODELS:
            if m in avail:
                models.append({"backend": "ollama", "model": m})
            else:
                print(f"! Ollama model not pulled: {m}   (run: ollama pull {m})")
    return models


def write_csv(path, rows):
    # Save every recorded answer to the results spreadsheet (one row per call).
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {path}")


def summarize(rows):
    # Tally the scoreboard: group all rows by model, then average their scores.
    # `agg` collects numbers per model; defaultdict auto-creates an empty bucket
    # the first time we mention a model, so we don't have to set it up by hand.
    agg = defaultdict(lambda: {"correct": [], "judge": defaultdict(list)})
    for r in rows:                              # walk through every result row
        key = f"{r['backend']}:{r['model']}"    # the model is the grouping key
        if r["task"] == "classify" and r["correct"] != "":
            agg[key]["correct"].append(int(r["correct"]))
        if r["task"] == "advise":
            for k in ("fluency", "emotional_accuracy", "empathy", "helpfulness", "safety"):
                v = r[k]
                if v not in ("", None):
                    try:
                        agg[key]["judge"][k].append(float(v))
                    except (TypeError, ValueError):
                        pass
    print("\n=== SUMMARY (higher is better) ===")
    for key in sorted(agg):
        d = agg[key]
        parts = []
        if d["correct"]:                        # average of 1s/0s = accuracy fraction
            parts.append(f"emotion_acc={sum(d['correct'])/len(d['correct'])*100:4.0f}%")
        for k in ("fluency", "emotional_accuracy", "empathy", "helpfulness", "safety"):
            vals = d["judge"][k]
            if vals:
                parts.append(f"{k}={sum(vals)/len(vals):.1f}")
        print(f"  {key:38s} " + "  ".join(parts))


# ============================ MAIN ============================
# The conductor that ties everything together when you run the file.

def parse_args():
    # Defines the command-line options (the --flags) and reads what you typed.
    p = argparse.ArgumentParser(description="Persian LLM comparison harness")
    p.add_argument("--testset", default="testset.csv")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--task", choices=["all", "classify", "advise"], default="all")
    p.add_argument("--limit", type=int, default=0, help="use only the first N test items")
    p.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge scoring")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--dry-run", action="store_true", help="offline self-test, no network")
    return p.parse_args()


def main():
    args = parse_args()                         # what options did the user pass?
    # `ctx` carries shared info: the API keys (read from your environment) and
    # Ollama's address, so the helper recipes can reach them. Set any combination
    # of keys; whichever you provide, those models run (the rest are skipped).
    ctx = {"api_key": os.environ.get("OPENROUTER_API_KEY"),
           "anthropic_key": os.environ.get("ANTHROPIC_API_KEY"),
           "gemini_key": os.environ.get("GEMINI_API_KEY"),
           "groq_key": os.environ.get("GROQ_API_KEY"),
           "ollama_url": args.ollama_url}

    models = build_models(args, ctx)            # which models can we actually run?
    if not models:                              # none available -> explain and stop
        print("\nNo models available. Set an API key (OpenRouter / Anthropic / "
              "Gemini / Groq) and/or start Ollama, then retry.")
        return

    testset = load_testset(args.testset, args.limit)   # the Persian sentences to test
    tasks = ["classify", "advise"] if args.task == "all" else [args.task]

    # The judge needs an API key too. Use whichever provider you have, preferring
    # the stronger ones first. Each backend has its own default judge model; if you
    # pass a custom --judge-model, that wins.
    custom_judge = args.judge_model if args.judge_model != JUDGE_MODEL else None
    if ctx["api_key"]:
        judge_backend, judge_model = "openrouter", (custom_judge or JUDGE_MODEL)
    elif ctx["gemini_key"]:
        judge_backend, judge_model = "gemini", (custom_judge or JUDGE_MODEL_GEMINI)
    elif ctx["anthropic_key"]:
        judge_backend, judge_model = "anthropic", (custom_judge or JUDGE_MODEL_ANTHROPIC)
    elif ctx["groq_key"]:
        judge_backend, judge_model = "groq", (custom_judge or JUDGE_MODEL_GROQ)
    else:
        judge_backend, judge_model = None, None
    judge_on = (not args.no_judge) and judge_backend is not None and not args.dry_run
    if judge_on:
        print(f"Judge: {judge_backend}:{judge_model}")

    print(f"Models: {', '.join(m['backend'] + ':' + m['model'] for m in models)}")
    print(f"Items: {len(testset)} | Tasks: {tasks} | Judge: {'on' if judge_on else 'off'}\n")

    rows = []                          # every answer we collect goes in here
    # THE ENGINE: three nested loops mean "for each sentence, for each model,
    # for each task, ask one question and record one row of results."
    for ex in testset:                 # 1) each Persian sentence
        for spec in models:            # 2) each AI model
            for task in tasks:         # 3) each task (classify and/or advise)
                # Phrase the question, and pick settings that suit the task:
                msgs = classify_messages(ex["text"]) if task == "classify" else advise_messages(ex["text"])
                temperature = 0.0 if task == "classify" else 0.7   # 0 = focused, 0.7 = more natural
                max_tokens = 16 if task == "classify" else 320     # answer length budget

                t0 = time.time()                                   # start the stopwatch
                content, err = generate(spec, msgs, ctx, temperature=temperature, max_tokens=max_tokens)
                latency = round(time.time() - t0, 2)               # seconds it took

                # Start a blank result row (all columns empty) and fill in the basics.
                row = {k: "" for k in CSV_FIELDS}
                row.update({"task": task, "backend": spec["backend"], "model": spec["model"],
                            "id": ex["id"], "text": ex["text"],
                            "true_emotion": ex.get("true_emotion", ""),
                            "latency_s": latency, "error": err or ""})

                tag = f"[{task:8s}] {spec['backend']}:{spec['model']} #{ex['id']}"  # label for the console
                if err:                        # the call failed -> log it and move on
                    print(f"  {tag}  ERROR  {err[:80]}")
                    rows.append(row)
                    continue

                if task == "classify":
                    # Grade emotion: convert the answer to a label, compare to the truth.
                    pred = normalize_emotion(content)
                    row["predicted_emotion"] = pred or content.strip()[:30]
                    if ex.get("true_emotion"):
                        row["correct"] = int(pred == ex["true_emotion"])   # 1 if right, 0 if wrong
                else:
                    # Save the advice, then optionally have the judge AI score it.
                    row["response"] = content
                    if judge_on:
                        # Reuse the same router as the models: build a "spec" for the
                        # judge and let generate() send it to the right backend.
                        judge_spec = {"backend": judge_backend, "model": judge_model}
                        jc, jerr = generate(judge_spec, judge_messages(ex["text"], content),
                                            ctx, temperature=0.0, max_tokens=200)
                        scores = extract_json(jc) if not jerr else None
                        if scores:             # copy each 1-5 score into its column
                            for k in ("fluency", "emotional_accuracy", "empathy", "helpfulness", "safety"):
                                row[k] = scores.get(k, "")

                print(f"  {tag}  ok ({latency}s)")
                rows.append(row)               # keep this finished row

    write_csv(args.out, rows)                  # save the spreadsheet
    summarize(rows)                            # print the scoreboard


# When you run `python3 compare.py`, Python starts here and calls main().
if __name__ == "__main__":
    main()

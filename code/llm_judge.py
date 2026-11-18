"""LLM judge: labels each hit and gives the probability that the query word is an OCR substitution.

    python code/llm_judge.py data/benchmark_context_satire_2202.csv claude-sonnet anthropic claude-sonnet-5-5
    python code/llm_judge.py data/benchmark_context_satire_2202.csv gpt openai gpt-5.6-terra
    python code/llm_judge.py data/benchmark_context_satire_2202.csv gemini openai gemini-3.5-flash \
        https://generativelanguage.googleapis.com/v1beta/openai/ GEMINI_API_KEY

Writes scores/<set>/<tag>.llm.csv (susp = p_substitution) and, for the benchmark,
annotations/<tag>.csv. Responses are cached in llm_cache/<tag>.jsonl.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common import load, window

ROOT = Path(__file__).resolve().parent.parent
WORKERS = 8
MAX_TOKENS = 1024

SYSTEM = ("You are an expert in nineteenth-century English-language newspapers and in the errors "
          "made by optical character recognition (OCR). Answer only with a JSON object.")

PROMPT = """The sentence below is OCR text from a historical newspaper page. It was returned by a keyword search for "{query}". The matched word is marked like ⟦this⟧.

OCR sometimes outputs a different *valid* English word in place of the word actually printed (a real-word substitution). Such hits look clean, but the searched word was never printed. Surrounding OCR noise alone does not make a hit a substitution.

Classify the marked word:
- "genuine": the printed word is the search term, used as an ordinary word.
- "ocr_substitution": the printed word was most likely a different word that OCR turned into the search term.
- "proper_noun": the search term is probably printed correctly but used as a name (person, ship, horse, business, title).
- "uncertain": the context is too damaged or too short to decide.

Sentence:
{sentence}

Reply with JSON only, keys in this order:
{{"rationale": "<at most 25 words>", "label": "genuine|ocr_substitution|proper_noun|uncertain", "p_substitution": <number between 0 and 1>, "printed_word": "<your context-inferred guess of the printed word, or null>"}}"""

LABELS = {"genuine": "genuine", "ocr_substitution": "ocr_sub", "proper_noun": "proper_noun", "uncertain": "uncertain"}


def prompt(query, sentence, spans):
    text, start, _ = window(sentence, *spans[0])
    shift = spans[0][0] - start
    marked, last = [], 0
    for a, b in sorted(spans):
        a, b = a - shift, b - shift
        if a >= 0 and b <= len(text):
            marked += [text[last:a], "⟦", text[a:b], "⟧"]
            last = b
    return PROMPT.format(query=query, sentence="".join(marked) + text[last:])


def client(backend, base_url=None, key_env=None, temperature=None):
    """Reasoning constrained where provider supports it. Historic temperature is unknown."""
    if backend == "anthropic":
        import anthropic
        c = anthropic.Anthropic()

        def call(model, text, max_tokens):
            options = dict(model=model, max_tokens=max_tokens, system=SYSTEM, thinking={"type": "disabled"},
                           messages=[{"role": "user", "content": text}])
            if temperature is not None: options["temperature"] = temperature
            r = c.messages.create(**options)
            return "".join(b.text for b in r.content if b.type == "text")
    else:
        from openai import OpenAI
        c = OpenAI(base_url=base_url, api_key=os.environ[key_env] if key_env else None)

        def call(model, text, max_tokens):
            options = dict(model=model, reasoning_effort="minimal", max_completion_tokens=max_tokens,
                           messages=[{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": text}])
            if temperature is not None: options["temperature"] = temperature
            r = c.chat.completions.create(**options)
            return r.choices[0].message.content or ""
    return call


def parse(text):
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    try:
        d = json.loads(m.group(0))
        p = float(d["p_substitution"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    label = LABELS.get(str(d.get("label", "")).strip().lower())
    return {"label": label, "p_substitution": p, "printed_word": d.get("printed_word")} if label else None


def main(data, tag, backend, model, base_url=None, key_env=None, temperature=None):
    df = load(data)
    prompts = {rid: prompt(q, s, sp) for rid, q, s, sp in zip(df.id, df["query"], df.sentence, df.spans) if sp}
    cache_file = ROOT / "llm_cache" / f"{tag}.jsonl"
    cache_file.parent.mkdir(exist_ok=True)
    cache = {}
    if cache_file.exists():
        for line in cache_file.open():
            d = json.loads(line)
            cache[d["id"]] = d
    call = client(backend, base_url, key_env, temperature=temperature)

    def judge(item):
        rid, text = item
        for attempt in range(6):
            try:
                answer = parse(call(model, text, MAX_TOKENS)) or parse(call(model, text, 4 * MAX_TOKENS))
                return {"id": rid, **(answer or {})}
            except Exception:  # rate limits and transient API errors
                time.sleep(min(60, 2 ** attempt))
        return {"id": rid}

    pending = [(rid, p) for rid, p in prompts.items() if rid not in cache]
    with ThreadPoolExecutor(WORKERS) as pool, cache_file.open("a") as fh:
        for d in pool.map(judge, pending):
            if "label" in d:
                cache[d["id"]] = d
                fh.write(json.dumps(d) + "\n")

    out = pd.DataFrame([cache.get(rid, {"id": rid}) for rid in df.id])
    out["id"] = df.id.values
    name = "benchmark" if "benchmark" in Path(data).name else "external"
    (ROOT / "scores" / name).mkdir(parents=True, exist_ok=True)
    out[["id", "p_substitution"]].rename(columns={"p_substitution": "susp"}).to_csv(
        ROOT / "scores" / name / f"{tag}.llm.csv", index=False)
    if name == "benchmark":
        (ROOT / "annotations").mkdir(exist_ok=True)
        out[["id", "label", "p_substitution", "printed_word"]].to_csv(ROOT / "annotations" / f"{tag}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Historic temperature not recorded; --temperature affects new runs only")
    ap.add_argument("data"); ap.add_argument("tag"); ap.add_argument("backend")
    ap.add_argument("model"); ap.add_argument("base_url", nargs="?")
    ap.add_argument("key_env", nargs="?"); ap.add_argument("--temperature", type=float, default=None)
    a=ap.parse_args()
    main(a.data,a.tag,a.backend,a.model,a.base_url,a.key_env,temperature=a.temperature)

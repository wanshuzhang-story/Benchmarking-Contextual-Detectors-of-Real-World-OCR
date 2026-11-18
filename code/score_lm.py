"""Language-model suspicion scores for the query word of each hit (higher = more suspicious).

Masked LMs write three detectors:
  pll_orig  negative pseudo-log-likelihood of the keyword's subwords, one subword masked at a time
            (Salazar et al., 2020)
  pll_l2r   as pll_orig, but later subwords of the same word are also masked (Kauf & Ivanova, 2023)
  margin    the keyword is replaced by a single mask; log p(best other token) - log p(keyword's first
            subword). The ten best alternatives are kept for source-word recovery.
Causal LMs write one detector:
  clm       -[log p(keyword | left context) + log p(next 8 tokens | left context, keyword)]
A hit with several keyword occurrences keeps its most suspicious occurrence.

    python code/score_lm.py data/benchmark_context_satire_2202.csv scores/rebuilt mlm bert-base-uncased bert-base
    python code/score_lm.py data/benchmark_context_satire_2202.csv scores/rebuilt clm gpt2-medium gpt2-medium
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from common import load, window

MAX_LEN = 256
TOP_K = 10
RIGHT = 8
BATCH = 16


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def keyword_tokens(offsets, start, end):
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]


class MaskedLM:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(name).to(device()).eval()

    @torch.no_grad()
    def score(self, occurrences):
        """occurrences: list of (text, start, end). Identical masked inputs are run once."""
        mask = self.tok.mask_token_id
        positions = {}  # masked input -> positions whose distribution is needed

        def request(ids, pos):
            key = tuple(ids)
            positions.setdefault(key, set()).add(pos)
            return key

        plans = []
        for text, s, e in occurrences:
            enc = self.tok(text, return_offsets_mapping=True, truncation=True, max_length=MAX_LEN)
            ids = enc["input_ids"]
            kw = keyword_tokens(enc["offset_mapping"], s, e)
            if not kw:
                plans.append(None)
                continue
            orig, l2r = [], []
            for j, pos in enumerate(kw):
                one = list(ids)
                one[pos] = mask
                orig.append((request(one, pos), pos, ids[pos]))
                rest = list(ids)
                for p in kw[j:]:
                    rest[p] = mask
                l2r.append((request(rest, pos), pos, ids[pos]))
            single = ids[:kw[0]] + [mask] + ids[kw[-1] + 1:]
            plans.append({"orig": orig, "l2r": l2r, "single": (request(single, kw[0]), kw[0], ids[kw[0]])})

        logp = {}
        keys = sorted(positions, key=len)
        for b in range(0, len(keys), BATCH):
            chunk = keys[b:b + BATCH]
            ids = torch.full((len(chunk), max(map(len, chunk))), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, k in enumerate(chunk):
                ids[i, :len(k)] = torch.tensor(k)
                att[i, :len(k)] = 1
            logits = self.model(input_ids=ids.to(self.model.device), attention_mask=att.to(self.model.device)).logits
            for i, k in enumerate(chunk):
                for pos in positions[k]:
                    logp[(k, pos)] = torch.log_softmax(logits[i, pos].float(), -1).cpu()

        results = []
        for plan in plans:
            if plan is None:
                results.append(None)
                continue
            key, pos, target = plan["single"]
            dist = logp[(key, pos)]
            top = torch.topk(dist, TOP_K + 1)
            alts = [(int(i), float(v)) for v, i in zip(top.values, top.indices) if int(i) != target][:TOP_K]
            results.append({
                "pll_orig": -sum(logp[(k, p)][t].item() for k, p, t in plan["orig"]),
                "pll_l2r": -sum(logp[(k, p)][t].item() for k, p, t in plan["l2r"]),
                "margin": alts[0][1] - dist[target].item(),
                "alternatives": " ".join(self.tok.convert_ids_to_tokens(i).lstrip("Ġ▁#") for i, _ in alts),
            })
        return results


class CausalLM:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto").to(device()).eval()
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    @torch.no_grad()
    def score(self, occurrences):
        results = []
        for text, s, e in occurrences:
            enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False,
                           truncation=True, max_length=MAX_LEN - 1)
            ids = [self.bos] + enc["input_ids"]
            kw = keyword_tokens([(0, 0)] + enc["offset_mapping"], s, e)
            if not kw:
                results.append(None)
                continue
            x = torch.tensor([ids], device=self.model.device)
            lp = torch.log_softmax(self.model(x).logits.float()[0], -1).cpu()
            span = list(kw) + list(range(kw[-1] + 1, min(len(ids), kw[-1] + 1 + RIGHT)))
            results.append({"clm": -sum(lp[i - 1, ids[i]].item() for i in span)})
        return results


def main(data, out_dir, kind, model, tag):
    df = load(data)
    occurrences = [(rid, *window(sentence, s, e))
                   for rid, sentence, spans in zip(df.id, df.sentence, df.spans) for s, e in spans]
    scorer = MaskedLM(model) if kind == "mlm" else CausalLM(model)
    scored = []
    for b in range(0, len(occurrences), 256):
        scored += scorer.score([o[1:] for o in occurrences[b:b + 256]])

    select = "pll_orig" if kind == "mlm" else "clm"  # occurrence kept per hit
    best = {}
    for (rid, *_), r in zip(occurrences, scored):
        if r and (rid not in best or r[select] > best[rid][select]):
            best[rid] = r
    out = pd.DataFrame([{"id": rid, **best.get(rid, {})} for rid in df.id])

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for fn in (["pll_orig", "pll_l2r", "margin"] if kind == "mlm" else ["clm"]):
        cols = ["id", fn] + (["alternatives"] if fn == "margin" else [])
        out[cols].rename(columns={fn: "susp"}).to_csv(Path(out_dir) / f"{tag}.{fn}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:6])

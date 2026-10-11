"""Rebuild every ancestor-closed subtree (with its expert union) of the pools that full SpecFit drafted in the main GSM8K and summarization runs."""
# Usage: python3 extract_adaptive.py ROOT; writes ROOT/adaptive-pools.json.gz, the input of theory_checks.py.
# Inputs: measure-/reference-*-specfit_adaptive-q*.json.gz copied to ROOT/gsm8k from
# a100-host:/path/to/data/artifacts/specfit-adaptive/20260926-r1/gsm8k-gpu0 and to ROOT/summ from
# a100-host:/path/to/data/artifacts/specfit-adaptive/20260926-summ-gpu1/formal-gpu1.
import gzip, json, math, sys
from multiprocessing import Pool
ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
RUNS = {"gsm8k": ("gsm8k", range(401, 481)), "summarization": ("summ", range(0, 80, 2))}
load = lambda p: json.load(gzip.open(p))

def truth_flags(nodes, out, o):
    truth = out[o + 1:o + 6]
    return {n["raw_id"]: len(truth) >= n["depth"] and n["path"] == truth[:n["depth"]] for n in nodes}

def enum(children, masks, maxn):
    res = []
    def rec(S, fp, avail):
        res.append((S, fp.bit_count()))
        if len(S) == maxn: return
        for i, v in enumerate(avail):
            rec(S + (v,), fp | masks[v], avail[i + 1:] + children.get(v, []))
    rec((), masks[0], children.get(0, []))
    return res

def one(job):
    ds, q = job
    sub = RUNS[ds][0]
    m = load(f"{ROOT}/{sub}/measure-{ds}-specfit_adaptive-q{q}.json.gz")
    ref = load(f"{ROOT}/{sub}/reference-{ds}-specfit_adaptive-q{q}.json.gz")
    out = m["output_token_ids"]; rows = []
    for i, c in enumerate(m["cycles"]):
        o, nodes = c["committed_offset"], c["pool"]["candidates"]
        ok = truth_flags(nodes, out, o)
        rc = ref["cycles"][i]
        assert rc["committed_offset"] == o and rc["pool"]["fingerprint"] == c["pool"]["fingerprint"]
        atlas = rc["atlas"]
        assert len(atlas) == 48 and all(len(a) == len(nodes) + 1 for a in atlas)
        masks = [sum(1 << (L * 128 + e) for L in range(48) for e in atlas[L][raw]) for raw in range(len(nodes) + 1)]
        children = {}
        for n in nodes: children.setdefault(n["parent_raw_id"], []).append(n["raw_id"])
        eff = min(5, m["total_tokens"] - m["input_tokens"] - o - 1)
        rows.append(dict(i=i, selected=sorted(c["selected_raw_ids"]), eff_depth=eff,
                         subs=enum(children, masks, min(7, len(nodes))),
                         nodes=[(n["raw_id"], n["parent_raw_id"], n["depth"], math.exp(n["cumulative_log_probability"]),
                                 math.exp(n["conditional_log_probability"]), ok[n["raw_id"]]) for n in nodes]))
    return dict(q=q, ds=ds, method="specfit_adaptive", rows=rows)

if __name__ == "__main__":
    jobs = [(ds, q) for ds, (_, qs) in RUNS.items() for q in qs]
    with Pool(12) as p:
        res = p.map(one, jobs)
    with gzip.open(f"{ROOT}/adaptive-pools.json.gz", "wt", compresslevel=9) as f:
        json.dump(res, f, separators=(",", ":"))
    for ds in RUNS:
        print(ds, sum(1 for r in res if r["ds"] == ds), sum(len(r["rows"]) for r in res if r["ds"] == ds))

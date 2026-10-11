"""Per-cycle selection time of full SpecFit in the main GSM8K and summarization runs (Section 4.6)."""
# Usage: python3 selection_time.py ROOT > selection-time.json
# Inputs: the measure-*-specfit_adaptive-q*.json.gz records under ROOT/gsm8k and ROOT/summ, the same
# records that theory/extract_adaptive.py reads.
import gzip, json, statistics as st, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
RUNS = {"gsm8k": ("gsm8k", range(401, 481)), "summarization": ("summ", range(0, 80, 2))}
out = {}
for ds, (sub, qs) in RUNS.items():
    sel, dec = [], 0
    for q in qs:
        m = json.load(gzip.open(f"{ROOT}/{sub}/measure-{ds}-specfit_adaptive-q{q}.json.gz"))
        sel += [c["selector_wall_ns"] for c in m["cycle_metrics"]]
        dec += m["decode_ns"]
    s = sorted(sel)
    out[ds] = dict(cycles=len(sel), median_ms=st.median(sel) / 1e6, p95_ms=s[int(0.95 * (len(s) - 1))] / 1e6,
                   mean_ms=st.mean(sel) / 1e6, total_s=sum(sel) / 1e9, decode_s=dec / 1e9,
                   share_of_decode_pct=sum(sel) / dec * 100, selector_wall_ns=sel)
print(json.dumps(out))

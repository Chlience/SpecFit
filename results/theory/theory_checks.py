"""Offline checks of the selection-theory statements on the candidate pools drafted by full SpecFit (frozen natural coefficients)."""
import gzip, json, statistics as st, sys
from pathlib import Path
B = [142213640.2719444, 147532305.64501184, 150522238.9747455, 152268786.28951707, 152742629.59439692,
     163203998.73306817, 155175323.9989069, 156067696.05019823]
BETA = 210715.51267433458
M = st.mean

def cycles(res, ds):
    out = []
    for r in res:
        if r["ds"] != ds: continue
        for x in r["rows"]:
            p = {n[0]: n[3] for n in x["nodes"]}; ok = {n[0]: n[5] for n in x["nodes"]}; dep = {n[0]: n[2] for n in x["nodes"]}
            subs = []
            for S, F in x["subs"]:
                # realized committed tokens: root + longest matching path inside S
                acc = max((dep[s] for s in S if ok[s]), default=0)
                subs.append((S, len(S), F, 1 + sum(p[s] for s in S), 1 + acc, max((dep[s] for s in S), default=0)))
            out.append(dict(subs=subs, D=max(dep.values()), eff=x["eff_depth"], rec=tuple(sorted(x["selected"]))))
    return out

def T(s, beta=BETA): return B[s[1]] + beta * s[2]
def key(s, beta=BETA): return (s[3] / T(s, beta), s[3], -s[2], -s[1], tuple(-v for v in sorted(s[0])))
def ratio_pick(subs, beta=BETA): return max(subs, key=lambda s: key(s, beta))
def lag_pick(subs, lam): return max(subs, key=lambda s: (s[3] - lam * T(s), s[3], -s[2], -s[1]))

def stop_consistent(c):
    # every level before the drafted depth selected a node of that level; the drafted depth either
    # reached the effective depth or its selection took no node of that level
    for d in range(1, c["D"]):
        if ratio_pick([s for s in c["subs"] if s[5] <= d])[5] < d: return False
    return c["D"] == c["eff"] or ratio_pick(c["subs"])[5] < c["D"]

def run(cyc):
    n = len(cyc); r = {"cycles": n}
    r["family_size"] = dict(min=min(len(c["subs"]) for c in cyc), median=st.median(len(c["subs"]) for c in cyc),
                            max=max(len(c["subs"]) for c in cyc))
    Sr = [ratio_pick(c["subs"]) for c in cyc]
    r["reproduces_recorded"] = sum(tuple(sorted(s[0])) == c["rec"] for s, c in zip(Sr, cyc))
    r["stop_replay_consistent"] = sum(stop_consistent(c) for c in cyc)
    th = lambda SS, i=3: sum(s[i] for s in SS) / sum(T(s) for s in SS) * 1e9
    lam = th(Sr) / 1e9
    for it in range(100):
        SL = [lag_pick(c["subs"], lam) for c in cyc]
        new = th(SL) / 1e9
        if abs(new - lam) < 1e-18: break
        lam = new
    rho = [s[3] / T(s) for s in Sr]
    terms = [(rho[i] - lam) * (T(SL[i]) - T(Sr[i])) for i in range(n)]
    lo = [rho[i] < lam for i in range(n)]
    r["T1"] = dict(theta_ratio=th(Sr), lam_star=lam * 1e9, gap_pct=(1 - th(Sr) / (lam * 1e9)) * 100,
                   bound_pct=sum(terms) / sum(T(s) for s in Sr) / lam * 100,
                   negative_terms=sum(t < -1e-6 for t in terms),
                   differ_pct=sum(set(a[0]) != set(b[0]) for a, b in zip(SL, Sr)) / n * 100,
                   realized_ratio=th(Sr, 4), realized_lam=th(SL, 4), realized_diff_pct=(th(SL, 4) / th(Sr, 4) - 1) * 100,
                   rho_cv=st.pstdev(rho) / M(rho), dinkelbach_iters=it,
                   low_rho_pct=sum(lo) / n * 100,
                   low_rho_shorter_time_pct=sum(T(SL[i]) < T(Sr[i]) - 1e-6 for i in range(n) if lo[i]) / max(1, sum(lo)) * 100,
                   high_rho_longer_time_pct=sum(T(SL[i]) > T(Sr[i]) + 1e-6 for i in range(n) if not lo[i]) / max(1, n - sum(lo)) * 100)
    grid = [0.25, 0.5, 1, 1.5, 2, 3, 4, 6]
    viol, per = 0, {}
    for c in cyc:
        prev = None
        for g in grid:
            s = ratio_pick(c["subs"], BETA * g); v = s[2] / s[3]
            if prev is not None and v > prev + 1e-9: viol += 1
            prev = v
    for g in (1, 2):
        SS = [ratio_pick(c["subs"], BETA * g) for c in cyc]
        per[g] = dict(F_per_R=M([s[2] / s[3] for s in SS]), nodes=M([s[1] for s in SS]))
    r["T3a"] = dict(violations=viol, grid=grid, beta1=per[1], beta2=per[2])
    ruled = [c["D"] < c["eff"] for c in cyc]
    r["T4"] = dict(drafted_levels_mean=M(c["D"] for c in cyc),
                   drafted_levels_hist={d: sum(c["D"] == d for c in cyc) for d in range(1, 6)},
                   below_5_pct=sum(c["D"] < 5 for c in cyc) / n * 100,
                   rule_stop_before_5_pct=sum(ruled) / n * 100,
                   limit_cycles=sum(c["eff"] < 5 for c in cyc))
    return r

# Usage: python3 theory_checks.py [adaptive-pools.json.gz]; prints the content of theory-checks.json.
res = json.load(gzip.open(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "adaptive-pools.json.gz"))
out = {ds: run(cycles(res, ds)) for ds in ("gsm8k", "summarization")}
print(json.dumps(out, indent=1))

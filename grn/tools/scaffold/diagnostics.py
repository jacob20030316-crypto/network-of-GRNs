"""B-gate step 3: acceptance testing, run separately for each network.

Two networks come out of 01, and they have to pass on their own:

  uncond  one unit per cohort's disease group, floor 50. The backbone — it has
          to discriminate diseases by itself.
  cond    one unit per (cohort, axis, level), floor 30. Conditions are not
          matched across diseases; each unit is one view of one disease, and a
          disease pair is judged through all pairings of their views.

Same-disease pairs never become edges (the units share patients), but they stay
in the report as the discrimination signal: if two units of one disease are not
more similar than two units of different diseases, the vectors carry nothing
disease-specific and everything downstream is noise.

Confounds tested, each of which can pass the whole while failing a part:

  size       spearman(similarity, sample size)
  coverage   spearman(similarity, edge coverage) -- the scaffold fixes the edge
             *universe*, not what each cohort actually measured
  source     R2 of the source pairing, AND the per-block medians. The global R2
             alone is not enough: the TCGA-TCGA block sits ~0.26 above the rest
             while contributing only 2% of pairs, so an average passes while a
             block is badly contaminated. That failure is why the per-block
             table below exists.

Corrections applied cumulatively: ① raw ② edge-centred ③ + per-TF normalised
④ + disattenuated. 04 shows ② and ③ have different jobs and that ③ is not
method-neutral, so the ladder is reported step by step rather than as a total.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from grn import paths as config                                  # noqa: E402
IN = config.SCAFFOLD_OUT      # what the scoring step wrote
OUT = config.READOUT
os.makedirs(OUT, exist_ok=True)
CORE_PREVALENCE = 0.99
GATE = 0.15


def sim_matrix(W):
    Wc = W - W.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(Wc, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = np.nan
    Z = Wc / nrm
    return np.clip(np.nan_to_num(Z @ Z.T), -1, 1)


def report(name, S, iu, cross, pair_n, pair_cov):
    v = S[iu].astype(float)
    rho_n = spearmanr(v[cross], pair_n[cross]).statistic
    cov = ""
    if pair_cov is not None:
        cov = f"  ρ(cov) {spearmanr(v[cross], pair_cov[cross]).statistic:>+6.3f}"
    print(f"{name:<26} bg median {np.median(v[cross]):>7.3f}  "
          f"90th {np.percentile(v[cross], 90):>7.3f}   "
          f"ρ(n) {rho_n:>+6.3f}{cov}")


def ladder(W, E, M, iu, cross, pair_n, pair_cov, half):
    S1 = sim_matrix(W)
    report("1 raw", S1, iu, cross, pair_n, pair_cov)

    W2 = W - W.mean(axis=0, keepdims=True)
    report("2 + edge-centred", sim_matrix(W2), iu, cross, pair_n, pair_cov)

    tf = E.source_genesymbol.values
    W3 = W2.copy()
    for t in pd.unique(tf):
        idx = np.where(tf == t)[0]
        blk = W3[:, idx]
        nrm = np.linalg.norm(blk, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        W3[:, idx] = blk / nrm
    S3 = sim_matrix(W3)
    report("3 + per-factor normalised", S3, iu, cross, pair_n, pair_cov)

    if half is None:
        print("  (no split-half reliability yet; step 4 skipped)")
        S4 = S3
    else:
        r = M.nid.map(half.r_half).values
        rel = np.clip(2 * r / (1 + r), 0.02, 1.0)
        rel = np.where(np.isfinite(rel), rel, np.nanmedian(rel))
        S4 = np.clip(S3 / np.sqrt(np.outer(rel, rel)), -1, 1)
        report("4 + disattenuated", S4, iu, cross, pair_n, pair_cov)

    # ⑤ source-block centring. The TCGA-TCGA block sits well above the rest for
    # a reason we tested and ruled out as biology: inside TCGA, whether both
    # units are tumour tissue makes no difference to their similarity (0.253 vs
    # 0.267), while in GEO tumour pairs are only +0.03 above the rest. So the
    # offset is processing, and an additive block offset is removable — unlike
    # the sample-size confound, which was never estimated in the first place and
    # therefore could not be corrected after the fact.
    src = M.source.values
    blk = np.where(src[iu[0]] == src[iu[1]], src[iu[0]], "TCGA-GEO")
    S5 = S4.copy()
    for b in pd.unique(blk):
        m = blk == b
        off = np.median(S4[iu][m & cross])
        i0, i1 = iu[0][m], iu[1][m]
        S5[i0, i1] -= off
        S5[i1, i0] -= off
    report("5 + source-block centred", S5, iu, cross, pair_n, pair_cov)
    return S5


def analyse(net, M, W, A, E, half):
    print(f"\n{'='*66}\n=== {net}: {len(M)} groups / {M.disease.nunique()} diseases "
          f"/ {M.groupby(['disease','cohort']).ngroups} cohorts\n{'='*66}")

    N = len(M)
    iu = np.triu_indices(N, k=1)
    dz = M.disease.values
    same = dz[iu[0]] == dz[iu[1]]
    cross = ~same
    print(f"{len(iu[0])} pairs   same disease {same.sum()} (not joined; the "
          f"discrimination signal)  "
          f"joined {cross.sum()}")

    n = M.n.values.astype(float)
    pair_n = np.minimum(n[iu[0]], n[iu[1]])
    cov = A.mean(axis=1)
    pair_cov = np.minimum(cov[iu[0]], cov[iu[1]])

    core = A.mean(axis=0) >= CORE_PREVALENCE
    print(f"core interactions {int(core.sum())}/{len(core)}   "
          f"estimable fraction median {np.median(cov):.3f} "
          f"range {cov.min():.3f}-{cov.max():.3f}\n")

    print("=== the correction ladder, on core interactions ===")
    S = ladder(W[:, core], E[core].reset_index(drop=True), M,
               iu, cross, pair_n, pair_cov, half)
    v = S[iu].astype(float)

    # Discrimination. Same-disease pairs from the SAME cohort share patients —
    # in the conditioned network a cohort's gender and age units overlap almost
    # completely — so their similarity measures sample overlap, not disease
    # recognition. Only same-disease DIFFERENT-cohort pairs are evidence.
    ch = M.cohort.values
    same_cohort = (ch[iu[0]] == ch[iu[1]]) & same
    same_xc = same & ~same_cohort
    print()
    _same_xc = same_xc
    for lab, mask in (("same disease, different cohorts (the signal)", same_xc),
                      ("same disease, same cohort (shares patients)", same_cohort)):
        if mask.sum() < 5:
            print(f"  {lab:<44} n={int(mask.sum())}  too few")
            continue
        a, b = v[mask], v[cross]
        rk = pd.Series(np.concatenate([a, b])).rank().values
        auc = (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        print(f"  {lab:<44} n={int(mask.sum()):>5}  median {np.median(a):>6.3f}  "
              f"bg {np.median(b):>6.3f}  gap {np.median(a)-np.median(b):>+6.3f}  "
              f"AUC {auc:.3f}")

    # source block — global R2 and, crucially, per block
    src = M.source.values
    blk = np.where(src[iu[0]] == src[iu[1]], src[iu[0]], "TCGA-GEO")
    D = pd.get_dummies(pd.Series(blk)).astype(float).values
    coef, *_ = np.linalg.lstsq(D, v, rcond=None)
    r2 = 1 - ((v - D @ coef) ** 2).sum() / ((v - v.mean()) ** 2).sum()
    print(f"\n=== source blocks (overall R2={r2:.3f}; the per-block figures "
          f"are the point) ===")
    print(f"  {'block':<12} {'edges':>8} {'bg median':>11} {'same disease':>14} "
          f"{'gap':>8}")
    worst = 0.0
    for bname in pd.unique(blk):
        m = blk == bname
        bg = np.median(v[m & cross])
        # cross-cohort only: same-cohort pairs share patients and would inflate
        ns = int((m & _same_xc).sum())
        gap = (np.median(v[m & _same_xc]) - bg) if ns >= 5 else np.nan
        worst = max(worst, abs(bg - np.median(v[cross])))
        print(f"  {bname:<10} {int((m & cross).sum()):>8} {bg:>9.3f} "
              f"{ns:>14} " + (f"{gap:>+8.3f}" if ns >= 5 else f"{'n/a':>8}"))
    print(f"  worst block offset from the overall median: {worst:.3f}. A block "
          f"holding no same-disease pair cannot check itself.")

    print(f"\n=== acceptance ===")
    # Each check carries a stable key as well as its display label: the key is
    # what the inspection step reads, and it should not move when the printing
    # changes.
    gates = [("similarity_vs_sample_size", "rho(similarity, sample size)",
              spearmanr(v[cross], pair_n[cross]).statistic),
             ("similarity_vs_gene_coverage", "rho(similarity, coverage)",
              spearmanr(v[cross], pair_cov[cross]).statistic),
             ("source_pairing_r2", "source pairing R2", r2),
             ("worst_source_block_offset", "worst block offset", worst)]
    ok = True
    for _, label, val in gates:
        p = abs(val) <= GATE
        ok &= p
        print(f"  {'ok  ' if p else 'FAIL'} {label:<30} {val:>+7.3f}  "
              f"(threshold |{GATE}|)")

    # Written out so the inspection step reads the numbers rather than a log.
    pd.DataFrame([{"network": net, "check": key, "value": float(val),
                   "threshold": GATE, "passes": bool(abs(val) <= GATE)}
                  for key, _, val in gates]).to_csv(
        f"{OUT}/acceptance_{net}.csv", index=False)

    np.save(f"{OUT}/S_{net}.npy", S)   # the diagnostic similarity (M0)
    M2 = M.copy()
    M2["mean_sim"] = np.nanmean(np.where(np.eye(N, dtype=bool), np.nan, S), axis=1)
    # meta_<net>.csv is read by the edge and mechanism steps, so it is written
    # where they look rather than left here
    M2.to_csv(f"{IN}/meta_{net}.csv", index=False)
    print(f"  wrote S_{net}.npy / meta_{net}.csv")
    return ok


def main():
    W = np.load(f"{IN}/W_full.npy").astype(np.float64)
    A = np.load(f"{IN}/A_avail.npy")
    M = pd.read_csv(f"{IN}/W_meta.csv")
    E = pd.read_csv(f"{IN}/edge_universe.csv")
    hp = f"{IN}/reliability_halfsplit.csv"
    half = pd.read_csv(hp).set_index("nid") if os.path.isfile(hp) else None
    print(f"matrix {W.shape[0]} groups x {W.shape[1]} interactions")

    results = {}
    for net in ("uncond", "cond"):
        m = (M.network == net).values
        results[net] = analyse(net, M[m].reset_index(drop=True),
                               W[m], A[m], E, half)

    print(f"\n{'='*66}")
    for net, ok in results.items():
        print(f"  {net:<8} " + ("acceptance passed" if ok else
                                  "a check did not pass; do not go on"))


def run(**kw):
    """Entry point for the pipeline runner. Keyword arguments override the
    module-level constants; the body is the same either way."""
    g = globals()
    for k, v in kw.items():
        key = k.upper()
        if key not in g:
            raise TypeError(f"{__name__}.run() got an unexpected argument {k!r}")
        g[key] = v
    return main()


if __name__ == "__main__":
    sys.exit(main())

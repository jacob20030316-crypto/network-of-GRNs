"""Six similarity variants x three ladder settings, all through one gate.

The method implementations live in `CollecTRI/similarity.py` and are imported,
not copied. They used to be duplicated here, and a fix applied to one copy
silently missed the other — which is how the inflated conditioned-network AUC
described below survived for a while.

The three legacy methods come from the earlier de novo work
(`GRN_Program/Conditional_GRN_analysis/grn_similarity.py`). They were written
against SCENIC regulon matrices, where every cohort has its own TF set and its
own target set, so all three lean on set overlap. On the scaffold the edge
universe is fixed and shared, which changes what each one measures — some for
the better, one into degeneracy:

  M0  global cosine       the incumbent. Edge-centred cosine over all edges.
  M1  hub-TF Jaccard      weighted degree per TF -> rank -> top-X% -> Jaccard.
  M3  S / S2 / S3         mean over TFs of the two units' target-profile
                          similarity. The legacy w1*Jaccard(TF sets) term is
                          identically 1 on the scaffold and is dropped.
  M3  S3-signed           S3 without the legacy `.abs()`. Scaffold weights are
                          signed correlations, and taking absolute values
                          collapses the low-dimensional per-TF vectors onto one
                          direction.

Two things the gate does that a naive one would not:

**Discrimination uses same-disease CROSS-COHORT pairs only.** Same-disease
same-cohort pairs share patients — in the conditioned network a cohort's gender
and age units overlap almost completely — so counting them measures sample
overlap, not disease recognition, and pushes the AUC to ~0.95.

**AUC is also reported against a tissue-matched background.** The headline AUC
compares same-disease pairs against all cross-disease pairs, most of which are
cross-tissue, so it credits the method for recognising tissue. The
tissue-matched column is the honest number.

    python 02_method_comparison.py --network cond --core
    python 02_method_comparison.py --network uncond --core --ladder centre
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from grn import paths as config                                  # noqa: E402
from grn.tools.scaffold import similarity as S                   # noqa: E402

IN = config.SCAFFOLD_OUT
OUT = config.READOUT
os.makedirs(OUT, exist_ok=True)
CORE_PREVALENCE = 0.99


def gate(name, Sm, M, iu, same_xc, cross, pair_n, pair_cov, pair_tissue=None):
    """The acceptance criteria, identical for every method.

    `same_xc` is same-disease cross-cohort; `cross` is different-disease. Pairs
    that are same-disease same-cohort belong to neither and are excluded from
    both the signal and the background.
    """
    v = Sm[iu].astype(float)
    bg = np.median(v[cross])
    same = np.median(v[same_xc]) if same_xc.sum() else np.nan

    auc = np.nan
    if same_xc.sum() >= 5:
        a, b = v[same_xc], v[cross]
        rk = pd.Series(np.concatenate([a, b])).rank().values
        auc = (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))

    rho_n = spearmanr(v[cross], pair_n[cross]).statistic
    rho_c = (spearmanr(v[cross], pair_cov[cross]).statistic
             if pair_cov is not None else np.nan)

    src = M.source.values
    blk = np.where(src[iu[0]] == src[iu[1]], src[iu[0]], "TCGA-GEO")
    D = pd.get_dummies(pd.Series(blk)).astype(float).values
    coef, *_ = np.linalg.lstsq(D, v, rcond=None)
    r2 = 1 - ((v - D @ coef) ** 2).sum() / ((v - v.mean()) ** 2).sum()

    tis_gap = tis_r2 = auc_t = np.nan
    if pair_tissue is not None:
        st = pair_tissue["same_tissue"]
        if (cross & st).sum() and (cross & ~st).sum():
            tis_gap = np.median(v[cross & st]) - np.median(v[cross & ~st])
        Dt = pd.get_dummies(pd.Series(pair_tissue["stratum"])).astype(float).values
        ct, *_ = np.linalg.lstsq(Dt, v, rcond=None)
        tis_r2 = 1 - ((v - Dt @ ct) ** 2).sum() / ((v - v.mean()) ** 2).sum()
        b2 = v[cross & st]
        if same_xc.sum() >= 5 and len(b2) >= 20:
            a = v[same_xc]
            rk = pd.Series(np.concatenate([a, b2])).rank().values
            auc_t = ((rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2)
                     / (len(a) * len(b2)))

    return dict(method=name, background=bg, same_disease=same,
                gap=same - bg, auc=auc, auc_tissue_matched=auc_t,
                spearman_n=rho_n, spearman_cov=rho_c, source_r2=r2,
                same_tissue_gap=tis_gap, tissue_r2=tis_r2,
                n_same_xc=int(same_xc.sum()), n_cross=int(cross.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=["uncond", "cond"], default="uncond")
    ap.add_argument("--core", action="store_true")
    ap.add_argument("--ladder", choices=["none", "centre", "full"], default="full")
    ap.add_argument("--hub-percent", type=float, default=S.HUB_PERCENT)
    args = ap.parse_args(argv)

    W = np.load(f"{IN}/W_full.npy").astype(np.float64)
    M = pd.read_csv(f"{IN}/W_meta.csv")
    E = pd.read_csv(f"{IN}/edge_universe.csv")
    ap_ = f"{IN}/A_avail.npy"
    A_all = np.load(ap_) if os.path.isfile(ap_) else None
    keep = (M.network == args.network).values
    W, M = W[keep], M[keep].reset_index(drop=True)
    if A_all is not None:
        A_all = A_all[keep]
    print(f"network = {args.network}   correction ladder = {args.ladder}")

    N = len(M)
    iu = np.triu_indices(N, k=1)
    n = M.n.values.astype(float)
    pair_n = np.minimum(n[iu[0]], n[iu[1]])

    pair_cov = None
    if A_all is not None:
        cov = A_all.mean(axis=1)
        pair_cov = np.minimum(cov[iu[0]], cov[iu[1]])
        if args.core:
            core = A_all.mean(axis=0) >= CORE_PREVALENCE
            W, E = W[:, core], E[core].reset_index(drop=True)
            print(f"core interactions {int(core.sum())} / {len(core)}")
    elif args.core:
        raise SystemExit("--core needs A_avail.npy")

    dz, ch = M.disease.values, M.cohort.values
    same_d = dz[iu[0]] == dz[iu[1]]
    same_c = ch[iu[0]] == ch[iu[1]]
    same_xc = same_d & ~same_c          # the signal: same disease, different cohorts
    cross = ~same_d                     # the pairs that become edges
    print(f"matrix {W.shape[0]} groups x {W.shape[1]} interactions")
    print(f"same disease, different cohorts {int(same_xc.sum())} (the signal)  "
          f"same disease, same cohort {int((same_d & same_c).sum())} "
          f"(shares patients; counted on neither side)  "
          f"different diseases {int(cross.sum())}")

    pair_tissue = None
    if os.path.isfile(config.COHORT_TISSUE):
        T = pd.read_csv(config.COHORT_TISSUE).set_index(["disease", "cohort"])
        tc = np.array([T.tissue_class.get((d, c), "unknown")
                       for d, c in zip(M.disease, M.cohort)])
        a, b = tc[iu[0]], tc[iu[1]]
        pair_tissue = {
            "same_tissue": a == b,
            "stratum": np.array(["|".join(sorted((x, y))) for x, y in zip(a, b)]),
        }
        print(f"tissue labels: {len(pd.unique(tc))} classes, "
              f"cross-disease same-tissue pairs "
              f"{int((pair_tissue['same_tissue'] & cross).sum())}\n")
    else:
        print(f"  ! {config.COHORT_TISSUE} is absent; tissue columns skipped\n")

    if args.ladder != "none":
        W = S.apply_ladder(W, E, args.ladder)
        print("applied to every method: " + ("edge-centring" if args.ladder == "centre"
                            else "edge-centring and per-factor normalisation")
              + "; the similarity is then disattenuated and source-block centred\n")

    tfs = pd.unique(E.source_genesymbol.values)
    tf_pos = {t: i for i, t in enumerate(tfs)}
    tf_idx = [[] for _ in tfs]
    for j, t in enumerate(E.source_genesymbol.values):
        tf_idx[tf_pos[t]].append(j)
    tf_idx = [np.array(x, dtype=int) for x in tf_idx]
    n_tf = len(tfs)

    methods = [
        ("M0", "M0 global cosine", lambda: S.m0_global_cosine(W, tf_idx)),
        ("M1", f"M1 hub-TF Jaccard top{args.hub_percent:g}%",
         lambda: S.m1_hub_jaccard(W, tf_idx, args.hub_percent, n_tf)),
        ("M3_S", "M3 S  (wJaccard, raw)",
         lambda: S.m3_s_regulon(W, tf_idx, n_tf, "none", "jaccard")),
        ("M3_S2", "M3 S2 (wJaccard, L1)",
         lambda: S.m3_s_regulon(W, tf_idx, n_tf, "l1", "jaccard")),
        ("M3_S3", "M3 S3 (cosine, L1)",
         lambda: S.m3_s_regulon(W, tf_idx, n_tf, "l1", "cosine")),
        ("M3_S3signed", "M3 S3-signed (cosine, L1, sign kept)",
         lambda: S.m3_s_regulon(W, tf_idx, n_tf, "l1", "cosine", use_abs=False)),
    ]

    suffix = (f"_{args.network}" + ("_core" if args.core else "")
              + ("" if args.ladder == "full" else f"_{args.ladder}"))
    rows = []
    for key, name, fn in methods:
        t0 = time.time()
        Sm = fn()
        if args.ladder != "none":
            Sm = S.disattenuate(Sm, M, IN)
            Sm = S.block_centre(Sm, M, iu, cross)
        rows.append(gate(name, Sm, M, iu, same_xc, cross, pair_n, pair_cov,
                         pair_tissue))
        np.save(f"{IN}/S_{key}{suffix}.npy", Sm)
        print(f"  {name:<38} {time.time() - t0:>5.1f}s")

    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/similarity_gate{suffix}.csv", index=False)

    print(f"\n=== one standard for all (network {args.network}, "
          f"ladder {args.ladder}) ===")
    print(f"{'method':<38} {'bg':>7} {'same':>7} {'gap':>7} {'AUC':>6} "
          f"{'AUC tissue':>12} {'rho(n)':>8} {'rho(cov)':>9} {'src R2':>8}")
    for _, r in R.iterrows():
        print(f"{r.method:<38} {r.background:>7.3f} {r.same_disease:>7.3f} "
              f"{r.gap:>+7.3f} {r.auc:>6.3f} {r.auc_tissue_matched:>11.3f} "
              f"{r.spearman_n:>+7.3f} {r.spearman_cov:>+7.3f} {r.source_r2:>7.3f}")

    print("\nReading it: any of rho(n), rho(cov) or src R2 beyond |0.15| "
          "disqualifies the method, however high its AUC.")
    print("Among those that pass, compare the tissue-matched AUC: the "
          "unmatched column credits a method for recognising tissue.")
    print(f"\nwrote similarity_gate{suffix}.csv")


def run(**kw):
    """Entry point for the pipeline runner. Keyword arguments override the
    command-line defaults; the body is the same either way."""
    argv = []
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())

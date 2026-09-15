"""Every disease pair, every shared regulon, scored against that factor's own
background.

Two changes from `06_pair_network.py`. It computed the mechanism layer only for
the forty pairs with the smallest disease-pair permutation p, a screen at the
edge level that decides what the factor level is allowed to see and answers a
different question from the one asked here. And it applied a percentile gate
before reporting, so the quantity used to select was the quantity used to test.
Here every pair is scored, every transcription factor present in both diseases
is tested, and the correction alone decides what survives.

The null for a factor is its own distribution of target-Jaccard over unrelated
disease pairs. A regulon of two targets lands on 0.5 or 1.0 by arithmetic and a
factor whose targets are stable everywhere overlaps everywhere, so neither the
size of a regulon nor the ubiquity of a programme earns significance; a factor
has to beat the line it sets for itself. The background holds about a thousand
values, fixed by the forty-eight units available rather than by computation, so
the smallest attainable p is near 1e-3 and the correction is applied within a
pair rather than across all of them.

    python 08_pair_factors_all.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from grn import paths as config                                  # noqa: E402
from grn.tools.denovo import pair_network as pn                  # noqa: E402

OUT = config.DENOVO_OUT

MIN_SURVIVORS, Q = 3, 0.05
MIN_SHARED, MIN_BG = 5, 500   # independent of the statistic being tested


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def main():
    sets, M = pn.load_regulons()
    uids = list(M.index)
    ov = config.sample_overlap(np.array(uids))
    dz = M.disease.values
    n = len(uids)
    iu = np.triu_indices(n, 1)
    cross = (dz[iu[0]] != dz[iu[1]]) & ~ov[iu[0], iu[1]]
    print(f"{n} nodes / {M.disease.nunique()} diseases / "
          f"{cross.sum()} cross-disease node pairs",
          flush=True)

    bg = pn.tf_background(sets, M, uids, iu, cross)
    print(f"factors with a background: {len(bg)}, median background size "
          f"{int(np.median([len(v) for v in bg.values()]))}", flush=True)

    byd = defaultdict(list)
    for i, u in enumerate(uids):
        byd[dz[i]].append(i)
    ds = sorted(byd)

    rows = []
    for x in range(len(ds)):
        for y in range(x + 1, len(ds)):
            vps = [(i, j) for i in byd[ds[x]] for j in byd[ds[y]]
                   if not ov[i, j]]
            if not vps:
                continue
            acc, tgt = defaultdict(list), defaultdict(set)
            for i, j in vps:
                A, B = sets[uids[i]], sets[uids[j]]
                for t, j_, _ in pn._per_tf(A, B):
                    if t not in bg:
                        continue
                    acc[t].append((j_, A[t][0], B[t][0]))
                    tgt[t] |= (A[t][1] & B[t][1])
            for t, v in acc.items():
                med = float(np.median([q[0] for q in v]))
                rank = int(np.searchsorted(bg[t], med))
                rows.append(dict(
                    pair=f"{ds[x]}||{ds[y]}", disease_a=ds[x], disease_b=ds[y],
                    tf=t, n_unit_pairs=len(vps), n_present=len(v),
                    jaccard_median=med, bg_n=len(bg[t]),
                    bg_pctl=rank / len(bg[t]),
                    p=(len(bg[t]) - rank + 1) / (len(bg[t]) + 1),
                    n_shared_targets=len(tgt[t]),
                    nes_a=float(np.median([q[1] for q in v])),
                    nes_b=float(np.median([q[2] for q in v])),
                    shared_targets="|".join(sorted(tgt[t])[:40])))
    D = pd.DataFrame(rows)
    # A factor whose two regulons share nothing has an overlap of zero and
    # nothing to test; carrying such tests only divides the correction. Both
    # filters are properties of the data rather than of the quantity being
    # tested, so neither presupposes the answer, unlike the percentile gate
    # that 06_pair_network.py applied before reporting.
    D["tested"] = (D.n_shared_targets >= MIN_SHARED) & (D.bg_n >= MIN_BG)
    D["q_within"] = np.nan
    T = D[D.tested]
    D.loc[D.tested, "q_within"] = (
        T.groupby("pair").p.transform(lambda v: bh(v.values)))
    D.loc[D.tested, "q_global"] = bh(T.p.values)
    k = D[D.tested].groupby("pair").apply(
        lambda g: (g.q_within < Q).sum(), include_groups=False)
    keep = set(k[k >= MIN_SURVIVORS].index)
    D["carried"] = D.pair.isin(keep) & (D.q_within < Q)
    # a factor shared by many pairs is annotated, not removed
    surv = D[D.q_within < Q]
    D["generic_frac"] = (D.tf.map(surv.tf.value_counts()).fillna(0)
                         / max(D.pair.nunique(), 1))
    D.sort_values(["pair", "p"]).to_csv(
        f"{OUT}/denovo_pair_factors_all.csv", index=False)

    print(f"\n{D.pair.nunique()} disease pairs, {len(D)} factors present on "
          f"both sides, "
          f"{int(D.tested.sum())} of them tested"
          f" (shared targets >= {MIN_SHARED}, background >= {MIN_BG})")
    print(f"smallest attainable p = {D.p.min():.2e}, set by the background size")
    print(f"within-pair BH q<{Q}: {(D.q_within < Q).sum()} surviving, "
          f"{(k > 0).sum()} pairs keeping at least one")
    print(f"  surviving per pair: median {k.median():.0f}  quartiles "
          f"{k.quantile(.25):.0f}-{k.quantile(.75):.0f}  max {k.max()}")
    for m in (3, 5, 8):
        print(f"  pairs keeping >= {m} factors: {(k >= m).sum()}")
    print(f"global BH q<{Q}: {(D.q_global < Q).sum()} surviving")
    print(f"\ncarried to interpretation: {len(keep)} disease pairs, "
          f"{int(D.carried.sum())} factors")
    print("\nthe 12 pairs with the most survivors:")
    print(k.sort_values(ascending=False).head(12).to_string())
    print(f"\nwrote {OUT}/denovo_pair_factors_all.csv")
    return 0


def run(**kw):
    """Entry point for the pipeline runner. This stage takes no parameters."""
    if kw:
        raise TypeError(f"{__name__}.run() takes no arguments, got {sorted(kw)}")
    return main()


if __name__ == "__main__":
    sys.exit(main())

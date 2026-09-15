"""Build the edge tables. Construction only — no validation, no tiering.

One rule, applied uniformly to every pair of units:

    same disease, OR the two units share any sample   -> no edge
    otherwise                                          -> edge = corrected similarity

The sample-overlap clause is not redundant with the disease clause. Disease
labels are ours, and several of them denote overlapping sample sets: TCGA's
GBMLGG cohort is the union of GBM and LGG, `Endometrioid_Cancer` and
`Uterine_Corpus_Endometrial_Carcinoma` both resolve to TCGA UCEC, and
`Uterine_Carcinosarcoma` sits inside `Sarcoma`. Six disease pairs overlap, three
of them completely. Without this clause those pairs top the strength ranking —
they were measuring sample identity, not shared regulation.

Nothing is dropped or down-weighted for being hard to verify. Whether an edge
can be replicated is a property of what data exists, not of the edge, and
mixing that judgement into the graph would make the construction depend on
cohort availability. Verification lives in `07_validate.py` and is reported
separately, on the subset where it is possible.

The one thing that does belong here is calibration: a raw similarity is not
interpretable until we know what a similarity of 0.2 means against chance. Two
nulls, because the graph is read at two levels:

  unit-pair    the empirical background of all cross-disease pairs. After the
               correction ladder this is centred at 0, so an edge's percentile
               in that distribution is its calibrated strength.

  disease-pair permutation of disease labels over units, preserving how many
               units each disease has and which source each unit came from.
               This is the null that matters for "are these two diseases
               related", because it asks whether the observed aggregate could
               arise from units being shuffled among diseases of the same size.

Outputs edges_<net>.csv (unit level) and disease_pairs_<net>.csv.

    python 06_edges.py --net uncond
    python 06_edges.py --net cond --perms 200
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from grn import paths as config                                  # noqa: E402
OUT = config.SCAFFOLD_OUT
os.makedirs(OUT, exist_ok=True)
METHOD = "M3_S3signed"          # chosen in 04
SEED = 0


def overlapping_units(M):
    """Boolean unit x unit: do these two units share any sample?

    Built from the sample lists in analysis_units.csv via a sample -> units
    index, so it is linear in the number of samples rather than quadratic in
    units.
    """
    U = pd.read_csv(config.ANALYSIS_UNITS).set_index("unit_id")
    idx = {nid: i for i, nid in enumerate(M.nid)}
    from collections import defaultdict
    owners = defaultdict(list)
    for nid in M.nid:
        if nid not in U.index:
            continue
        for s in str(U.loc[nid, "samples"]).split("|"):
            owners[s].append(idx[nid])
    ov = np.zeros((len(M), len(M)), dtype=bool)
    for us in owners.values():
        if len(us) > 1:
            for a in range(len(us)):
                for b in range(a + 1, len(us)):
                    ov[us[a], us[b]] = ov[us[b], us[a]] = True
    return ov


def load(net):
    S = np.load(f"{OUT}/S_{METHOD}_{net}_core.npy")
    M = pd.read_csv(f"{OUT}/meta_{net}.csv")
    T = pd.read_csv(config.COHORT_TISSUE).set_index(["disease", "cohort"])
    M["tissue_class"] = [T.tissue_class.get((d, c), "unknown")
                         for d, c in zip(M.disease, M.cohort)]
    if len(M) != S.shape[0]:
        raise SystemExit(f"meta_{net}.csv has {len(M)} rows but S is {S.shape}; "
                         f"the steps are out of step -- rerun from vectors")
    return S, M


def disease_pair_table(S, M, iu, cross, rng, perms):
    """Aggregate to disease pairs and calibrate against label permutation.

    The aggregate is the median over the block of unit pairs. It is a reporting
    step, not part of the graph: the graph's nodes stay at unit level, and this
    table is how a disease-level claim gets made from it.
    """
    dz = M.disease.values
    a, b = dz[iu[0]], dz[iu[1]]
    key = np.where(a < b, a, b) + "||" + np.where(a < b, b, a)
    v = S[iu].astype(float)

    df = pd.DataFrame({"pair": key[cross], "w": v[cross],
                       "ca": M.cohort.values[iu[0]][cross],
                       "cb": M.cohort.values[iu[1]][cross]})
    obs = df.groupby("pair").agg(
        strength=("w", "median"), n_unit_pairs=("w", "size"),
        q25=("w", lambda x: x.quantile(.25)),
        q75=("w", lambda x: x.quantile(.75)))
    obs["n_cohort_pairs"] = df.groupby("pair").apply(
        lambda g: len({(x, y) for x, y in zip(g.ca, g.cb)}))

    # Permutation: shuffle disease labels among units of the same source, so
    # each disease keeps its unit count and the source composition is preserved.
    #
    # The permuted medians are POOLED rather than compared pair-by-pair. With
    # 2211 pairs, BH at 0.05 needs p ~ 2e-5 for the top pair, which a per-pair
    # count over a few hundred permutations cannot resolve (its floor is
    # 1/(perms+1)). Under the null, pairs whose blocks are the same size are
    # exchangeable, so their permuted medians estimate the same distribution:
    # pooling within a block-size stratum turns `perms` draws into
    # perms x (pairs in stratum), which does resolve that range.
    print(f"  {perms} permutations, stratified by source and tissue, "
          f"pooled by block size ...",
          flush=True)
    # Preserving source alone is not enough. Measured on the unconditioned
    # network: 11.9% of all disease pairs are same-tissue, but 11 of the top 20
    # by strength were blood-blood. Blood units are systematically more similar
    # to each other, so against a null that mixes tissues they always win and
    # the ranking becomes a ranking of tissue. Shuffling within tissue makes a
    # blood-blood pair compete with other blood-blood pairs.
    src = (M.source.astype(str) + "|" + M.tissue_class.astype(str)).values
    sizes = obs.n_unit_pairs.values
    # strata: log-spaced block sizes, so a pair is compared to pairs with a
    # comparable number of unit pairs rather than to the whole graph
    edges_b = np.unique(np.quantile(sizes, np.linspace(0, 1, 9)))
    stratum = np.clip(np.searchsorted(edges_b, sizes, side="right") - 1,
                      0, len(edges_b) - 1)
    pool = {k: [] for k in np.unique(stratum)}
    order = {p: i for i, p in enumerate(obs.index)}

    for p in range(perms):
        perm = np.arange(len(M))
        for s in np.unique(src):
            m = np.where(src == s)[0]
            perm[m] = rng.permutation(m)
        pz = dz[perm]
        pa, pb = pz[iu[0]], pz[iu[1]]
        # `& cross` is essential and was missing. The observed medians are taken
        # over the unit pairs that BECOME EDGES; the permuted medians have to be
        # taken over the same population or the two are not comparable. Without
        # it the null also swallows the 6,984 unit pairs the edge rules exclude —
        # same disease, or overlapping patients — whose similarity has a median
        # of 0.721 against 0.000 for real edges. That lifts the whole null and
        # the test becomes so conservative that nothing can pass: measured, this
        # one term is the difference between 0 and 6 pairs at BH<0.05.
        pc = (pa != pb) & cross
        pk = np.where(pa < pb, pa, pb) + "||" + np.where(pa < pb, pb, pa)
        g = pd.DataFrame({"pair": pk[pc], "w": v[pc]}).groupby("pair").agg(
            med=("w", "median"), k=("w", "size"))
        gs = np.clip(np.searchsorted(edges_b, g.k.values, side="right") - 1,
                     0, len(edges_b) - 1)
        for k in np.unique(gs):
            pool[k].append(g.med.values[gs == k])
        if (p + 1) % 50 == 0:
            print(f"    {p+1}/{perms}", flush=True)

    pv = np.ones(len(obs))
    for k in np.unique(stratum):
        null = np.concatenate(pool[k]) if pool[k] else np.array([0.0])
        null.sort()
        m = stratum == k
        # P(null >= observed)
        pv[m] = (len(null) - np.searchsorted(null, obs.strength.values[m],
                                             side="left") + 1) / (len(null) + 1)
    obs["p_perm"] = pv
    obs["n_null"] = [len(np.concatenate(pool[k])) if pool[k] else 0
                     for k in stratum]
    obs["q_bh"] = bh(obs.p_perm.values)
    return obs.reset_index()


def bh(p):
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=["uncond", "cond"], required=True)
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args(argv)

    S, M = load(args.net)
    N = len(M)
    iu = np.triu_indices(N, k=1)
    dz = M.disease.values
    same = dz[iu[0]] == dz[iu[1]]
    ov = overlapping_units(M)[iu]
    blocked = same | ov
    cross = ~blocked
    v = S[iu].astype(float)
    print(f"{args.net}: {N} groups / {M.disease.nunique()} diseases")
    print(f"  not joined: same disease {int(same.sum())}  "
          f"different diseases but sharing samples "
          f"{int((ov & ~same).sum())}   joined {int(cross.sum())}")
    if (ov & ~same).sum():
        bad = pd.DataFrame({"a": dz[iu[0]][ov & ~same],
                            "b": dz[iu[1]][ov & ~same]})
        pr = (bad.a + " ~ " + bad.b).value_counts()
        print("  disease pairs blocked by shared samples:")
        for k, c in pr.items():
            print(f"     {k}  ({c} group pairs)")

    # unit-level edge table, calibrated against the cross-disease background
    bg = v[cross]
    pct = pd.Series(bg).rank(pct=True).values
    E = pd.DataFrame({
        "unit_a": M.nid.values[iu[0]][cross],
        "unit_b": M.nid.values[iu[1]][cross],
        "disease_a": dz[iu[0]][cross], "disease_b": dz[iu[1]][cross],
        "cohort_a": M.cohort.values[iu[0]][cross],
        "cohort_b": M.cohort.values[iu[1]][cross],
        "axis_a": M.axis.values[iu[0]][cross], "axis_b": M.axis.values[iu[1]][cross],
        "source_a": M.source.values[iu[0]][cross],
        "source_b": M.source.values[iu[1]][cross],
        "weight": bg, "bg_percentile": pct,
    })
    E.to_csv(f"{OUT}/edges_{args.net}.csv", index=False)
    print(f"  wrote edges_{args.net}.csv  ({len(E)} edges)")
    print(f"  edge weight: median {np.median(bg):.3f}  "
          f"90th {np.percentile(bg,90):.3f}  "
          f"99th {np.percentile(bg,99):.3f}  max {bg.max():.3f}")

    D = disease_pair_table(S, M, iu, cross, np.random.default_rng(SEED),
                           args.perms)
    D.to_csv(f"{OUT}/disease_pairs_{args.net}.csv", index=False)
    print(f"\n  wrote disease_pairs_{args.net}.csv  ({len(D)} disease pairs)")
    sig = D[D.q_bh < 0.05].sort_values("strength", ascending=False)
    print(f"  disease pairs at FDR<0.05 under permutation: {len(sig)} / {len(D)}")
    print(f"\n  the 15 strongest:")
    print(f"  {'disease pair':<58} {'strength':>9} {'groups':>7} "
          f"{'cohorts':>8} {'q':>8}")
    for _, r in sig.head(15).iterrows():
        print(f"  {r.pair[:56]:<58} {r.strength:>7.3f} {r.n_unit_pairs:>6} "
              f"{r.n_cohort_pairs:>6} {r.q_bh:>8.4f}")


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

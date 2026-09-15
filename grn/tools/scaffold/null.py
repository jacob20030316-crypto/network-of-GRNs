"""Mechanism-layer calibration: is a pair's read-out better than a shuffled one?

R1 calibrates an EDGE by shuffling disease labels within source x tissue strata
and asking where the observed edge weight falls. This does the same thing one
layer down, for the two quantities the mechanism read-out actually rests on:

  strength    the median recurrence of the twelve factors the pair would report
  agreement   the split-half reproducibility of that ranking

Neither is calibrated by the edge-level p: measured on this network, the two
layers are almost uncorrelated (Spearman -0.08), so a pair can carry a strong,
well-calibrated edge and still have nothing that recurs.

**Why the null has to be matched on the SHAPE of a pair, not its size.** Both
quantities depend on how much genuine variation a pair contains, and that is
governed by how many patient groups each disease contributes, not by the number
of combinations, which is their product. A pair of 1 x 26 has twenty-six
combinations and no variation at all on one side; a pair of 4 x 7 has fewer
combinations and variation on both. Measured on this network, the smaller
side's group count predicts the strength statistic at least as well as the
combination count (Spearman -0.73 against -0.71), and binning on combinations
alone puts a 4 x 7 pair in the same stratum as a 1 x 20 one, whose strength is
inflated by extreme-value selection over 837 factors. Shuffled pairs are
therefore binned on both, and each real pair is compared only against shuffled
pairs of comparable shape.

**Why both diseases must contribute at least MIN_GROUPS patient groups.** The
claim being tested is that a factor recurs across patient subgroups of both
diseases. A disease represented by a single group cannot support it: every
combination reuses the same network on that side. This is a requirement of the
proposition, not a sample-size preference, and it removes 22% of pairs.

**Why a pair whose tissue-matched background cannot be estimated is dropped.**
Recurrence counts how often a factor clears the 90th percentile of unrelated
group pairs sharing the tissue pairing under test, so the threshold is specific
to those tissues. Rare pairings, such as anything involving bone, uvea, adipose
or thymus, do not contain enough unrelated group pairs for that percentile to be
estimated, and the earlier version silently fell back to a background pooled
over all tissue pairings. That background is easier to clear, and measured here
it showed: pairs on the fallback background passed both criteria 9.8% of the
time against 2.3% for pairs with a proper one, and 43 of 59 passing pairs were
fallbacks. The bar has to be specific to the tissues or the statistic means
something different from pair to pair, so a pair now enters only if at least
BG_MIN unrelated group pairs of its tissue pairing are available.

**Why the bar is chosen per combination and not per disease pair.** Six diseases
contribute patient groups from more than one tissue, because their cohorts were
annotated differently. Taking the tissue of a disease's first group and applying
that bar to all of its combinations judges the rest against the wrong
background, and where the wrong background is the looser one the pair passes for
that reason alone: measured on the earlier version, the 11 pairs involving a
multi-tissue disease passed at 100% against 29% elsewhere. Each combination is
therefore scored against the background of its own two tissues, so recurrence
becomes the fraction of combinations in which the factor cleared that
combination's own bar. Combinations whose tissue pairing has no estimable
background are dropped, and the pair with them if too few remain.

**Why the split-half is group-disjoint here too.** The two halves are formed
from disjoint patient groups, as in the ablation. A random split leaves the same
groups on both sides and inflates the agreement; using one definition in the
ablation and another here would make the word mean two things.

**What group-disjoint does NOT mean.** The halves share no patient GROUP; they
do share patients. A cohort's clinical axes re-partition the same people, so
`sex=male` and `age=o60` of one cohort are different groups drawn from one
sample set, and nothing here stops them landing on opposite sides. Measured on
this network, no draw for any reported pair is patient-disjoint. A split that
were would have to keep whole components of the sample-overlap graph together,
and 40 of the 66 diseases are a single component -- every TCGA disease, and
every disease carried by one submission -- so for most pairs no such split
exists. What this statistic measures is therefore the stability of the ranking
under a change of clinical stratification, not its replication in independent
patients. The sample-level split is the one in `reliability.py`, which halves
the samples of a unit and is used to disattenuate the similarity.

Shuffling is confined to a source x tissue stratum, exactly as in `03_edges.py`,
so a shuffled pair keeps the tissue pairing and cohort composition of a real one
and its background is drawn the same way.

    python 13_mechanism_null.py --net cond --shuffles 20
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

from grn import paths as config                                  # noqa: E402
from grn.tools.scaffold import mechanism as mech                 # noqa: E402

OUT = config.READOUT

TOP_TFS = 12          # the number of factors a pair reports
TOPK = 100            # list length for the split-half comparison
MIN_VIEWS = 8         # below this a split-half is not defined
MIN_GROUPS = 3        # patient groups each disease must contribute
MAX_VIEWS = 96        # cap, as in the ablation
BG_MIN = 200          # unrelated group pairs needed for a tissue-matched bar
DRAWS = 100           # split-half draws per pair
BG_N = 3000
# combination-count bins; a real pair is compared only within its own bin
BINS = [8, 20, 40, 96, 10 ** 9]              # combination count
GBINS = [3, 4, 6, 10, 10 ** 9]                # groups on the smaller side


def bin_of(n, gmin):
    """Stratum key: combination count and the smaller side's group count."""
    return (int(np.searchsorted(BINS, n, side="right") - 1),
            int(np.searchsorted(GBINS, gmin, side="right") - 1))


def stats_for(C, HI, rng, ga, gb, draws=DRAWS):
    """The two quantities, computed exactly as the read-out computes them.

    strength  median recurrence of the top TOP_TFS under the reported order
    agreement split-half overlap of the top TOPK under that same order, the two
              halves drawn from disjoint patient groups. Disjoint groups are not
              disjoint patients: see the module docstring.
    ranks     the recurrence of each of the top TOP_TFS, highest first. A factor
              is reported because it is among the best twelve of 837, so its
              recurrence cannot be judged against the 0.10 a single factor would
              give; it has to be judged against what the kth best factor reaches
              when the disease labels are arbitrary.
    """
    n = C.shape[1]
    ok = np.isfinite(C).all(axis=1) & np.isfinite(HI).all(axis=1)
    nanr = np.full(TOP_TFS, np.nan)
    if ok.sum() < TOPK + 10 or n < 4:
        return np.nan, np.nan, nanr
    Cf = C[ok]
    sup = (Cf >= HI[ok]).astype(np.float32)
    recur = sup.mean(axis=1)
    order = np.lexsort((np.median(Cf, axis=1), recur))
    top = order[-TOP_TFS:][::-1]
    strength = float(np.median(recur[top]))
    ranks = recur[top].astype(float)

    ua, ub = np.unique(ga), np.unique(gb)
    if len(ua) < 2 or len(ub) < 2:
        return strength, np.nan, ranks
    k = min(n // 4, 48)
    if k < 1:
        return strength, np.nan, ranks
    hits = []
    for _ in range(draws):
        pa, pb = rng.permutation(ua), rng.permutation(ub)
        m1 = np.where(np.isin(ga, pa[:len(ua) // 2]) &
                      np.isin(gb, pb[:len(ub) // 2]))[0]
        m2 = np.where(np.isin(ga, pa[len(ua) // 2:]) &
                      np.isin(gb, pb[len(ub) // 2:]))[0]
        if len(m1) < k or len(m2) < k:
            continue
        a = rng.choice(m1, k, replace=False)
        b = rng.choice(m2, k, replace=False)
        oa = np.lexsort((np.median(Cf[:, a], axis=1), sup[:, a].mean(axis=1)))
        ob = np.lexsort((np.median(Cf[:, b], axis=1), sup[:, b].mean(axis=1)))
        hits.append(len(set(oa[-TOPK:]) & set(ob[-TOPK:])) / TOPK)
    if len(hits) < max(5, draws // 10):
        return strength, np.nan, ranks
    return strength, float(np.mean(hits)), ranks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=["uncond", "cond"], default="cond")
    ap.add_argument("--shuffles", type=int, default=40,
                    help="label shuffles; each yields many shuffled pairs")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    _W, M, _E, tfs, _cols, Z, _tg = mech.build(args.net)
    keep = np.array([i for i, z in enumerate(Z) if z is not None])
    Zk = [Z[i] for i in keep]
    print(f"{args.net}: {len(M)} groups / {len(keep)} usable factors", flush=True)

    ov = config.sample_overlap(M.nid.values)
    dz = M.disease.values
    tis = M.tissue_class.values
    src = (M.source.astype(str) + "|" + M.tissue_class.astype(str)).values
    rng = np.random.default_rng(args.seed)

    # ---- backgrounds, cached by tissue pairing ----
    bi, bj = mech.sample_pairs(M, ov, np.random.default_rng(0), 5000)
    bg_global = mech.tf_cos(Z, bi, bj)[keep]
    hi_cache = {}

    def hi_for(ta, tb):
        """The bar, or None where the tissue pairing cannot support one."""
        tkey = "|".join(sorted((ta, tb)))
        if tkey not in hi_cache:
            p, q = mech.sample_pairs(M, ov, np.random.default_rng(1), BG_N,
                                     tissue_pair=tkey)
            hi_cache[tkey] = (np.nanpercentile(mech.tf_cos(Z, p, q)[keep],
                                               mech.HIGH_PCTL, axis=1)
                              if len(p) >= BG_MIN else None)
        return hi_cache[tkey]

    def bars(aa, bb):
        """One background column per combination, from its own two tissues.

        Returns the retained combinations alongside, since a combination whose
        tissue pairing has no estimable background cannot be scored.
        """
        cols, keep_c = [], []
        for i, (x, y) in enumerate(zip(aa, bb)):
            h = hi_for(tis[x], tis[y])
            if h is not None:
                cols.append(h)
                keep_c.append(i)
        if len(keep_c) < MIN_VIEWS:
            return None, None
        return np.column_stack(cols), np.asarray(keep_c)

    def cosines(aa, bb):
        C = np.empty((len(aa), len(Zk)), dtype=np.float32)
        for t, z in enumerate(Zk):
            C[:, t] = np.einsum("ij,ij->i", z[aa], z[bb])
        return C.T

    def pairs_from(labels, cap_rng):
        """Label pairs that can carry the claim, capped, with their stratum."""
        by = {d: np.where(labels == d)[0] for d in pd.unique(labels)}
        ds = sorted(by)
        out = []
        for x in range(len(ds)):
            for y in range(x + 1, len(ds)):
                ia, ib = by[ds[x]], by[ds[y]]
                if min(len(ia), len(ib)) < MIN_GROUPS:
                    continue
                aa, bb = np.meshgrid(ia, ib, indexing="ij")
                aa, bb = aa.ravel(), bb.ravel()
                good = ~ov[aa, bb]
                if good.sum() < MIN_VIEWS:
                    continue
                aa, bb = aa[good], bb[good]
                if len(aa) > MAX_VIEWS:
                    s = cap_rng.choice(len(aa), MAX_VIEWS, replace=False)
                    aa, bb = aa[s], bb[s]
                out.append((f"{ds[x]}||{ds[y]}", aa, bb,
                            min(len(ia), len(ib))))
        return out

    # ---- observed ----
    alias = mech.same_patient_diseases(M, ov)
    obs = []
    for name, aa, bb, gmin in pairs_from(dz, np.random.default_rng(3)):
        if name in alias:
            continue
        HI, kc = bars(aa, bb)
        if HI is None:
            continue
        aa, bb = aa[kc], bb[kc]
        s, a, rk = stats_for(cosines(aa, bb), HI, rng, aa, bb, args.draws)
        d = dict(pair=name, n_views=len(aa), n_groups_min=gmin,
                 bin=str(bin_of(len(aa), gmin)), strength=s, agreement=a)
        d.update({f"r{j + 1}": rk[j] for j in range(TOP_TFS)})
        obs.append(d)
        if len(obs) % 250 == 0:
            print(f"  observed {len(obs)}", flush=True)
    O = pd.DataFrame(obs).dropna(subset=["strength", "agreement"])
    print(f"{len(O)} observed disease pairs (>={MIN_GROUPS} groups each side, "
          f">={MIN_VIEWS} combinations, "
          f">={BG_MIN} tissue-matched background pairs; alias pairs excluded)",
          flush=True)

    # ---- null: shuffle labels within source x tissue, exactly as 03_edges ----
    from collections import defaultdict
    null = defaultdict(lambda: {"strength": [], "agreement": [], "ranks": []})
    for it in range(args.shuffles):
        perm = np.arange(len(M))
        for s_ in np.unique(src):
            m_ = np.where(src == s_)[0]
            perm[m_] = rng.permutation(m_)
        pz = dz[perm]
        for name, aa, bb, gmin in pairs_from(pz, rng):
            HI, kc = bars(aa, bb)
            if HI is None:
                continue
            aa, bb = aa[kc], bb[kc]
            st, ag, rk = stats_for(cosines(aa, bb), HI, rng, aa, bb,
                                   args.draws)
            if not (np.isfinite(st) and np.isfinite(ag)):
                continue
            b = str(bin_of(len(aa), gmin))
            null[b]["strength"].append(st)
            null[b]["agreement"].append(ag)
            null[b]["ranks"].append(rk)
        tot = sum(len(v["strength"]) for v in null.values())
        print(f"  shuffle {it + 1}/{args.shuffles}, {tot} null values so far",
              flush=True)

    # ---- p per observed pair, within its own bin ----
    # Two marginal tests are reported for continuity, but the pair is selected
    # on one calibrated quantity: how often chance produces a pair at least as
    # good on BOTH counts at once. The null values are stored paired, so this
    # needs no assumption about how the two statistics covary.
    for b in O["bin"].unique():
        m_ = (O["bin"] == b).values
        if b not in null or not null[b]["strength"]:
            continue
        ns = np.asarray(null[b]["strength"])
        na = np.asarray(null[b]["agreement"])
        B = len(ns)
        so = O["strength"].values[m_][:, None]
        ao = O["agreement"].values[m_][:, None]
        dom = ((ns[None, :] >= so) & (na[None, :] >= ao)).sum(axis=1)
        O.loc[m_, "p_joint"] = (1 + dom) / (1 + B)
        O.loc[m_, "n_null_joint"] = B
        R = np.asarray(null[b]["ranks"])                     # (B, TOP_TFS)
        for j in range(TOP_TFS):
            v = O[f"r{j + 1}"].values[m_][:, None]
            hit = (R[None, :, j] >= v).sum(axis=1)
            O.loc[m_, f"p_r{j + 1}"] = (1 + hit) / (1 + B)

    np.savez_compressed(
        f"{OUT}/mechanism_null_draws_{args.net}.npz",
        **{f"{b}|{k}": np.asarray(null[b][k])
           for b in null for k in ("strength", "agreement", "ranks")})

    for col in ("strength", "agreement"):
        pv = np.ones(len(O))
        for b in O["bin"].unique():
            nl = np.sort(np.asarray(null[b][col])) if b in null else np.array([])
            m_ = (O["bin"] == b).values
            if len(nl) == 0:
                continue
            v = O[col].values[m_]
            pv[m_] = (len(nl) - np.searchsorted(nl, v, side="left") + 1) / (len(nl) + 1)
        O[f"p_{col}"] = pv
        O[f"n_null_{col}"] = [len(null[b][col]) if b in null else 0
                              for b in O["bin"]]
    def bh(pv):
        pv = np.asarray(pv, float)
        ok = np.isfinite(pv)
        q = np.full(len(pv), np.nan)
        x = pv[ok]
        n = len(x)
        if n:
            o = np.argsort(x)
            r = np.empty(n)
            r[o] = np.minimum.accumulate(
                (x[o] * n / np.arange(1, n + 1))[::-1])[::-1]
            q[ok] = np.minimum(r, 1.0)
        return q

    for col in ("strength", "agreement", "joint"):
        O[f"q_{col}"] = bh(O[f"p_{col}"].values)

    O = O.sort_values("p_joint")
    O.to_csv(f"{OUT}/mechanism_null_{args.net}.csv", index=False)

    print("\n=== the null within each stratum (combinations, smaller side) ===")
    print(f"{'stratum':<14}{'null':>7}{'obs':>6}{'null strength':>15}"
          f"{'null agreement':>16}")
    for b in sorted(null, key=lambda x: str(x)):
        if not null[b]["strength"]:
            continue
        print(f"{str(b):<14}{len(null[b]['strength']):>7}"
              f"{int((O['bin'] == b).sum()):>6}"
              f"{np.median(null[b]['strength']):>13.3f}"
              f"{np.median(null[b]['agreement']):>14.3f}")

    both = (O.p_strength < 0.05) & (O.p_agreement < 0.05)
    print(f"\n{len(O)} pairs: p_strength<0.05 {int((O.p_strength < 0.05).sum())}, "
          f"p_agreement<0.05 {int((O.p_agreement < 0.05).sum())}，"
          f"both {int(both.sum())}")
    print("\nthe 25 pairs passing both, ordered by p_strength:")
    print(O[both].head(25)[["pair", "n_views", "n_groups_min", "strength",
                            "p_strength", "agreement", "p_agreement"]]
          .to_string(index=False, max_colwidth=46))
    print(f"\nwrote {OUT}/mechanism_null_{args.net}.csv")
    return 0


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

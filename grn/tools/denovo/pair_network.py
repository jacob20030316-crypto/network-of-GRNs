"""View 2 — the disease-pair layer on the de novo networks.

The scaffold route answers "are these two diseases related" by scoring a fixed
edge table. It cannot see a regulatory relationship the table does not contain.
The de novo route can, so this builds the same layer-2 object from it: every
disease a node, every pair an edge, and for each pair the transcription factors
whose regulons the two diseases actually share.

**The similarity is chosen, not assumed.** Comparing two inferred networks by the
overlap of their TF sets throws away everything the inference produced — which
targets, and how strongly. Several definitions are put on one ruler here and the
ruler is the same one used for the scaffold: the same disease seen through two
independent cohorts, sharing no patients, must score higher than two different
diseases. Whatever wins that is what builds the network.

**Sparsity is already fixed upstream.** Every unit was inferred from exactly 100
samples over three independent draws with a majority vote, so a unit's regulon
count reflects its biology rather than its cohort size — without that, any
similarity here would mostly be measuring how many samples each disease had.

**A TF that is shared by everything is annotated, not deleted.** With 39 diseases
across 24 cohorts, a tissue class holds too few diseases to build the leave-out
background that view 1 uses, and a filter estimated from two diseases would
mostly delete the pair being tested. So genericity is reported as a column —
in how many of the pair-level read-outs this TF appears — and the reader can see
it next to the finding.

    python 06_pair_network.py --compare              # pick the similarity
    python 06_pair_network.py --emit-agent
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from grn import paths as config                                  # noqa: E402

OUT = config.DENOVO_OUT

SEED = 0


def load_regulons():
    """unit_id -> {tf: (nes, frozenset(targets))}, plus unit metadata."""
    U = pd.read_csv(config.ANALYSIS_UNITS).set_index("unit_id")
    T = pd.read_csv(config.COHORT_TISSUE).set_index(["disease", "cohort"])
    tag2id = {re.sub(r"[^A-Za-z0-9_.-]", "_", i): i for i in U.index}
    sets, meta = {}, []
    for p in sorted(glob.glob(f"{OUT}/*.regulons.tsv")):
        tag = os.path.basename(p).replace(".regulons.tsv", "")
        if re.search(r"__r\d+$", tag):          # per-draw tables, not units
            continue
        uid = tag2id.get(tag)
        if uid is None:
            continue
        R = pd.read_csv(p, sep="\t")
        if not len(R):
            continue
        sets[uid] = {str(t).upper():
                     (float(n), frozenset(g.upper()
                                          for g in str(tg).split("|") if g))
                     for t, n, tg in zip(R.tf, R.nes, R.targets)}
        u = U.loc[uid]
        meta.append(dict(unit_id=uid, disease=u.disease, cohort=u.cohort,
                         source=u.source, n=int(u.n), n_regulon=len(R),
                         tissue=T.tissue_class.get((u.disease, u.cohort),
                                                   "unknown")))
    M = pd.DataFrame(meta).set_index("unit_id")
    return sets, M.loc[[u for u in M.index if u in sets]]


# ---------------------------------------------------------------- similarities
def _per_tf(a, b):
    """Yield (tf, jaccard, weight) over TFs present in both units."""
    for t in a.keys() & b.keys():
        na, ta = a[t]
        nb, tb = b[t]
        u = len(ta | tb)
        if u:
            yield t, len(ta & tb) / u, float(np.sqrt(na * nb))


def sim_tfset(a, b):
    """Baseline: do the two networks name the same TFs at all."""
    return len(a.keys() & b.keys()) / max(len(a.keys() | b.keys()), 1)


def sim_edge_jaccard(a, b):
    ea = {(t, g) for t, (_, tg) in a.items() for g in tg}
    eb = {(t, g) for t, (_, tg) in b.items() for g in tg}
    return len(ea & eb) / max(len(ea | eb), 1)


def sim_targ_mean(a, b):
    """Mean per-TF target Jaccard over the TFs both networks found."""
    v = [j for _, j, _ in _per_tf(a, b)]
    return float(np.mean(v)) if v else 0.0


def sim_targ_nes_shared(a, b):
    """NES-weighted per-TF target Jaccard, normalised over shared TFs.
    Asks: where both networks agree a TF is a regulator, do they agree on what
    it regulates — with the confidently-called regulons counting for more."""
    num = den = 0.0
    for _, j, w in _per_tf(a, b):
        num += w * j
        den += w
    return num / den if den else 0.0


def sim_targ_nes_union(a, b):
    """Same, but normalised over the union of TFs, so a TF found by only one of
    the two networks counts against the score. A weighted Jaccard proper."""
    num = den = 0.0
    for _, j, w in _per_tf(a, b):
        num += w * j
    for t in a.keys() | b.keys():
        if t in a and t in b:
            den += float(np.sqrt(a[t][0] * b[t][0]))
        else:
            den += float((a.get(t) or b.get(t))[0])
    return num / den if den else 0.0


METHODS = {
    "factor-set Jaccard": sim_tfset,
    "edge-set Jaccard": sim_edge_jaccard,
    "mean per-factor target Jaccard": sim_targ_mean,
    "per-factor Jaccard x NES (shared factors)": sim_targ_nes_shared,
    "per-factor Jaccard x NES (union)": sim_targ_nes_union,
}


def matrix(sets, uids, fn):
    n = len(uids)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = fn(sets[uids[i]], sets[uids[j]])
    return S


def masks(M, uids):
    ov = config.sample_overlap(np.array(uids))
    dz = M.loc[uids].disease.values
    ch = M.loc[uids].cohort.values
    ti = M.loc[uids].tissue.values
    iu = np.triu_indices(len(uids), 1)
    same_d = dz[iu[0]] == dz[iu[1]]
    same_c = ch[iu[0]] == ch[iu[1]]
    same_t = ti[iu[0]] == ti[iu[1]]
    o = ov[iu]
    return iu, dict(pos=same_d & ~same_c & ~o, neg=~same_d & ~o,
                    neg_t=~same_d & ~o & same_t, ov=ov)


def bh(p):
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def compare(sets, M, uids):
    iu, mk = masks(M, uids)
    print(f"\n=== similarity definitions, on one ruler ===")
    print(f"  positives: same disease, different cohorts, no shared samples  "
          f"n={int(mk['pos'].sum())}")
    print(f"  negatives: different diseases n={int(mk['neg'].sum())}   "
          f"of which same tissue "
          f"n={int(mk['neg_t'].sum())}")
    print(f"\n  {'definition':<42}{'AUC':>7}{'AUC tissue':>12}{'pos med':>10}"
          f"{'neg med':>10}{'p':>11}")
    rows = []
    for name, fn in METHODS.items():
        S = matrix(sets, uids, fn)
        v = S[iu]
        a, b = v[mk["pos"]], v[mk["neg"]]
        u = mannwhitneyu(a, b, alternative="greater")
        au = u.statistic / (len(a) * len(b))
        bt = v[mk["neg_t"]]
        aut = (mannwhitneyu(a, bt, alternative="greater").statistic
               / (len(a) * len(bt))) if len(bt) >= 5 else np.nan
        print(f"  {name:<26}{au:>9.3f}{aut:>11.3f}{np.median(a):>10.4f}"
              f"{np.median(b):>10.4f}{u.pvalue:>11.2e}")
        rows.append(dict(method=name, auc=au, auc_tissue_matched=aut,
                         pos_median=np.median(a), neg_median=np.median(b),
                         p=u.pvalue))
    R = pd.DataFrame(rows).sort_values("auc", ascending=False)
    R.to_csv(f"{OUT}/denovo_method_comparison.csv", index=False)
    print(f"\n  wrote denovo_method_comparison.csv    best: {R.iloc[0].method}")
    return R.iloc[0].method


def build_pairs(sets, M, uids, method, perms):
    """Disease-pair table: aggregate unit pairs, calibrate by label permutation."""
    fn = METHODS[method]
    S = matrix(sets, uids, fn)
    iu, mk = masks(M, uids)
    dz = M.loc[uids].disease.values
    ti = M.loc[uids].tissue.values
    v = S[iu]
    cross = mk["neg"]
    a, b = dz[iu[0]], dz[iu[1]]
    key = np.where(a < b, a, b) + "||" + np.where(a < b, b, a)

    obs = (pd.DataFrame({"pair": key[cross], "w": v[cross]})
           .groupby("pair").agg(strength=("w", "median"),
                                n_unit_pairs=("w", "size")))
    # Shuffle disease labels within tissue class, so a same-tissue pair competes
    # against other same-tissue pairs rather than against the whole graph.
    rng = np.random.default_rng(SEED)
    pool = defaultdict(list)
    sizes = obs.n_unit_pairs.values
    edges_b = np.unique(np.quantile(sizes, np.linspace(0, 1, 5)))
    stratum = np.clip(np.searchsorted(edges_b, sizes, side="right") - 1,
                      0, len(edges_b) - 1)
    for _ in range(perms):
        perm = np.arange(len(uids))
        for t in np.unique(ti):
            m = np.where(ti == t)[0]
            perm[m] = rng.permutation(m)
        pz = dz[perm]
        pa, pb = pz[iu[0]], pz[iu[1]]
        pc = (pa != pb) & cross          # the permuted set must obey the same
        pk = np.where(pa < pb, pa, pb) + "||" + np.where(pa < pb, pb, pa)
        g = (pd.DataFrame({"pair": pk[pc], "w": v[pc]})
             .groupby("pair").agg(med=("w", "median"), k=("w", "size")))
        gs = np.clip(np.searchsorted(edges_b, g.k.values, side="right") - 1,
                     0, len(edges_b) - 1)
        for s in np.unique(gs):
            pool[s].append(g.med.values[gs == s])
    pv = np.ones(len(obs))
    for s in np.unique(stratum):
        null = np.concatenate(pool[s]) if pool[s] else np.array([0.0])
        null.sort()
        m = stratum == s
        pv[m] = (len(null) - np.searchsorted(null, obs.strength.values[m],
                                             side="left") + 1) / (len(null) + 1)
    obs["p_perm"] = pv
    obs["q_bh"] = bh(pv)
    tis = dict(zip(M.loc[uids].disease, M.loc[uids].tissue))
    obs["tissue_a"] = [tis.get(p.split("||")[0]) for p in obs.index]
    obs["tissue_b"] = [tis.get(p.split("||")[1]) for p in obs.index]
    obs["same_tissue"] = obs.tissue_a == obs.tissue_b
    return obs.reset_index(), S, iu, cross, key


def tf_background(sets, M, uids, iu, cross, min_obs=30):
    """Each TF's own distribution of target-Jaccard over random cross-disease
    unit pairs.

    A raw Jaccard cannot be compared between TFs. A regulon with two targets
    lands on 0.5 or 1.0 by arithmetic, so ranking a pair's TFs by raw agreement
    returns whichever regulons are smallest — measured here before the fix, the
    top of every pair was HOX factors sharing one target at Jaccard 1.00, while
    STAT1 sharing 30 targets ranked sixth. This is the same saturation the
    scaffold route hit, and it takes the same fix: score each TF against its own
    background rather than against the other TFs.
    """
    a, b = iu[0][cross], iu[1][cross]
    bg = defaultdict(list)
    for x, y in zip(a, b):
        A, B = sets[uids[x]], sets[uids[y]]
        for t, j, _ in _per_tf(A, B):
            bg[t].append(j)
    return {t: np.sort(np.array(v)) for t, v in bg.items() if len(v) >= min_obs}


def pair_mechanisms(sets, M, uids, pairs, iu, cross, top_pairs,
                    high_pctl=90, min_shared=5):
    """For each disease pair, the TFs whose regulons the two diseases share.

    Ranked by RECURRENCE — in how many of the pair's unit pairs the TF clears
    the 90th percentile of its own background — which is the same read-out rule
    view 1 uses, so the two halves of the paper report the same quantity. A
    random TF recurs at 0.10 by construction, so the fold over that is what
    compares across pairs.
    """
    bg = tf_background(sets, M, uids, iu, cross)
    # A TF whose background is almost all zeros has a 90th percentile of 0, and
    # then `jaccard >= threshold` is satisfied by sharing NOTHING. Measured
    # before this line, USF2, PITX3 and TLX2 came back at recurrence 1.00 with
    # zero shared targets. Such a TF carries no information either way and is
    # dropped rather than ranked.
    bg = {t: v for t, v in bg.items() if np.quantile(v, high_pctl / 100) > 0}
    hi = {t: float(np.quantile(v, high_pctl / 100)) for t, v in bg.items()}
    byd = defaultdict(list)
    for i, u in enumerate(uids):
        byd[M.loc[u].disease].append(i)
    ovm = config.sample_overlap(np.array(uids))
    rows = []
    for pair in pairs.nsmallest(top_pairs, "p_perm").pair.tolist():
        da, db = pair.split("||")
        vps = [(x, y) for x in byd[da] for y in byd[db] if not ovm[x, y]]
        if not vps:
            continue
        acc, tgt = defaultdict(list), defaultdict(set)
        for x, y in vps:
            A, B = sets[uids[x]], sets[uids[y]]
            for t, j, _ in _per_tf(A, B):
                if t not in bg:
                    continue
                acc[t].append((j, A[t][0], B[t][0]))
                tgt[t] |= (A[t][1] & B[t][1])
        for t, vals in acc.items():
            j = np.array([x[0] for x in vals])
            med = float(np.median(j))
            # recurrence is over ALL the pair's unit pairs, not only those where
            # both networks happened to call the TF: a TF missing from one side
            # is a view that does not support it, not a view that abstains
            rec = float((j >= hi[t]).sum() / len(vps))
            rows.append(dict(
                pair=pair, disease_a=da, disease_b=db, tf=t,
                n_unit_pairs=len(vps), n_present=len(vals),
                presence=len(vals) / len(vps),
                recurrence=rec, recur_fold=rec / (1 - high_pctl / 100),
                jaccard_median=med,
                bg_pctl=float(np.searchsorted(bg[t], med) / len(bg[t])),
                bg_n=len(bg[t]),
                nes_a=float(np.median([x[1] for x in vals])),
                nes_b=float(np.median([x[2] for x in vals])),
                n_shared_targets=len(tgt[t]),
                shared_targets="|".join(sorted(tgt[t])[:40])))
    R = pd.DataFrame(rows)
    if not len(R):
        return R
    # View 1 ranks by recurrence because a disease pair there is seen through
    # dozens of condition views. Here the network is unconditional and most
    # disease pairs hold one or two unit pairs, so recurrence takes two or three
    # distinct values and cannot order anything — it is reported, and the
    # background percentile does the ranking.
    R["passes_gate"] = ((R.bg_pctl >= 0.90) & (R.n_shared_targets >= min_shared)
                        & (R.jaccard_median > 0))
    # genericity is annotated, never subtracted: with 39 diseases over 24
    # cohorts a tissue class holds too few diseases to build the leave-out
    # background view 1 uses, and a filter estimated from two diseases would
    # mostly delete the pair under test
    K = R[R.passes_gate]
    R["generic_frac"] = (R.tf.map(K.tf.value_counts()).fillna(0)
                         / max(R.pair.nunique(), 1))
    return R.sort_values(["pair", "bg_pctl", "n_shared_targets"],
                         ascending=[True, False, False])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="compare the definitions and stop, without building")
    ap.add_argument("--method", default=None)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--top-pairs", type=int, default=40)
    ap.add_argument("--top-tfs", type=int, default=12)
    ap.add_argument("--emit-agent", action="store_true")
    args = ap.parse_args(argv)

    sets, M = load_regulons()
    uids = list(M.index)
    print(f"de novo: {len(uids)} nodes / {M.disease.nunique()} diseases / "
          f"{M.cohort.nunique()} cohorts")
    print(f"  regulons {M.n_regulon.min()}-{M.n_regulon.max()}"
          f" (median {int(M.n_regulon.median())}), every node drawn at m=100")

    method = args.method or compare(sets, M, uids)
    if args.compare:
        return

    print(f"\n=== disease-pair network ({method}, {args.perms} permutations) ===")
    pairs, S, iu, cross, key = build_pairs(sets, M, uids, method, args.perms)
    pairs.to_csv(f"{OUT}/denovo_pairs.csv", index=False)
    print(f"  {len(pairs)} disease pairs   FDR<0.05: "
          f"{int((pairs.q_bh < 0.05).sum())}   p_perm<0.05: "
          f"{int((pairs.p_perm < 0.05).sum())}")
    print(f"\n  the 15 strongest:")
    print(f"  {'disease pair':<58}{'strength':>9}{'nodes':>7}{'p':>9}{'tissue':>22}")
    for _, r in pairs.nsmallest(15, "p_perm").iterrows():
        tk = f"{r.tissue_a}|{r.tissue_b}"
        print(f"  {r.pair[:56]:<58}{r.strength:>8.4f}{int(r.n_unit_pairs):>7}"
              f"{r.p_perm:>9.4f}{tk[:20]:>22}")

    RALL = pair_mechanisms(sets, M, uids, pairs, iu, cross, args.top_pairs)
    R = RALL[RALL.passes_gate]
    R.to_csv(f"{OUT}/denovo_pair_mechanisms.csv", index=False)
    print(f"\n  wrote denovo_pair_mechanisms.csv"
          f" ({len(R)} rows / {R.pair.nunique()} disease pairs)")

    print(f"\n=== the regulons each pair shares (first 6) ===")
    for pair in pairs.nsmallest(6, "p_perm").pair:
        g = R[R.pair == pair]
        if not len(g):
            continue
        print(f"\n  {pair}")
        print(f"    {'factor':<12}{'recur':>8}{'nodes':>7}{'bg pctl':>9}"
              f"{'Jaccard':>9}{'shared':>8}{'NES':>7}{'general':>9}")
        for _, r in g.head(args.top_tfs).iterrows():
            print(f"    {r.tf:<12}{r.recurrence:>8.2f}{r.n_unit_pairs:>6.0f}"
                  f"{r.bg_pctl:>9.3f}{r.jaccard_median:>9.3f}"
                  f"{int(r.n_shared_targets):>7}"
                  f"{min(r.nes_a, r.nes_b):>7.1f}{r.generic_frac:>8.2f}")

    # Which pairs are worth a story. A pair whose top TFs are all high-genericity
    # is reporting the programme that every epithelial tumour shares — real, but
    # not about these two diseases. The pairs to tell are the ones carrying TFs
    # that are unusual for this pair AND rare across the other pairs.
    SPEC = 0.30
    summ = (R.groupby("pair").head(args.top_tfs).groupby("pair")
            .agg(n_tf=("tf", "size"),
                 n_specific=("generic_frac", lambda x: int((x < SPEC).sum())),
                 mean_generic=("generic_frac", "mean"),
                 top_shared=("n_shared_targets", "max")))
    summ = summ.join(pairs.set_index("pair")[["strength", "p_perm",
                                              "same_tissue"]])
    summ = summ.sort_values(["n_specific", "p_perm"], ascending=[False, True])
    summ.to_csv(f"{OUT}/denovo_pair_summary.csv")
    print(f"\n=== pairs with the most specific factors (of the top 12, "
          f"generality <{SPEC}) ===")
    print(f"  {'disease pair':<56}{'specific':>9}{'mean gen':>10}"
          f"{'strength':>10}{'p':>8}")
    for p, r in summ.head(12).iterrows():
        print(f"  {p[:54]:<56}{int(r.n_specific):>7}{r.mean_generic:>11.2f}"
              f"{r.strength:>8.4f}{r.p_perm:>8.4f}")

    if args.emit_agent:
        prompts = []
        for _, p in pairs.nsmallest(args.top_pairs, "p_perm").iterrows():
            g = R[R.pair == p.pair].head(args.top_tfs)
            if not len(g):
                continue
            # What did not make the list. These are TFs whose two regulons agree
            # perfectly — often at Jaccard 1.00 — on too few targets to be a
            # mechanism. Before each TF was scored against its own background
            # they topped every pair: HOX factors sharing one gene outranked
            # STAT1 sharing thirty. Naming them lets a story say what it is NOT
            # built on, and stops an agent from reaching for them itself.
            gp = RALL[RALL.pair == p.pair]
            thin = gp[(gp.bg_pctl >= 0.90) & ~gp.passes_gate].nlargest(
                args.top_tfs, "jaccard_median")
            prompts.append(dict(
                view=2, pair=p.pair,
                disease_a=p.pair.split("||")[0],
                disease_b=p.pair.split("||")[1],
                tissue_pairing=f"{p.tissue_a}|{p.tissue_b}",
                same_tissue=bool(p.same_tissue),
                strength=round(float(p.strength), 4),
                n_unit_pairs=int(p.n_unit_pairs), p_perm=round(float(p.p_perm), 4),
                tfs=[dict(tf=r.tf, recurrence=round(r.recurrence, 2),
                          recurrence_fold_over_chance=round(r.recur_fold, 1),
                          bg_percentile=round(r.bg_pctl, 3),
                          presence=round(r.presence, 2),
                          target_jaccard=round(r.jaccard_median, 3),
                          n_shared_targets=int(r.n_shared_targets),
                          nes_a=round(r.nes_a, 1), nes_b=round(r.nes_b, 1),
                          generic_fraction=round(r.generic_frac, 2),
                          shared_targets=r.shared_targets.split("|")[:25])
                     for _, r in g.iterrows()],
                excluded=dict(
                    n_below_threshold=int((~gp.passes_gate).sum()),
                    by_thin_overlap=[
                        dict(tf=r.tf, target_jaccard=round(r.jaccard_median, 3),
                             n_shared_targets=int(r.n_shared_targets),
                             bg_percentile=round(r.bg_pctl, 3))
                        for _, r in thin.iterrows()],
                    note="the two regulons of these factors agree almost "
                         "entirely, often at Jaccard 1.00, "
                         "but share too few targets (<5) to be a mechanism. "
                         "Do not build an account on them. "
                         "Also: a high generic_fraction (>0.30) above means "
                         "the factor is common to most pairs or generic to the "
                         "tissue, rather than specific to this pair."),
                note="targets come from de novo inference; those absent from "
                     "the curated table are the new ones. "
                     "generic_fraction is how often the factor appears across "
                     "all pair read-outs; "
                     "a high value suggests a generic or tissue programme "
                     "rather than something specific to this pair."))
        pth = f"{OUT}/denovo_agent_prompts.jsonl"
        with open(pth, "w", encoding="utf-8") as fh:
            for d in prompts:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\nwrote {pth} ({len(prompts)} disease pairs)")


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

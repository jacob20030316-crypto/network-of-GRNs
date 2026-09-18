"""View 1 — which TFs hold a disease pair together no matter which subgroup you look at.

The unit of analysis is a DISEASE PAIR, read through every one of its views. A
disease pair is seen many times over: this cohort's women against that disease's
antibody-positive patients, this one's over-60s against that one's stage-II
patients, and so on. The claim view 1 makes is:

    a mechanism that survives every one of those views is the real one.

So the ranking is by how often a TF survives, not by how striking it looks once.

**Why not rank by how unusual a TF is on the single strongest edge.** Measured on
`COVID-19 || Sjögrens_Syndrome` (4 x 7 = 28 view pairs): on its strongest edge,
dozens of TFs sit at background percentile 0.998 — that edge is strong overall,
so everything on it looks unusual and the ordering degenerates. Ranking that way
put E2F4 near the top at a recurrence of 0.21, meaning 6 of 28 views supported
it; change the patient subgroup and it is gone. Ranking by recurrence instead
returns STAT2 and IRF9 at 0.93 — 26 of 28 views — which are the two components
of ISGF3, and both diseases are interferon-signature diseases. The two orderings
share nothing in their top 12.

**And why the raw recurrence cannot be the gate either.** A TF clears the bar in
8 of 8 views far more easily than in 500 of 598, so any absolute cut-off quietly
favours thinly-sampled pairs — the same tilt that disqualified `strength`, in a
new costume. Measured before the fix: spearman(view count, TFs passing) = -0.396,
and the pairs with the most views produced no output at all, including an
alias pair whose two labels denote the same patients. Since the bar is the 90th
percentile of each TF's own background, a random TF recurs at 0.10 by
construction, so the FOLD over that expectation is what compares across pairs.

Fields per TF:

  recurrence   fraction of the pair's view pairs where this TF clears the
               tissue-matched background's 90th percentile. The headline.
  recur_fold   that rate over the 0.10 a random TF would give. The gate.
  bg_pctl      median over views of the TF's percentile against its own
               background. Reported.
  median_cos   median agreement across the pair's view pairs. The tie-break,
               as in the split-half agreement and the permutation null.
  direction    concordant / discordant — the weights are signed correlations,
               so opposite coupling is visible and is not the same as absent.
  tissue_prog  whether the whole tissue class shares this TF anyway, computed
               with the two diseases under test held out of their own class.

    python 04_mechanism.py --net cond --top-pairs 40
    python 04_mechanism.py --net cond --pair "COVID-19||Sjögrens_Syndrome"
    python 04_mechanism.py --net cond --top-pairs 40 --emit-agent
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

from grn import paths as config                                  # noqa: E402

OUT = config.SCAFFOLD_OUT

CORE_PREVALENCE = 0.99
HIGH_PCTL = 90          # a view "supports" a TF if it clears this percentile


def build(net):
    """Per-TF unit profiles after the correction ladder, on core edges."""
    from grn.tools.scaffold import similarity as S4
    W = np.load(f"{OUT}/W_full.npy").astype(np.float64)
    A = np.load(f"{OUT}/A_avail.npy")
    M = pd.read_csv(f"{OUT}/W_meta.csv")
    E = pd.read_csv(f"{OUT}/edge_universe.csv")
    m = (M.network == net).values
    W, A, M = W[m], A[m], M[m].reset_index(drop=True)
    core = A.mean(axis=0) >= CORE_PREVALENCE
    W, E = W[:, core], E[core].reset_index(drop=True)
    W = S4.apply_ladder(W, E)

    tfs = pd.unique(E.source_genesymbol.values)
    pos = {t: i for i, t in enumerate(tfs)}
    cols = [[] for _ in tfs]
    for j, t in enumerate(E.source_genesymbol.values):
        cols[pos[t]].append(j)
    cols = [np.array(c, dtype=int) for c in cols]

    # per-TF unit x target profile, L2-normalised, so a per-TF cosine is one dot
    Z, targets = [], []
    for c in cols:
        if len(c) < 2:
            Z.append(None); targets.append(None); continue
        X = W[:, c]
        s = np.abs(X).sum(axis=1, keepdims=True); s[s < 1e-12] = 1.0
        X = X / s
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n < 1e-12] = np.nan
        Z.append(np.nan_to_num(X / n))
        targets.append(list(E.target_genesymbol.values[c]))

    T = pd.read_csv(config.COHORT_TISSUE).set_index(["disease", "cohort"])
    M["tissue_class"] = [T.tissue_class.get((d, c), "unknown")
                         for d, c in zip(M.disease, M.cohort)]
    return W, M, E, tfs, cols, Z, targets


def sample_pairs(M, ov, rng, n_sample, tissue_pair=None, exclude=(), within=None):
    """Random unit pairs: different diseases, no shared patients, optional
    tissue-pairing constraint, optional exclusion of named diseases."""
    N = len(M)
    dz = M.disease.values
    tp = M.tissue_class.values
    ii = rng.integers(0, N, n_sample * 40)
    jj = rng.integers(0, N, n_sample * 40)
    ok = (dz[ii] != dz[jj]) & ~ov[ii, jj]
    if exclude:
        ex = np.isin(dz, list(exclude))
        ok &= ~ex[ii] & ~ex[jj]
    if tissue_pair is not None:
        key = np.array(["|".join(sorted((x, y))) for x, y in zip(tp[ii], tp[jj])])
        ok &= key == tissue_pair
    if within is not None:
        ok &= (tp[ii] == within) & (tp[jj] == within)
    return ii[ok][:n_sample], jj[ok][:n_sample]


def same_patient_diseases(M, ov):
    """Disease pairs whose units share patients — the same people under two
    labels (GBM inside GBMLGG, rectal inside colorectal).

    Their similarity is the identity of the samples, not a finding, so they are
    kept out of the read-out entirely. They remain the positive control: any
    correction step that fails to leave them on top is a broken correction.
    Note this is a DISEASE-level test built from the unit-level overlap; two
    such diseases still have unit pairs that share nobody, which is exactly why
    the pair survives the edge rules and has to be excluded by name.
    """
    dz = M.disease.values
    out = set()
    for a, b in itertools.combinations(sorted(set(dz)), 2):
        ia = np.where(dz == a)[0]
        ib = np.where(dz == b)[0]
        if len(ia) and len(ib) and ov[np.ix_(ia, ib)].any():
            out.add("||".join((a, b)))
            out.add("||".join((b, a)))
    return out


def generality(A, top_k):
    """How many of the read-out pairs list this TF among their top ranks.

    A regulator that heads every pair's list is more likely a property of the
    corpus than of any one pair; one that heads a single pair is specific to it.
    Computed on the REPORTED ranks, not on the candidate pool: the gate only
    asks for twice the chance recurrence, so the pool is wide and its sharing
    rate says nothing about what the paper actually claims.
    """
    from collections import Counter
    c, n = Counter(), A.pair.nunique()
    for _, s in A[~A.tissue_programme & A.passes_gate].groupby("pair"):
        c.update(s.sort_values(["recurrence", "median_cos"], ascending=False)
                 .head(top_k).tf)
    return {t: v / n for t, v in c.items()}, n


def tf_cos(Z, ii, jj):
    """Per-TF cosine for a set of unit pairs -> (n_tf, n_pairs)."""
    out = np.full((len(Z), len(ii)), np.nan)
    for t, z in enumerate(Z):
        if z is not None:
            out[t] = (z[ii] * z[jj]).sum(axis=1)
    return out


def split_half_agreement(C, hi, rng, tops=(100, 12), draws=200):
    """How much of this pair's TF ranking is a property of the pair rather than
    of the views it happened to get.

    Split the pair's view pairs into two disjoint halves, rank the TFs inside
    each half by exactly the rule this script reports with — recurrence first,
    median cosine as tie-break — and measure how much the two top-K lists agree.
    A pair whose two halves return the same TFs has a read-out that does not
    depend on which patients were sampled; one whose halves disagree does.

    This is the same measurement as `Verification/11_mechanism_curve.py`, but at
    each pair's OWN view count rather than a fixed k, which is what makes it a
    confidence for the row actually being emitted: pairs with more views and
    more agreement among them score higher, and both of those are reasons to
    trust the read-out.
    """
    n = C.shape[1]
    out = {f"agreement_top{t}": np.nan for t in tops}
    out["agreement_k"] = np.nan
    if n < 4:
        return out
    k = min(n // 2, 48)
    ok = np.isfinite(C).all(axis=1)
    if ok.sum() < max(tops) + 10:
        return out
    Cf = C[ok]
    sup = (Cf >= hi[ok][:, None]).astype(np.float32)
    got = {t: [] for t in tops}
    for _ in range(draws):
        s = rng.permutation(n)
        a, b = s[:k], s[k:2 * k]
        # lexsort: last key is primary, so recurrence ranks and the median
        # cosine breaks ties — the same order `keep` is sorted by below
        oa = np.lexsort((np.median(Cf[:, a], axis=1), sup[:, a].mean(axis=1)))
        ob = np.lexsort((np.median(Cf[:, b], axis=1), sup[:, b].mean(axis=1)))
        for t in tops:
            got[t].append(len(set(oa[-t:]) & set(ob[-t:])) / t)
    for t in tops:
        out[f"agreement_top{t}"] = float(np.mean(got[t]))
    out["agreement_k"] = k
    return out


def tissue_program(Z, M, ov, bg_global, rng, tclass, exclude, min_diseases,
                   n_sample=3000):
    """Per-TF percentile of how ordinary a TF is for a tissue class.

    `exclude` drops the diseases under test from their own class before the
    profile is built. Without it a class holding few diseases is largely made of
    the pair being tested and the filter deletes the finding — measured, kidney
    holds 4 diseases, all renal cancers, and the unexcluded filter flagged 136
    TFs as generic, which is most of what a kidney-kidney pair could ever share.
    Classes with fewer than `min_diseases` left get no filter at all.
    """
    dz = M.disease.values
    left = set(dz[M.tissue_class.values == tclass]) - set(exclude)
    if len(left) < min_diseases:
        return None, len(left)
    ii, jj = sample_pairs(M, ov, rng, n_sample, exclude=exclude, within=tclass)
    if len(ii) < 200:
        return None, len(left)
    C = tf_cos(Z, ii, jj)
    pct = np.full(len(Z), np.nan)
    for t in range(len(Z)):
        if not np.isfinite(C[t]).any() or not np.isfinite(bg_global[t]).any():
            continue
        nb = np.sort(bg_global[t][np.isfinite(bg_global[t])])
        pct[t] = np.searchsorted(nb, np.nanmedian(C[t])) / len(nb)
    return pct, len(left)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=["uncond", "cond"], default="cond")
    ap.add_argument("--top-pairs", type=int, default=40,
                    help="disease pairs to read out, taken by p_perm")
    ap.add_argument("--pair", default=None)
    ap.add_argument("--pairs-from", default=None,
                    help="read out the pairs that survive the mechanism-layer "
                         "null, named in this CSV, instead of taking a number "
                         "of them off the edge layer. The edge layer ranks a "
                         "pair by the median over 44,350 interactions, which is "
                         "not what is being selected on here.")
    ap.add_argument("--q-col", default="q_joint")
    ap.add_argument("--q-max", type=float, default=0.05)
    ap.add_argument("--min-views", type=int, default=8,
                    help="a pair needs this many view pairs to be worth reading")
    ap.add_argument("--top-tfs", type=int, default=12)
    ap.add_argument("--tissue-cut", type=float, default=0.90)
    ap.add_argument("--min-fold", type=float, default=2.0,
                    help="a TF must recur at this multiple of the rate a random "
                         "TF would, before it is allowed to rank")
    ap.add_argument("--min-class-diseases", type=int, default=3)
    ap.add_argument("--emit-agent", action="store_true")
    args = ap.parse_args(argv)

    W, M, E, tfs, cols, Z, targets = build(args.net)
    ov = config.sample_overlap(M.nid.values)
    rng = np.random.default_rng(0)
    print(f"{args.net}: {len(M)} groups / {M.disease.nunique()} diseases / "
          f"{len(tfs)} factors ({sum(z is not None for z in Z)} with >=2 targets)")

    ii, jj = sample_pairs(M, ov, rng, 5000)
    bg_global = tf_cos(Z, ii, jj)
    print(f"global factor background built from {bg_global.shape[1]} random "
          f"cross-disease group pairs")

    # ---- pick the disease pairs, by p_perm
    D = pd.read_csv(f"{OUT}/disease_pairs_{args.net}.csv")
    alias = same_patient_diseases(M, ov)
    if args.pair:
        todo = [args.pair]
    elif args.pairs_from:
        Q = pd.read_csv(args.pairs_from)
        if args.q_col not in Q.columns:
            raise SystemExit(f"{args.pairs_from} has no column '{args.q_col}'")
        todo = sorted(Q.loc[Q[args.q_col] < args.q_max, "pair"])
        print(f"{len(todo)} disease pairs passed the mechanism-layer null"
              f" ({args.q_col} < {args.q_max}, of {len(Q)} evaluable)")
    else:
        sub = D[D.n_unit_pairs >= args.min_views]
        n_alias = int(sub.pair.isin(alias).sum())
        sub = sub[~sub.pair.isin(alias)]
        rank_col = "p_perm" if "p_perm" in D.columns else "strength"
        todo = list(sub.nsmallest(args.top_pairs, "p_perm").pair
                    if rank_col == "p_perm"
                    else sub.nlargest(args.top_pairs, "strength").pair)
        print(f"taking the top {len(todo)} disease pairs by {rank_col}"
              f" (with at least {args.min_views} combinations; "
              f"{n_alias} pairs whose labels denote the same patients excluded)")
    print()

    tclass = dict(zip(M.disease, M.tissue_class))
    bg_cache, tp_cache = {}, {}
    rows, prompts = [], []

    for pair in todo:
        da, db = pair.split("||")
        ia = np.where(M.disease.values == da)[0]
        ib = np.where(M.disease.values == db)[0]
        vp = [(x, y) for x in ia for y in ib if not ov[x, y]]
        if len(vp) < 2:
            print(f"{pair}: too few combinations; skipped")
            continue
        P = np.array([p[0] for p in vp])
        Q = np.array([p[1] for p in vp])

        ta, tb = tclass.get(da, "unknown"), tclass.get(db, "unknown")
        tkey = "|".join(sorted((ta, tb)))
        if tkey not in bg_cache:
            bi, bj = sample_pairs(M, ov, np.random.default_rng(1), 3000,
                                  tissue_pair=tkey)
            bg_cache[tkey] = tf_cos(Z, bi, bj) if len(bi) >= 200 else None
        bg = bg_cache[tkey]
        matched = bg is not None
        use = bg if matched else bg_global
        hi = np.nanpercentile(use, HIGH_PCTL, axis=1)

        prog = {}
        for tc in {ta, tb}:
            ck = (tc, frozenset((da, db)))
            if ck not in tp_cache:
                tp_cache[ck] = tissue_program(
                    Z, M, ov, bg_global, np.random.default_rng(2), tc,
                    exclude=(da, db), min_diseases=args.min_class_diseases)
            prog[tc] = tp_cache[ck]

        # every view pair at once — this is the aggregation view 1 is about
        C = tf_cos(Z, P, Q)
        recur = np.nanmean(C >= hi[:, None], axis=1)
        agr = split_half_agreement(C, hi, np.random.default_rng(3))

        R = []
        for t in range(len(Z)):
            if Z[t] is None or not np.isfinite(use[t]).any():
                continue
            nb = np.sort(use[t][np.isfinite(use[t])])
            med = float(np.nanmedian(C[t]))
            pa, pb = prog[ta][0], prog[tb][0]
            gen = max([x[t] for x in (pa, pb) if x is not None] or [np.nan])
            R.append(dict(
                pair=pair, tf=tfs[t], n_targets=len(cols[t]),
                n_views=len(vp), recurrence=float(recur[t]),
                bg_pctl=float(np.searchsorted(nb, med) / len(nb)),
                median_cos=med,
                direction="concordant" if med > 0 else "discordant",
                tissue_generic=float(gen) if np.isfinite(gen) else np.nan,
                tissue_pairing=tkey, **agr))
        R = pd.DataFrame(R)
        R["tissue_programme"] = R.tissue_generic.fillna(0) >= args.tissue_cut
        # Recurrence ranks, but its RAW value is not comparable across pairs: a
        # TF clears the bar in 8 of 8 views far more easily than in 500 of 598,
        # so an absolute cut-off silently favours thinly-sampled pairs — the same
        # tilt that disqualified `strength`. The bar is the 90th percentile of
        # the TF's own background, so a random TF recurs at 0.10 by construction;
        # the fold over that expectation is what is comparable.
        R["recur_fold"] = R.recurrence / (1 - HIGH_PCTL / 100)
        R["passes_gate"] = R.recur_fold >= args.min_fold
        rows.append(R)

        keep = R[~R.tissue_programme & R.passes_gate].sort_values(
            ["recurrence", "median_cos"], ascending=False)
        d = D[D.pair == pair]
        print(f"=== {pair} ===")
        if len(d):
            d = d.iloc[0]
            print(f"    strength {d.strength:.3f}  p_perm {d.p_perm:.4f}  "
                  f"combinations {len(vp)}  cohort pairs {int(d.n_cohort_pairs)}  "
                  f"tissue {tkey}"
                  + ("" if matched else "  [too few same-tissue pairs; "
                                       "fell back to the global background]"))
        print(f"    removed as tissue programme {int(R.tissue_programme.sum())}, "
              f"below {args.min_fold:g}x expected recurrence "
              f"{int((~R.passes_gate).sum())}, "
              f"{len(keep)} left")
        if np.isfinite(agr["agreement_top100"]):
            print(f"    split-half agreement top100 {agr['agreement_top100']:.3f} / "
                  f"top12 {agr['agreement_top12']:.3f}  "
                  f"({agr['agreement_k']} combinations per half)")
        print(f"    {'factor':<12}{'dir':>12}{'targets':>9}{'recur':>8}{'fold':>7}"
              f"{'bg pctl':>9}{'med cos':>9}")
        for _, r in keep.head(args.top_tfs).iterrows():
            print(f"    {r.tf:<12}"
                  f"{'concordant' if r.direction=='concordant' else 'discordant':>12}"
                  f"{r.n_targets:>5}{r.recurrence:>8.2f}{r.recur_fold:>6.1f}×"
                  f"{r.bg_pctl:>9.3f}{r.median_cos:>+9.3f}")
        print()

        if args.emit_agent:
            top = keep.head(args.top_tfs)
            # What did NOT survive. Two piles, and both belong in the evidence
            # package for reasons the surviving list cannot serve:
            #
            #   by_recurrence   TFs that look unusual on average — they clear the
            #                   background's 90th percentile — but fail the
            #                   recurrence gate, i.e. they were carried by some
            #                   patient subgroups and vanished in others. These
            #                   are exactly what a single-cohort analysis would
            #                   have reported, so naming them is what turns a
            #                   story from "these two share X" into "these two
            #                   share X rather than Y, though Y looks stronger in
            #                   part of the data". That contrast IS the claim
            #                   this view makes; without it every story reads as
            #                   if nothing was ever in competition.
            #
            #   tissue_prog     TFs the whole organ system shares anyway. Given
            #                   only the survivors, an agent asked about two
            #                   blood diseases will reach for SPI1 or IKZF1 from
            #                   its own knowledge and build a haematopoietic
            #                   story — the very thing the filter removed. Named
            #                   explicitly, it knows where the boundary is.
            #
            # "Would have been reported" is defined by doing it: draw single
            # views at random, rank the TFs inside each one alone, and take that
            # view's top 12. That is literally the answer a one-cohort study
            # would have published. Whatever appears there and not in the final
            # list is the concrete cost of not crossing views.
            #
            # An earlier definition — high average background percentile but
            # failing the recurrence gate — returned nothing at all, because a
            # TF whose median percentile clears 0.90 almost always clears the
            # bar in enough views to pass. The average and the survival rate are
            # not independent; a single view and the aggregate are.
            sv, kept = {}, set(keep.tf)
            for v in rng.choice(len(vp), min(3, len(vp)), replace=False):
                pv = np.full(len(Z), np.nan)
                for t in range(len(Z)):
                    if Z[t] is None or not np.isfinite(use[t]).any():
                        continue
                    nb = np.sort(use[t][np.isfinite(use[t])])
                    pv[t] = np.searchsorted(nb, C[t, v]) / len(nb)
                for t in np.argsort(np.nan_to_num(pv, nan=-1))[::-1][:args.top_tfs]:
                    if tfs[t] not in kept:
                        sv.setdefault(tfs[t], []).append(float(pv[t]))
            rr = dict(zip(R.tf, R.recurrence))
            near = pd.DataFrame([
                dict(tf=t, single_view_pctl=float(np.max(p)),
                     n_views_topped=len(p), recurrence=rr.get(t, np.nan))
                for t, p in sv.items()]).sort_values(
                    "single_view_pctl", ascending=False).head(args.top_tfs) \
                if sv else pd.DataFrame(
                    columns=["tf", "single_view_pctl", "n_views_topped",
                             "recurrence"])
            tp = R[R.tissue_programme].nlargest(args.top_tfs, "recurrence")
            prompts.append(dict(
                pair=pair, disease_a=da, disease_b=db, tissue_pairing=tkey,
                n_views=len(vp),
                n_cohort_pairs=int(d.n_cohort_pairs) if len(d) else None,
                strength=float(d.strength) if len(d) else None,
                readout_confidence=dict(
                    split_half_agreement_top100=None
                    if not np.isfinite(agr["agreement_top100"])
                    else round(agr["agreement_top100"], 3),
                    split_half_agreement_top12=None
                    if not np.isfinite(agr["agreement_top12"])
                    else round(agr["agreement_top12"], 3),
                    views_per_half=None if not np.isfinite(agr["agreement_k"])
                    else int(agr["agreement_k"]),
                    note="how far two group-disjoint halves give the same "
                         "factor ranking; "
                         "chance gives about 0.12 on the top 100. The higher it "
                         "is, the less the read-out depends on "
                         "which patients were sampled."),
                tfs=[dict(tf=r.tf, direction=r.direction,
                          n_targets=int(r.n_targets),
                          recurrence=round(r.recurrence, 2),
                          recurrence_fold_over_chance=round(r.recur_fold, 1),
                          bg_percentile=round(r.bg_pctl, 3),
                          targets=targets[list(tfs).index(r.tf)][:25])
                     for _, r in top.iterrows()],
                excluded=dict(
                    n_failed_recurrence=int((~R.passes_gate).sum()),
                    n_tissue_programme=int(R.tissue_programme.sum()),
                    single_view_would_report=[
                        dict(tf=r.tf,
                             best_single_view_percentile=round(r.single_view_pctl, 3),
                             n_of_3_views_topping=int(r.n_views_topped),
                             recurrence_across_all_views=round(r.recurrence, 2))
                        for _, r in near.iterrows()],
                    as_tissue_programme=[
                        dict(tf=r.tf, recurrence=round(r.recurrence, 2),
                             tissue_generic_percentile=(
                                 None if not np.isfinite(r.tissue_generic)
                                 else round(r.tissue_generic, 3)))
                        for _, r in tp.iterrows()],
                    note="single_view_would_report holds factors that led an "
                         "individual combination -- three were drawn at random "
                         "and ranked on their own -- "
                         "but did not survive aggregation over all of them; "
                         "these are what a single-cohort study would have "
                         "reported. An account should say "
                         "why the factors above survived and these did not. "
                         "as_tissue_programme holds factors generic to the "
                         "tissue (SPI1 and IKZF1 in blood, for instance), "
                         "already removed; do not build this pair's mechanism "
                         "on them.")))

    if not rows:
        print("no disease pairs to read out")
        return
    A = pd.concat(rows, ignore_index=True)
    path = f"{OUT}/mechanism_pairs_{args.net}.csv"
    if os.path.isfile(path):
        old = pd.read_csv(path)
        old = old[~old.pair.isin(A.pair.unique())]
        A = pd.concat([old, A], ignore_index=True)
    # a merged file can still carry alias pairs written by an older run
    A = A[~A.pair.isin(alias)].reset_index(drop=True)
    # generality is a property of the whole read-out, so it is computed once the
    # table is complete rather than pair by pair
    gen, n_read = generality(A, args.top_tfs)
    A["generality"] = A.tf.map(gen).fillna(0.0).round(3)
    for d in prompts:
        for t in d["tfs"]:
            t["generality"] = round(gen.get(t["tf"], 0.0), 3)
    A.to_csv(path, index=False)
    print(f"wrote mechanism_pairs_{args.net}.csv "
          f"({len(A)} rows / {A.pair.nunique()} disease pairs)")
    g = pd.Series(gen).sort_values(ascending=False)
    print(f"\ngenerality ({len(g)} factors reached the top {args.top_tfs} of "
          f"some pair, "
          f"over {n_read} pairs): median {g.median():.3f}, highest "
          + ", ".join(f"{t} {v:.2f}" for t, v in g.head(5).items()))

    K = A[~A.tissue_programme & A.passes_gate]
    rec = (K[K.recur_fold >= 4].groupby("tf")
           .agg(n_pairs=("pair", "nunique"), mean_rec=("recur_fold", "mean"))
           .sort_values("n_pairs", ascending=False))
    print(f"\nfactors recurring across many pairs (>=4x expected):")
    print("  " + ", ".join(f"{t}({int(r.n_pairs)} pairs)"
                           for t, r in rec.head(15).iterrows()))

    conf = (A.groupby("pair")
            .agg(views=("n_views", "first"),
                 top100=("agreement_top100", "first"),
                 top12=("agreement_top12", "first"))
            .dropna().sort_values("top100", ascending=False))
    if len(conf):
        print(f"\nsplit-half agreement, the confidence of a read-out "
              f"(chance is about 0.12): "
              f"median {conf.top100.median():.3f}  "
              f"range {conf.top100.min():.3f}-{conf.top100.max():.3f}")
        print(f"  {'the 6 most reproducible':<52}{'comb':>7}{'top100':>8}"
              f"{'top12':>7}")
        for p, r in conf.head(6).iterrows():
            print(f"  {p[:50]:<52}{int(r.views):>7}{r.top100:>8.3f}{r.top12:>7.3f}")
        print(f"  {'the 3 least reproducible':<52}")
        for p, r in conf.tail(3).iterrows():
            print(f"  {p[:50]:<52}{int(r.views):>7}{r.top100:>8.3f}{r.top12:>7.3f}")

    if args.emit_agent:
        # highest-confidence pairs first, so an agent working through the file
        # top-down spends its effort where the read-out is most stable
        prompts.sort(key=lambda d: -(d["readout_confidence"]
                                     ["split_half_agreement_top100"] or 0))
        p = f"{OUT}/agent_prompts_{args.net}.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for d in prompts:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"wrote {p} ({len(prompts)} disease pairs, ordered by the "
              f"confidence of their read-out, for the interpretation agents)")


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

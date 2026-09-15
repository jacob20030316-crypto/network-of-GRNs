"""Are the shared targets of a de novo pair the right genes for that factor?

The de novo route infers each disease's regulons from expression alone, and a
pair is carried forward on the genes two independently inferred regulons hold
in common. Nothing in that procedure consults a curated table, so whether those
genes are the ones the factor is actually known to act on is an open question,
and the answer is the main external check the route has.

Two references, neither of which was used to build anything:

    CollecTRI   the targets a curated resource records for that same factor
    ReMap       the genes that factor is observed bound to in ChIP experiments

**Why the null permutes factor identity rather than genes.** A shared target set
is not a random draw from the genome — it is a set of co-expressed genes, and
co-expressed genes are enriched for being co-regulated by *something*. Against a
random-gene null almost any set would look enriched. The question is narrower:
are these genes specific to *this* factor? So the null keeps the gene set and
changes whose reference it is checked against, which asks exactly that.

**Why the swap is size-matched.** A factor with two thousand recorded targets
overlaps any gene set more often than one with thirty, so an unmatched swap
would measure how large the reference sets are. Each factor is therefore
replaced only by factors whose reference set is of comparable size.

    python -m grn.tools.denovo.enrichment
    python -m grn.tools.denovo.enrichment --pair Cervical_Cancer||Head_and_Neck_Cancer
"""
import os
import sys

import numpy as np
import pandas as pd

from grn import paths as config

OUT = config.DENOVO_OUT

N_PERM = 10000
SIZE_BAND = 0.5        # a stand-in must have a reference set within 2x, either way
SEED = 0


def collectri_targets():
    """factor -> the targets CollecTRI records for it."""
    d = pd.read_csv(config.COLLECTRI, sep="\t",
                    usecols=["source_genesymbol", "target_genesymbol"]).dropna()
    d = d[~d.source_genesymbol.str.contains("_", na=False)]
    return {k: frozenset(v.str.upper())
            for k, v in d.groupby(d.source_genesymbol.str.upper()).target_genesymbol}


def remap_targets():
    """factor -> the genes it is observed bound to, merged over ReMap experiments."""
    path = os.path.join(config.EXTERNAL, "chea3", "ReMap_ChIP-seq.gmt")
    if not os.path.isfile(path):
        raise SystemExit(f"no ReMap library at {path}")
    d = {}
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) > 2:
            d.setdefault(f[0].split("_")[0].upper(), set()).update(
                g.upper() for g in f[2:] if g)
    return {k: frozenset(v) for k, v in d.items()}


def carried(path=None):
    """The factor entries the pair-level selection carried forward."""
    D = pd.read_csv(path or os.path.join(OUT, "denovo_pair_factors_all.csv"))
    if "carried" in D.columns:
        S = D[D.carried.astype(bool)]
        if len(S):
            return S.reset_index(drop=True)
    S = D[D.q_within < 0.05]
    keep = S.groupby("pair").size()
    keep = set(keep[keep >= 3].index)
    return S[S.pair.isin(keep)].reset_index(drop=True)


def _rate(entries, ref, assign):
    """Fraction of shared targets that fall in the reference set of the factor
    each entry is assigned to. Pooled over entries, so a pair contributing many
    factors weighs more than one contributing three — the same way the read-out
    itself is weighted."""
    hit = tot = 0
    for (targets, _), t in zip(entries, assign):
        r = ref.get(t)
        if r is None:
            continue
        hit += len(targets & r)
        tot += len(targets)
    return hit / tot if tot else np.nan


def test(entries, ref, rng, n_perm=N_PERM, band=SIZE_BAND):
    """Observed rate, its size-matched permutation null, fold and p."""
    usable = [(s, t) for s, t in entries if t in ref]
    if not usable:
        return None
    sizes = {t: len(v) for t, v in ref.items()}
    pool = sorted(ref)
    pool_sizes = np.array([sizes[t] for t in pool], dtype=float)

    # for each factor, the stand-ins whose reference set is of similar size
    stand_ins = {}
    for _, t in usable:
        n = sizes[t]
        lo, hi = n * (1 - band), n * (1 + band) / (1 - band)
        idx = np.where((pool_sizes >= lo) & (pool_sizes <= hi))[0]
        idx = np.array([i for i in idx if pool[i] != t])
        if len(idx) < 5:                       # too rare a size to match on
            order = np.argsort(np.abs(np.log1p(pool_sizes) - np.log1p(n)))
            idx = np.array([i for i in order if pool[i] != t][:25])
        stand_ins[t] = idx

    obs = _rate(usable, ref, [t for _, t in usable])
    null = np.empty(n_perm)
    for b in range(n_perm):
        swap = [pool[rng.choice(stand_ins[t])] for _, t in usable]
        null[b] = _rate(usable, ref, swap)
    m = float(np.nanmean(null))
    return {"n_entries": len(usable),
            "n_shared_targets": int(sum(len(s) for s, _ in usable)),
            "observed": obs, "null_mean": m,
            "fold": obs / m if m else np.nan,
            "p": (1 + int((null >= obs).sum())) / (1 + n_perm)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", default=None, help="report one pair on its own")
    ap.add_argument("--perms", type=int, default=N_PERM)
    ap.add_argument("--band", type=float, default=SIZE_BAND)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    S = carried()
    print(f"{S.pair.nunique()} disease pairs, {len(S)} factor entries, "
          f"{S.tf.nunique()} distinct factors", flush=True)

    refs = {"CollecTRI": collectri_targets(), "ReMap": remap_targets()}
    for name, r in refs.items():
        print(f"  {name}: {len(r)} factors, "
              f"median {int(np.median([len(v) for v in r.values()]))} targets each")

    def entries_of(df):
        out = []
        for _, row in df.iterrows():
            g = frozenset(x.upper() for x in str(row.shared_targets).split("|") if x)
            if g:
                out.append((g, str(row.tf).upper()))
        return out

    rows = []
    groups = [("all pairs", S)]
    if args.pair:
        groups.append((args.pair, S[S.pair == args.pair]))
    else:
        for p in sorted(S.pair.unique()):
            groups.append((p, S[S.pair == p]))

    for label, df in groups:
        ent = entries_of(df)
        for name, ref in refs.items():
            rng = np.random.default_rng(args.seed)
            res = test(ent, ref, rng, args.perms, args.band)
            if res is None:
                continue
            rows.append({"group": label, "reference": name, **res})

    R = pd.DataFrame(rows)
    path = os.path.join(OUT, "denovo_shared_target_enrichment.csv")
    R.to_csv(path, index=False)

    print(f"\n{'group':<52} {'reference':<10} {'entries':>7} {'targets':>8} "
          f"{'obs':>7} {'null':>7} {'fold':>6} {'p':>9}")
    for _, r in R.iterrows():
        print(f"{r.group[:52]:<52} {r.reference:<10} {r.n_entries:>7} "
              f"{r.n_shared_targets:>8} {r.observed:>7.3f} {r.null_mean:>7.3f} "
              f"{r.fold:>6.2f} {r.p:>9.5f}")
    print(f"\nwrote {path}")
    return 0


def run(**kw):
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

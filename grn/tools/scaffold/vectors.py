"""B-gate step 1: build a CollecTRI edge-weight vector for every usable cohort.

Edge weight = correlation between the TF's and the target's expression across the
disease-group samples of that cohort. Every cohort gets a value for every edge in
a common edge universe, so the vectors are directly comparable and their sparsity
does not vary with sample size — the failure mode that killed the de novo route.

Two passes:
  A. read headers only, decide the common edge universe (both endpoints measured
     in >=90% of cohorts) — without this, edge availability varies per cohort and
     the coverage confound comes straight back
  B. read the needed genes, compute the vectors

The 90% prevalence rule bounds but does not remove the coverage confound: an
unmeasured or invariant gene still yields a zero edge weight, which is
indistinguishable from "measured, no correlation". So pass B also writes
`A_avail.npy` — a cohort x edge boolean saying which weights are real estimates.
03 uses it both to test whether coverage predicts similarity and to offer a
core-edge variant restricted to edges estimable everywhere.

Outputs: edge_universe.csv, W_full.npy, A_avail.npy, W_meta.csv
"""
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

from grn import paths as config                                  # noqa: E402

OUT = config.SCAFFOLD_OUT
os.makedirs(OUT, exist_ok=True)
TBL = config.COLLECTRI
PREVALENCE = 0.90
MIN_SYMBOL_HITS = 100      # below this the matrix is not gene-symbol indexed

os.makedirs(OUT, exist_ok=True)


def collectri():
    d = pd.read_csv(TBL, sep="\t",
                    usecols=["source_genesymbol", "target_genesymbol",
                             "is_stimulation", "is_inhibition"])
    d = d.dropna(subset=["source_genesymbol", "target_genesymbol"])
    d = d[~d.source_genesymbol.str.contains("_", na=False)]      # drop COMPLEX:
    d = d[d.source_genesymbol != d.target_genesymbol]            # drop self-loops
    d = d.drop_duplicates(["source_genesymbol", "target_genesymbol"])
    return d.reset_index(drop=True)


def case_mask(path, selector):
    """Row index of the disease-group samples."""
    sub = pd.read_csv(path, usecols=[0, 1], index_col=0)
    s = str(selector)
    if s == "trait==1":
        return set(sub.index[pd.to_numeric(sub.iloc[:, 0], errors="coerce") == 1])
    if s == "trait==0":
        return set(sub.index[pd.to_numeric(sub.iloc[:, 0], errors="coerce") == 0])
    return set(sub.index)


def main():
    E = collectri()
    print(f"CollecTRI: {len(E)} interactions, {E.source_genesymbol.nunique()} factors, "
          f"{E.target_genesymbol.nunique()} targets", flush=True)

    # config resolves `path` against input_matrices/, so the run stays inside
    # the project instead of reaching back to wherever preprocessing wrote
    U = config.verdicts(usable_only=True)
    print(f"{len(U)} admitted cohorts / {U.disease.nunique()} diseases", flush=True)

    # ---------- pass A: gene presence ----------
    t0 = time.time()
    all_symbols = set(E.source_genesymbol) | set(E.target_genesymbol)
    present, unmapped = {}, []
    for i, r in U.iterrows():
        try:
            with open(r.path, encoding="utf-8", newline="") as fh:
                cols = set(next(csv.reader(fh))[1:])
        except Exception:
            continue
        # A matrix whose gene columns never got mapped to symbols (Entrez ids,
        # Affymetrix probe ids) overlaps CollecTRI at zero and would contribute
        # an all-zero vector. Worse, it drags every gene's prevalence down and
        # so silently shrinks the edge universe for everyone. Adjudication
        # cannot catch this — it reads metadata, not column names.
        if len(cols & all_symbols) < MIN_SYMBOL_HITS:
            unmapped.append((r.disease, r.cohort,
                             sorted(cols)[:3], len(cols & all_symbols)))
            continue
        present[r.path] = cols
        if (i + 1) % 50 == 0:
            print(f"  [A] {i+1}/{len(U)}  ({time.time()-t0:.0f}s)", flush=True)

    if unmapped:
        print(f"\n{len(unmapped)} cohorts dropped: gene identifiers were never "
              f"mapped to symbols")
        for d, c, sample, hit in unmapped:
            print(f"  {d}/{c}  {hit} symbols matched, columns look like {sample}")

    U = U[U.path.isin(present)].reset_index(drop=True)
    genes = sorted(set(E.source_genesymbol) | set(E.target_genesymbol))
    cnt = pd.Series({g: sum(1 for c in present.values() if g in c) for g in genes})
    frac = cnt / len(present)
    keep_genes = set(frac.index[frac >= PREVALENCE])
    print(f"\ngenes measured in >= {PREVALENCE:.0%} of cohorts: "
          f"{len(keep_genes)} / {len(genes)}")

    Ek = E[E.source_genesymbol.isin(keep_genes) & E.target_genesymbol.isin(keep_genes)]
    Ek = Ek.reset_index(drop=True)
    print(f"common interaction set: {len(Ek)}, "
          f"{Ek.source_genesymbol.nunique()} factors, "
          f"{Ek.target_genesymbol.nunique()} targets", flush=True)
    Ek.to_csv(f"{OUT}/edge_universe.csv", index=False)

    need = sorted(set(Ek.source_genesymbol) | set(Ek.target_genesymbol))
    need_set = set(need)
    gpos = {g: j for j, g in enumerate(need)}
    src = Ek.source_genesymbol.map(gpos).values
    tgt = Ek.target_genesymbol.map(gpos).values

    # ---------- pass B: one vector per analysis unit ----------
    # Units come from data/analysis_units.csv: the unconditioned disease groups
    # plus every conditioning level inside them. Several units share a cohort,
    # so the matrix is read once and every unit of that cohort is built from it.
    UNITS = pd.read_csv(config.ANALYSIS_UNITS)
    UNITS = UNITS[UNITS.apply(
        lambda u: config.matrix_path(u.disease, u.cohort) in present, axis=1)]
    by_cohort = UNITS.groupby(["disease", "cohort"])
    print(f"{len(UNITS)} patient groups (pooled "
          f"{int((UNITS.network=='uncond').sum())} / divided "
          f"{int((UNITS.network=='cond').sum())}), from {by_cohort.ngroups} cohorts",
          flush=True)

    t0 = time.time()
    W, A, meta = [], [], []
    for k, ((disease, cohort), grp) in enumerate(by_cohort):
        path = config.matrix_path(disease, cohort)
        cols = present[path]
        use = [c for c in need if c in cols]
        try:
            with open(path, encoding="utf-8", newline="") as fh:
                hdr = next(csv.reader(fh))
            # the index column often has an empty name, so select by position
            use_pos = [0] + [j for j, c in enumerate(hdr) if c in need_set]
            df = pd.read_csv(path, index_col=0, usecols=use_pos)
        except Exception as e:
            print(f"  ! {disease}/{cohort}: {e}", flush=True)
            continue

        full = np.full((len(df), len(need)), np.nan, dtype=np.float32)
        for c in df.columns:
            j = gpos.get(c)
            if j is not None:
                full[:, j] = pd.to_numeric(df[c], errors="coerce").values
        pos = {s: i for i, s in enumerate(df.index.astype(str))}

        for _, u in grp.iterrows():
            rows = [pos[s] for s in str(u.samples).split("|") if s in pos]
            if len(rows) < 8:
                continue
            X = full[rows]
            # standardise each gene; genes with no variance -> all-zero column
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            sd[~np.isfinite(sd) | (sd < 1e-9)] = np.nan
            # a gene is *estimable* here iff it was measured and actually
            # varies; both failure modes end up as a zero column below, which is
            # indistinguishable from "measured, no correlation" unless recorded
            estimable = np.isfinite(sd)
            Z = np.nan_to_num((X - mu) / sd, nan=0.0)
            w = (Z[:, src] * Z[:, tgt]).mean(axis=0)   # per-edge correlation
            avail = estimable[src] & estimable[tgt]    # edge is a real estimate

            W.append(w.astype(np.float32))
            A.append(avail)
            meta.append(dict(nid=u.unit_id, disease=disease, cohort=cohort,
                             source=u.source, network=u.network, axis=u.axis,
                             level=u.level, n=len(rows),
                             n_genes_present=len(use),
                             n_genes_estimable=int(estimable.sum()),
                             n_edges_avail=int(avail.sum()),
                             frac_edges_avail=float(avail.mean())))
        if (k + 1) % 25 == 0:
            print(f"  [B] {k+1}/{by_cohort.ngroups} cohorts  {len(W)} groups done  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    Wm = np.vstack(W)
    Am = np.vstack(A)
    M = pd.DataFrame(meta)
    np.save(f"{OUT}/W_full.npy", Wm)
    np.save(f"{OUT}/A_avail.npy", Am)
    M.to_csv(f"{OUT}/W_meta.csv", index=False)
    print(f"\ndone: {Wm.shape[0]} groups x {Wm.shape[1]} interactions  "
          f"({time.time()-t0:.0f}s)")
    print(f"group size: median {M.n.median():.0f}  range {M.n.min()}-{M.n.max()}")
    print(f"fraction of interactions estimable: "
          f"median {M.frac_edges_avail.median():.3f}  "
          f"range {M.frac_edges_avail.min():.3f}-{M.frac_edges_avail.max():.3f}")
    for net, g in M.groupby("network"):
        print(f"  {net:<8} groups {len(g):>4}  diseases {g.disease.nunique():>3}  "
              f"median size {g.n.median():.0f}")
    core = Am.mean(axis=0) >= 0.99
    print(f"interactions estimable in >=99% of cohorts: "
          f"{int(core.sum())} / {Am.shape[1]} "
          f"(the core-interaction variant uses only these)")


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

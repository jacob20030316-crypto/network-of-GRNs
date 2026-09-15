"""B-gate step 2: how reproducible is one analysis unit's edge-weight vector?

For a given sample size m, draw two DISJOINT subsets of m samples from the same
unit, build an edge vector from each, and correlate them. That correlation is
the reliability of a size-m vector — the ceiling on any similarity we could ever
measure between two different diseases at that sample size.

Also records the per-unit split-half reliability, which is what the
disattenuation correction (03/04 step ④) needs. This matters more now than it
did at cohort level: the conditioned network's units are smaller (floor 30 vs
50), so they are systematically noisier, and without disattenuation their edges
would look weak for a purely technical reason.

Units come from data/analysis_units.csv, so a cohort's matrix is read once and
every unit built from it is measured in the same pass.
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
MS = [10, 15, 20, 30, 50, 100, 200]
REPEATS = 3
SEED = 0


def main():
    E = pd.read_csv(f"{OUT}/edge_universe.csv")
    M = pd.read_csv(f"{OUT}/W_meta.csv")
    U = pd.read_csv(config.ANALYSIS_UNITS)
    U = U[U.unit_id.isin(set(M.nid))]
    print(f"{len(U)} patient groups, from "
          f"{U.groupby(['disease','cohort']).ngroups} cohorts", flush=True)

    need = sorted(set(E.source_genesymbol) | set(E.target_genesymbol))
    need_set = set(need)
    gpos = {g: j for j, g in enumerate(need)}
    src = E.source_genesymbol.map(gpos).values
    tgt = E.target_genesymbol.map(gpos).values

    def vec(X):
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-9)] = np.nan
        Z = np.nan_to_num((X - mu) / sd, nan=0.0)
        return (Z[:, src] * Z[:, tgt]).mean(axis=0)

    rng = np.random.default_rng(SEED)
    rows, half = [], []
    t0 = time.time()
    for k, ((disease, cohort), grp) in enumerate(U.groupby(["disease", "cohort"])):
        path = config.matrix_path(disease, cohort)
        try:
            with open(path, encoding="utf-8", newline="") as fh:
                hdr = next(csv.reader(fh))
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
            idx = [pos[s] for s in str(u.samples).split("|") if s in pos]
            n = len(idx)
            if n < 16:
                continue
            X = full[idx]

            rs = []
            for _ in range(REPEATS):
                p = rng.permutation(n)
                a, b = p[: n // 2], p[n // 2: 2 * (n // 2)]
                va, vb = vec(X[a]), vec(X[b])
                ok = np.isfinite(va) & np.isfinite(vb)
                if ok.sum() > 1000:
                    rs.append(float(np.corrcoef(va[ok], vb[ok])[0, 1]))
            if rs:
                half.append(dict(nid=u.unit_id, network=u.network, n=n,
                                 half_n=n // 2, r_half=float(np.mean(rs))))

            for m in MS:
                if n < 2 * m:
                    continue
                rs = []
                for _ in range(REPEATS):
                    p = rng.permutation(n)
                    va, vb = vec(X[p[:m]]), vec(X[p[m:2 * m]])
                    ok = np.isfinite(va) & np.isfinite(vb)
                    if ok.sum() > 1000:
                        rs.append(float(np.corrcoef(va[ok], vb[ok])[0, 1]))
                if rs:
                    rows.append(dict(nid=u.unit_id, disease=disease,
                                     source=u.source, network=u.network,
                                     n=n, m=m, r=float(np.mean(rs))))
        if (k + 1) % 25 == 0:
            print(f"  {k+1} cohorts  {len(half)} groups measured  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    R = pd.DataFrame(rows)
    H = pd.DataFrame(half)
    R.to_csv(f"{OUT}/reliability_by_m.csv", index=False)
    H.to_csv(f"{OUT}/reliability_halfsplit.csv", index=False)

    print(f"\n=== reliability against sample size ===")
    print("  the correlation between edge weights estimated in two disjoint halves")
    print(f"{'m':>5} {'groups':>8} {'median r':>10} {'25%':>8} {'75%':>8}")
    for m, g in R.groupby("m"):
        print(f"{m:>5} {len(g):>7} {g.r.median():>9.3f} {g.r.quantile(.25):>8.3f} "
              f"{g.r.quantile(.75):>8.3f}")

    print(f"\n=== split-half reliability, used to disattenuate ===")
    for net, g in H.groupby("network"):
        print(f"  {net:<8} {len(g):>4} groups   median r={g.r_half.median():.3f}  "
              f"range {g.r_half.min():.3f}-{g.r_half.max():.3f}")
    H["bin"] = pd.cut(H.half_n, [0, 15, 25, 50, 100, 1e9],
                      labels=["<=15", "16-25", "26-50", "51-100", ">100"])
    print(H.groupby("bin", observed=True).r_half.agg(["median", "count"])
          .round(3).to_string())
    print("\nGroups of the conditioned network are smaller, so their reliability\n"
          "is expected to be lower; that is what the disattenuation is for.")


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

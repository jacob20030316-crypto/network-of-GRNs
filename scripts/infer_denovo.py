"""De novo GRN inference: GRNBoost2 + motif pruning, one network per unit.

The de novo route exists because the scaffold cannot discover regulation that is
not already in CollecTRI. This one can. The price is that everything about a
searched network — how many edges survive, how many regulons come out — moves
with sample size, which is the confound that made the earlier attempt
uninterpretable. Two fixes, both mandatory:

**Uniform downsampling.** Every unit is cut to exactly M samples before
inference, not merely required to clear a floor. A floor still leaves a ten-fold
spread above it, so "how many edges were found" stays a function of sample size.
Measured on the earlier run: regulon counts ran 21 to 401 across units.

**A fixed regulon count, applied afterwards.** Also measured on the earlier run:
a median of 319 regulons per unit against a TF universe of only 1,526 means each
unit claims a fifth of all TFs, so any two unrelated diseases overlap by a third
(Jaccard 0.328) and the three tiers — same cohort, same disease across cohorts,
different disease — sit at 2.56 / 2.68 / 2.45 with no gradient at all. The cut is
NOT applied here: pruning emits every regulon with its NES, so top-N selection is
one sort downstream and every N can be swept from a single inference run.

Scope: unconditioned units at or above the floor. The conditioned network cannot
be used — at n>=100 it retains only 9 cohorts.

**Parallelism:**
an outer process pool over units, each worker running its own small dask cluster,
rather than one big cluster processing units serially. GRNBoost2 does not scale
linearly with workers — measured here, a single unit on 44 workers took 17.8 min
wall for 212 min CPU — so N_OUTER units x N_INNER workers finishes far more work
per hour. The outer pool must be non-daemonic because each worker spawns dask
nannies, hence NestablePool, taken from that same file.

The module parameters are also taken from it (`top_n_targets=(49,)` only,
`keep_only_activating=True`, `min_genes=20`) rather than pySCENIC's defaults.
The defaults emit six module variants per TF and would multiply pruning cost by
six; keeping the earlier run's settings also makes the two comparable.

Resumable: a unit whose regulon table exists is skipped, and a cached adjacency
table is reused, so a run that died in pruning does not re-pay for GRNBoost2.

    python 01_infer.py --pilot            # the 9 two-cohort diseases, 18 units
    python 01_infer.py                    # all 48 units at the floor
    nohup python -u 01_infer.py --pilot > pilot.log 2>&1 &
"""
import argparse
import glob
import multiprocessing as mp
import multiprocessing.pool
import os
import re
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(OUT, exist_ok=True)
from grn import paths as config                                  # noqa: E402

OUT = config.DENOVO_OUT

FLOOR = 100          # unit sample-size floor
M = 100              # every unit is cut to exactly this many samples
SEED = 0
MIN_GENES = 8000
MIN_TF_HITS = 200


# --- NestablePool: outer workers must be able to spawn dask nannies ---------
_SPAWN = mp.get_context("spawn")


class _NoDaemonProcess(_SPAWN.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class _NoDaemonContext(type(_SPAWN)):
    Process = _NoDaemonProcess


class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *a, **kw):
        kw.setdefault("context", _NoDaemonContext())
        super().__init__(*a, **kw)


def gene_columns(cols):
    """Gene columns = the longest alphabetically sorted suffix of the header.

    The matrices carry a trait column and a few clinical columns before the gene
    block, and the gene block is emitted sorted. Checking sortedness finds the
    boundary without hard-coding which clinical columns exist — 'Age' and
    'Gender' sit before 'A1BG' positionally but after it alphabetically, so the
    suffix rule places the boundary correctly either way.
    """
    k = len(cols)
    while k > 1 and cols[k - 2] <= cols[k - 1]:
        k -= 1
    return list(cols[k - 1:])


def load_unit(disease, cohort, samples, unit_id, m, rep=0):
    path = config.matrix_path(disease, cohort)
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df = df[gene_columns(list(df.columns))]
    df.index = df.index.astype(str)
    X = df.loc[[s for s in samples if s in df.index]]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().all(axis=0) & (X.std(axis=0) > 1e-9)]
    # seeded on (unit, repeat) so a re-run draws the same samples. NOT `hash()`:
    # Python randomises string hashing per process, so the draw would differ
    # every run and a cached adjacency table would stop matching its matrix.
    rng = np.random.default_rng(zlib.crc32(unit_id.encode()) ^ SEED ^ (rep << 20))
    if len(X) > m:
        X = X.iloc[np.sort(rng.choice(len(X), m, replace=False))]
    return X


def aggregate(unit_id, repeats, majority=None):
    """Merge a unit's repeated draws into one regulon table.

    A single draw of M samples throws away the rest of a large cohort and gives
    no handle on how much of its network is the draw rather than the disease.
    Repeating the draw fixes both: sample size stays exactly M, more of the
    cohort is used, and how often a regulon survives across draws becomes a
    per-unit stability estimate that comes free.

    The merge is a majority vote, not a union. A regulon appearing in one draw
    of three is a property of those 100 patients, not of the disease, and a
    union would keep every such artefact while inflating the set size. Kept
    fields: NES as the median over the draws that found it, targets as the genes
    seen in a majority of those draws, and `frequency` — the vote itself, which
    is the stability estimate and should be reported, not silently used.

    Note the merged table is NOT comparable to a single-draw one: the vote is
    itself a denoising step, so a k=3 unit and a k=1 unit have had different
    amounts of filtering. Do not mix them in one analysis.
    """
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", unit_id)
    majority = majority or (repeats // 2 + 1)
    parts = []
    for r in range(repeats):
        p = f"{OUT}/{tag}__r{r}.regulons.tsv"
        if os.path.isfile(p):
            d = pd.read_csv(p, sep="\t")
            if len(d):
                parts.append(d.assign(rep=r))
    if len(parts) < majority:
        return None
    A = pd.concat(parts, ignore_index=True)
    rows = []
    for tf, g in A.groupby("tf"):
        if len(g) < majority:
            continue
        cnt = {}
        for s in g.targets:
            for t in str(s).split("|"):
                if t:
                    cnt[t] = cnt.get(t, 0) + 1
        keep = sorted(t for t, c in cnt.items() if c >= majority)
        if not keep:
            continue
        rows.append(dict(tf=tf, nes=float(np.median(g.nes)), n_targets=len(keep),
                         targets="|".join(keep), frequency=len(g) / repeats,
                         n_draws=len(parts)))
    if not rows:
        return None
    R = pd.DataFrame(rows).sort_values("nes", ascending=False)
    R.to_csv(f"{OUT}/{tag}.regulons.tsv", sep="\t", index=False)
    return R


def fingerprint(X, m):
    """Exactly which samples and genes went into this unit's inference.

    Every cached artefact is written beside one of these and re-checked before
    reuse. Without it a cache silently outlives the code that produced it: the
    first run here seeded the downsample with `hash()`, which Python randomises
    per process, so after that was changed to crc32 the cached adjacency table
    came from one draw of 100 patients while the matrix loaded next to it came
    from another. Nothing errors — modules_from_adjacencies happily accepts the
    mismatch — and the regulons are quietly wrong. A cache must prove it matches
    or be thrown away.
    """
    return (f"m={m}\nn_genes={X.shape[1]}\n"
            + "\n".join(sorted(X.index.astype(str))) + "\n")


def cache_ok(path, fp):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read() == fp
    except OSError:
        return False


def infer_one(job):
    """One draw of one unit, in its own process with its own dask cluster."""
    u, m, n_inner = job[:3]
    rep = job[3] if len(job) > 3 else 0
    suffix = f"__r{rep}" if len(job) > 3 else ""
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", u["unit_id"]) + suffix
    f_adj = f"{OUT}/{tag}.adjacencies.tsv.gz"
    f_reg = f"{OUT}/{tag}.regulons.tsv"
    f_fp = f"{OUT}/{tag}.samples.txt"
    t0 = time.time()
    cluster = client = None
    try:
        from distributed import Client, LocalCluster
        from arboreto.algo import grnboost2
        from ctxcore.rnkdb import FeatherRankingDatabase
        from pyscenic.utils import modules_from_adjacencies
        from pyscenic.prune import prune2df, df2regulons

        X = load_unit(u["disease"], u["cohort"], u["samples"], u["unit_id"], m, rep)
        fp = fingerprint(X, m)
        # the finished-unit check is fingerprinted too: otherwise a regulon table
        # built from a different draw is skipped as "done" forever
        if os.path.isfile(f_reg):
            if cache_ok(f_fp, fp):
                return dict(unit_id=u["unit_id"], rep=rep, status="skipped")
            print(f"    ! {u['unit_id']} has results but the sample fingerprint "
                  f"does not match; recomputing",
                  flush=True)
            for p in (f_reg, f_adj):
                if os.path.isfile(p):
                    os.remove(p)

        tfs = [l.strip() for l in open(config.TF_LIST) if l.strip()]
        present = [t for t in tfs if t in X.columns]
        if X.shape[1] < MIN_GENES or len(present) < MIN_TF_HITS:
            return dict(unit_id=u["unit_id"], rep=rep, status="too_few_genes",
                        n_genes=X.shape[1], n_tf_present=len(present))

        cluster = LocalCluster(n_workers=n_inner, threads_per_worker=1,
                               processes=True, dashboard_address=None,
                               memory_limit="auto")
        client = Client(cluster)

        t1 = time.time()
        if os.path.isfile(f_adj) and cache_ok(f_fp, fp):
            adj = pd.read_csv(f_adj, sep="\t")
            t_grn = 0.0
        else:
            adj = grnboost2(expression_data=X, tf_names=present,
                            client_or_address=client, seed=SEED, verbose=False)
            adj.to_csv(f_adj, sep="\t", index=False, compression="gzip")
            # written only after the table it describes, so an interrupted write
            # leaves an unfingerprinted table that the next run recomputes
            with open(f_fp, "w", encoding="utf-8") as fh:
                fh.write(fp)
            t_grn = time.time() - t1

        t2 = time.time()
        # prune2df takes MODULES, not the raw adjacency table — passing the
        # table is what made the first attempt die inside boltons' chunked_iter
        # with "truth value of a DataFrame is ambiguous".
        mods = list(modules_from_adjacencies(
            adj, X, thresholds=(), top_n_targets=(49,), top_n_regulators=(),
            keep_only_activating=True, min_genes=20))
        dbs = [FeatherRankingDatabase(fname=p, name=os.path.splitext(
            os.path.basename(p))[0]) for p in sorted(glob.glob(
                config.MOTIF_RANKINGS_GLOB))]
        df = prune2df(dbs, mods, config.MOTIF_ANNOTATIONS)
        regs = df2regulons(df)
        t_prune = time.time() - t2

        R = pd.DataFrame([dict(tf=r.transcription_factor,
                               nes=float(getattr(r, "score", np.nan)),
                               n_targets=len(r.genes),
                               targets="|".join(sorted(r.genes)))
                          for r in regs])
        # NES per TF from the pruning table — that is where it lives;
        # Regulon.score is only the max over that regulon's own contexts
        try:
            R["nes"] = R.tf.map(
                df[("Enrichment", "NES")].groupby(level="TF").max()).fillna(R.nes)
        except Exception:
            pass
        R.sort_values("nes", ascending=False).to_csv(f_reg, sep="\t", index=False)
        return dict(unit_id=u["unit_id"], rep=rep, disease=u["disease"],
                    cohort=u["cohort"], status="ok", n_orig=u["n"],
                    n_used=X.shape[0], n_genes=X.shape[1],
                    n_tf_present=len(present), n_adjacency=len(adj),
                    n_module=len(mods), n_regulon=len(R),
                    sec_grnboost=t_grn, sec_prune=t_prune,
                    sec_total=time.time() - t0)
    except Exception as e:
        import traceback
        return dict(unit_id=u["unit_id"], rep=rep, status="error",
                    error=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc()[-1500:])
    finally:
        try:
            if client is not None:
                client.close()
            if cluster is not None:
                cluster.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=int, default=FLOOR)
    ap.add_argument("-m", type=int, default=M)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="only diseases with two cohorts at the floor — the units "
                         "that can test whether de novo recognises a disease")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent downsampled draws per unit; >1 turns on the\n                          majority-vote merge and yields a per-unit stability")
    ap.add_argument("--n-outer", type=int, default=6, help="units in parallel")
    ap.add_argument("--n-inner", type=int, default=6, help="dask workers per unit")
    args = ap.parse_args()

    U = pd.read_csv(config.ANALYSIS_UNITS)
    W = pd.read_csv(os.path.join(config.SCAFFOLD_OUT, "W_meta.csv"))
    U = U[U.unit_id.isin(W.nid) & (U.network == "uncond") & (U.n >= args.floor)]
    if args.pilot:
        nc = U.groupby("disease").cohort.nunique()
        U = U[U.disease.isin(nc.index[nc >= 2])]
    U = U.sort_values(["disease", "cohort"]).reset_index(drop=True)
    if args.limit:
        U = U.head(args.limit)
    print(f"{len(U)} groups to infer / {U.disease.nunique()} diseases / "
          f"{U.cohort.nunique()} cohorts   draw m={args.m}   "
          f"{args.repeats} draws   "
          f"{args.n_outer} x {args.n_inner} workers "
          f"= {args.n_outer*args.n_inner} cores", flush=True)

    units = [{k: (str(r.samples).split("|") if k == "samples" else r[k])
               for k in ("unit_id", "disease", "cohort", "samples", "n")}
             for _, r in U.iterrows()]
    jobs = ([(u, args.m, args.n_inner) for u in units] if args.repeats == 1
            else [(u, args.m, args.n_inner, rep)
                  for rep in range(args.repeats) for u in units])

    t0 = time.time()
    done = []
    with NestablePool(args.n_outer) as pool:
        for i, res in enumerate(pool.imap_unordered(infer_one, jobs), 1):
            done.append(res)
            el = (time.time() - t0) / 60
            if res["status"] == "ok":
                print(f"[{i}/{len(jobs)}] ✓ {res['unit_id']}  "
                      f"regulon {res['n_regulon']}  module {res['n_module']}  "
                      f"GRN {res['sec_grnboost']/60:.1f}m + pruning "
                      f"{res['sec_prune']/60:.1f}m   elapsed {el:.0f}m", flush=True)
            elif res["status"] == "error":
                print(f"[{i}/{len(jobs)}] ✗ {res['unit_id']}  {res['error']}\n"
                      f"{res.get('tb','')}", flush=True)
            else:
                print(f"[{i}/{len(jobs)}] – {res['unit_id']}  {res['status']}",
                      flush=True)
            pd.DataFrame(done).to_csv(f"{OUT}/inference_log.csv", index=False)

    if args.repeats > 1:
        print(f"\n=== merging {args.repeats} draws by majority vote ===", flush=True)
        nagg = 0
        for u in units:
            R = aggregate(u["unit_id"], args.repeats)
            if R is not None:
                nagg += 1
                print(f"  {u['unit_id'][:56]:<58} {len(R):>4} regulon  "
                      f"at frequency 1.0: {int((R.frequency >= 0.999).sum()):>4}",
                      flush=True)
        print(f"  merged regulons for {nagg} / {len(units)} groups")

    L = pd.DataFrame(done)
    ok = L[L.status == "ok"]
    print(f"\ndone {len(ok)} / {len(jobs)}   total {(time.time()-t0)/60:.0f} min")
    if len(ok):
        print(f"  GRNBoost2 median {ok.sec_grnboost.median()/60:.1f} min, "
              f"pruning median {ok.sec_prune.median()/60:.1f} min")
        print(f"  regulons: median {ok.n_regulon.median():.0f}  "
              f"range {ok.n_regulon.min()}-{ok.n_regulon.max()}"
              f"   -- the fixed top-N is what removes this spread")


if __name__ == "__main__":
    sys.exit(main())

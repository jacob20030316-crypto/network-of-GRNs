"""The de novo network's confound checks, on the same footing as the scaffold's.

The scaffold route fixes sparsity by never selecting edges: every node scores
the same catalogue, so how much data a node was estimated from cannot change
how many relationships it carries. The de novo route infers its edges, where
that guarantee does not hold, and fixes the same confound differently — a fixed
draw of a hundred samples per node, three times, and a fixed number of regulons
kept per node.

Whether that worked is a question with a number, and until now only the
scaffold had one. These are the same checks, asked of this network:

  similarity_vs_sample_size   does node similarity track how many patients the
                              cohorts held? The fixed draw should make it not.
  similarity_vs_regulon_count does it track how many regulons were recovered?
                              The fixed count should make it not.
  similarity_vs_gene_coverage does it track how many genes the cohorts
                              measured?
  worst_source_block_offset   the offset of the worst source block — TCGA-TCGA,
                              GEO-GEO or the two crossed — from the overall
                              cross-disease median.

    python -m grn.tools.denovo.acceptance
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from grn import paths as config
from grn.tools.denovo import pair_network as pn

OUT = config.READOUT
GATE = 0.15
METHOD = "targ_nes_shared"        # the definition the route builds on


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", type=float, default=GATE)
    args = ap.parse_args(argv)

    sets, M = pn.load_regulons()
    uids = list(M.index)
    S = pn.matrix(sets, uids, pn.sim_targ_nes_shared)
    ov = config.sample_overlap(np.array(uids))
    n = len(uids)
    iu = np.triu_indices(n, 1)
    dz = M.disease.values
    cross = (dz[iu[0]] != dz[iu[1]]) & ~ov[iu[0], iu[1]]
    v = S[iu][cross].astype(float)

    print(f"{n} nodes / {M.disease.nunique()} diseases / "
          f"{int(cross.sum())} cross-disease node pairs", flush=True)

    # A pair is characterised by its smaller side: a comparison is limited by
    # whichever network was estimated from less.
    size = M.n.values.astype(float)
    reg = M.n_regulon.values.astype(float)
    pair_n = np.minimum(size[iu[0]], size[iu[1]])[cross]
    pair_r = np.minimum(reg[iu[0]], reg[iu[1]])[cross]

    cov = None
    if "n_genes" in M.columns:
        g = M.n_genes.values.astype(float)
        cov = np.minimum(g[iu[0]], g[iu[1]])[cross]

    src = M.source.astype(str).values
    blk = np.where(src[iu[0]] == src[iu[1]], src[iu[0]], "TCGA-GEO")[cross]
    med = np.median(v)
    worst = max(abs(np.median(v[blk == b]) - med) for b in pd.unique(blk))

    gates = [("similarity_vs_sample_size", spearmanr(v, pair_n).statistic),
             ("similarity_vs_regulon_count", spearmanr(v, pair_r).statistic),
             ("worst_source_block_offset", worst)]
    if cov is not None:
        gates.insert(2, ("similarity_vs_gene_coverage",
                         spearmanr(v, cov).statistic))

    print("\n=== acceptance ===")
    for key, val in gates:
        print(f"  {'ok  ' if abs(val) <= args.gate else 'FAIL'} {key:<30} "
              f"{val:>+8.4f}   threshold |{args.gate}|")

    path = os.path.join(OUT, "acceptance_denovo.csv")
    pd.DataFrame([{"network": "denovo", "check": k, "value": float(val),
                   "threshold": args.gate, "passes": bool(abs(val) <= args.gate)}
                  for k, val in gates]).to_csv(path, index=False)
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

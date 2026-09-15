"""Similarity between two GRNs on the CollecTRI scaffold — the library.

This file holds the method implementations and the correction ladder, nothing
else. It is imported by `03_edges.py` and `04_mechanism.py` here, and by
`Verification/02_method_comparison.py`, which owns the acceptance gate and the
six-variant comparison. Keep it that way: the comparison harness and the method
were once two copies of one file, and a fix to one silently missed the other.

The chosen method is M3 S3-signed; the comparison table is written by the
method-comparison step.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from grn import paths as config                                  # noqa: E402

OUT = config.SCAFFOLD_OUT
HUB_PERCENT = 10.0          # legacy default for the top-X% hub set
CORE_PREVALENCE = 0.99


# ============================================================
# M0 — global cosine (incumbent)
# ============================================================

def apply_ladder(W, E, steps="full"):
    """03's corrections ② and ③, applied once so every method gets them.

    Without this the comparison is rigged: M0 centred its edges internally
    while M1 and M3 saw raw weights, so their confound numbers measured the
    missing correction rather than the method. 03 showed the two steps have
    different jobs — edge-centring removes the sample-size and coverage
    confounds, per-TF normalisation cuts the source-block effect (R² 0.24 ->
    0.09) — and neither is a property of any one similarity.
    """
    W2 = W - W.mean(axis=0, keepdims=True)            # ② edge-centred
    if steps == "centre":
        # ③ is not method-neutral: it L2-normalises exactly the per-TF blocks
        # that M1 and M3 read, which collapses every cohort onto the same
        # profile and makes those methods report similarity 1.0 for all pairs.
        # "centre" applies only the universal step.
        return W2
    tf = E.source_genesymbol.values                    # ③ per-TF normalised
    W3 = W2.copy()
    for t in pd.unique(tf):
        idx = np.where(tf == t)[0]
        blk = W3[:, idx]
        nrm = np.linalg.norm(blk, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        W3[:, idx] = blk / nrm
    return W3


def disattenuate(S, M, out_dir):
    """03's correction ④, applied to any similarity matrix."""
    hp = os.path.join(out_dir, "reliability_halfsplit.csv")
    if not os.path.isfile(hp):
        return S
    H = pd.read_csv(hp).set_index("nid")
    r = M.nid.map(H.r_half).values
    rel = np.clip(2 * r / (1 + r), 0.02, 1.0)          # Spearman-Brown
    rel = np.where(np.isfinite(rel), rel, np.nanmedian(rel))
    lo, hi = np.nanmin(S), np.nanmax(S)
    return np.clip(S / np.sqrt(np.outer(rel, rel)), lo, hi)


def block_centre(S, M, iu, cross):
    """03's correction ⑤ — remove each source block's additive offset.

    Ruled a technical offset rather than biology: inside TCGA, whether both
    units are tumour tissue makes no difference to similarity (0.253 vs 0.267),
    while GEO tumour pairs sit only +0.03 above the rest. Without it both
    networks fail the source gate (R² 0.32 / 0.24); with it they pass at
    0.002 / 0.018 and the discrimination AUC is unharmed.
    """
    src = M.source.values
    blk = np.where(src[iu[0]] == src[iu[1]], src[iu[0]], "TCGA-GEO")
    out = S.copy()
    for b in pd.unique(blk):
        m = blk == b
        off = np.median(S[iu][m & cross])
        i0, i1 = iu[0][m], iu[1][m]
        out[i0, i1] -= off
        out[i1, i0] -= off
    return out


def m0_global_cosine(W, tf_idx, edge_centre=False):
    X = W - W.mean(axis=0, keepdims=True) if edge_centre else W
    X = X - X.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = np.nan
    Z = X / nrm
    return np.clip(np.nan_to_num(Z @ Z.T), -1, 1)


# ============================================================
# M1 — hub-TF weighted Jaccard
# ============================================================

def tf_weighted_degree(W, tf_idx, n_tf, reduce="mean"):
    """cohort x TF matrix of |edge weight| aggregated over that TF's edges.

    MEAN, not sum — and this is the one place the legacy port had to change.
    De novo, each cohort had its own target set per TF, so summing |w| carried
    cohort-specific information. On the scaffold every cohort holds every edge,
    so the sum mostly counts how many targets CollecTRI gives that TF: measured
    spearman(sum-degree ranking, target count) = 0.97, which makes every cohort
    rank the same TFs on top and drives the hub Jaccard to 1.0 for every pair.
    Taking the mean removes the out-degree term (spearman drops to 0.08) and the
    ranking becomes a property of the cohort again.
    """
    A = np.abs(W)
    D = np.zeros((W.shape[0], n_tf))
    for t in range(n_tf):
        cols = tf_idx[t]
        if len(cols):
            D[:, t] = A[:, cols].mean(axis=1) if reduce == "mean" \
                else A[:, cols].sum(axis=1)
    return D


def m1_hub_jaccard(W, tf_idx, top_percent=HUB_PERCENT, n_tf=None):
    D = tf_weighted_degree(W, tf_idx, n_tf)
    k = max(1, int(n_tf * top_percent / 100))
    # top-k TFs per cohort; every cohort picks exactly k, so |A|=|B|=k and
    # Jaccard = inter / (2k - inter)
    order = np.argsort(-D, axis=1)[:, :k]
    B = np.zeros((W.shape[0], n_tf), dtype=np.float32)
    np.put_along_axis(B, order, 1.0, axis=1)
    inter = B @ B.T
    return inter / (2 * k - inter)


# ============================================================
# M3 — S_regulon, per-TF target-profile similarity averaged over TFs
# ============================================================

def _pairwise_weighted_jaccard(X):
    """sum_k min(a_k,b_k) / sum_k max(a_k,b_k) for non-negative rows of X.

    min = (a+b-|a-b|)/2 and max = (a+b+|a-b|)/2, so the pairwise sums follow
    from the row sums plus one cityblock distance matrix — no (n,n,m) tensor.
    """
    s = X.sum(axis=1)
    tot = s[:, None] + s[None, :]
    l1 = cdist(X, X, metric="cityblock")
    num, den = tot - l1, tot + l1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, 0.0)
    return out


def _pairwise_cosine(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = np.nan
    Z = X / nrm
    return np.nan_to_num(Z @ Z.T)


def m3_s_regulon(W, tf_idx, n_tf=None, normalize="l1", target_sim="cosine",
                 use_abs=True, min_targets=2):
    """Mean over TFs of the two cohorts' similarity on that TF's target profile.

    The legacy w1*Jaccard(TF sets) term is omitted: it is identically 1 on the
    scaffold. Variants match the legacy naming —
        S  : normalize="none", target_sim="jaccard"
        S2 : normalize="l1",   target_sim="jaccard"
        S3 : normalize="l1",   target_sim="cosine"
    `use_abs` mirrors the legacy `.abs()`; set False to keep activation and
    repression apart, which the correlation-based weights actually distinguish.
    """
    n = W.shape[0]
    acc = np.zeros((n, n))
    used = 0
    for t in range(n_tf):
        cols = tf_idx[t]
        if len(cols) < min_targets:
            continue
        X = np.abs(W[:, cols]) if use_abs else W[:, cols].copy()
        if normalize == "l1":
            s = np.abs(X).sum(axis=1, keepdims=True)
            s[s < 1e-12] = 1.0
            X = X / s
        if target_sim == "jaccard":
            if not use_abs:
                raise ValueError("weighted Jaccard needs non-negative weights")
            acc += _pairwise_weighted_jaccard(X)
        else:
            acc += _pairwise_cosine(X)
        used += 1
    return acc / max(used, 1)


# ============================================================
# The gate — identical criteria for every method
# ============================================================

def tissue_stratified_z(S, iu, strata):
    """Re-express each pair's similarity relative to pairs of the same tissue pairing.

    This is the recommended way to handle the tissue confound, and it is
    deliberately NOT a subtraction. Regressing tissue out of the similarity
    would delete real biology: two blood-borne autoimmune diseases sharing
    immune regulation is a true finding, and tissue is a *mediator* there, not
    only a confounder. What we can fix is the reference frame — instead of
    asking "is this pair similar", ask "is this pair similar **for a pair of
    this tissue pairing**".

    `strata` labels every pair by its unordered tissue pair (blood-blood,
    blood-brain, ...). Within each stratum the similarities are centred and
    scaled, so a blood-blood pair now has to beat other blood-blood pairs, not
    the global background that is dominated by cross-tissue pairs.

    Strata with too few pairs to estimate a spread are left uncentred and
    reported, because a z-score over 3 pairs is noise wearing a z-score's hat.
    """
    v = S[iu].astype(float).copy()
    out = v.copy()
    thin = []
    for s in pd.unique(strata):
        m = strata == s
        if m.sum() < 30:
            thin.append((s, int(m.sum())))
            continue
        sd = v[m].std()
        out[m] = (v[m] - v[m].mean()) / (sd if sd > 1e-12 else 1.0)
    return out, thin

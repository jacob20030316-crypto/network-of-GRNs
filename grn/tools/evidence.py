"""External evidence, retrieved rather than remembered.

Stage 11 of the pipeline. The read-out hands over a disease pair and the
factors that recur across its patient groups; before anything writes a
mechanistic account, this fetches what is already on record about those factors
and those diseases.

Two sources, for two different questions.

  Open Targets   is this factor already associated with this disease? A score
                 aggregated from genetics, somatic mutations, drugs, pathways,
                 animal models and literature — none of which is co-expression
                 in patients, so it is independent of how we found the factor.

  Europe PMC     what has been published about this factor, this disease, and
                 the two together? Returns real identifiers.

Both are looked up, never recalled. A language model asked to remember whether
a factor is linked to a disease answers from its training data, cites papers
that may not exist, and cannot distinguish "no link is known" from "I do not
know". Retrieval fixes all three: an identifier here resolves, and an empty
result is recorded as an empty result.

Open Targets is a local dump, so it is offline, free and reproducible. Europe
PMC is the one network call, and every response is cached on disk, so a rerun
costs nothing and gives the same answer.

Scores are compared within a disease, never across. Open Targets coverage is
wildly uneven — a common cancer has thousands of scored genes and a rare
syndrome a handful — so a raw score ranks diseases rather than factors. Each
factor is therefore placed as a percentile of the other scored genes of the
same disease.
"""
import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd

from grn import paths

OT = os.path.join(paths.EXTERNAL, "opentargets")
CHEA3 = os.path.join(paths.EXTERNAL, "chea3")
CACHE = os.path.join(paths.AGENT_OUT, "cache")
ASSOC_CACHE = os.path.join(CACHE, "opentargets_associations.parquet")
PMC_CACHE = os.path.join(CACHE, "europepmc")
DISEASE_MAP = os.path.join(paths.TABLES, "disease_opentargets_map.csv")

EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
POLITE_DELAY = 0.34          # Europe PMC asks for no more than 3 requests a second
PMC_TIMEOUT = 60
PMC_RETRIES = 4

os.makedirs(CACHE, exist_ok=True)
os.makedirs(PMC_CACHE, exist_ok=True)

_assoc = None
_chea3 = {}


# ==========================================================================
# Open Targets — local, offline
# ==========================================================================

def disease_map():
    """Our disease labels -> Open Targets identifiers, as resolved in curation."""
    M = pd.read_csv(DISEASE_MAP)
    return M[M.ot_id.notna() & (M.ot_id != "")].set_index("disease")


def _build_assoc_cache():
    """Slim the 1.1 GB association dump down to the diseases this study covers.

    Kept as a cache rather than recomputed: the full dump is read once, the
    result is a few megabytes, and every later lookup is a dictionary hit.
    """
    import glob
    M = disease_map()
    keep = set(M.ot_id)
    sym = pd.concat([pd.read_parquet(f, columns=["id", "approvedSymbol"])
                     for f in sorted(glob.glob(f"{OT}/target/*.parquet"))])
    s2g = dict(zip(sym.id, sym.approvedSymbol))

    parts = []
    files = sorted(glob.glob(f"{OT}/association_overall_direct/*.parquet"))
    if not files:
        raise SystemExit(f"no Open Targets association dump under {OT}")
    for i, f in enumerate(files, 1):
        a = pd.read_parquet(f, columns=["diseaseId", "targetId", "associationScore"])
        parts.append(a[a.diseaseId.isin(keep)])
        print(f"  [{i}/{len(files)}] {sum(len(p) for p in parts):,} rows kept",
              flush=True)
    A = pd.concat(parts, ignore_index=True)
    A["symbol"] = A.targetId.map(s2g)
    A = A.dropna(subset=["symbol"])

    # Percentile within the disease's own distribution — the only comparison
    # that means anything, for the reason given in the module docstring.
    A["percentile"] = A.groupby("diseaseId").associationScore.rank(pct=True)
    A["n_scored"] = A.groupby("diseaseId").associationScore.transform("size")
    A = A[["diseaseId", "symbol", "associationScore", "percentile", "n_scored"]]
    A.to_parquet(ASSOC_CACHE, index=False)
    print(f"wrote {ASSOC_CACHE}  ({len(A):,} rows, "
          f"{A.diseaseId.nunique()} diseases)")
    return A


def associations():
    """The slimmed association table, built on first use."""
    global _assoc
    if _assoc is None:
        if os.path.isfile(ASSOC_CACHE):
            _assoc = pd.read_parquet(ASSOC_CACHE)
        else:
            print("building the Open Targets association cache "
                  "(first run only, a few minutes)…", flush=True)
            _assoc = _build_assoc_cache()
    return _assoc


def open_targets(gene, disease):
    """What Open Targets records for one factor in one disease.

    Returns `known=False` where the pair is simply not scored, which is a fact
    about the database and is reported as such rather than as an absence of
    association.
    """
    M = disease_map()
    if disease not in M.index:
        return {"gene": gene, "disease": disease, "known": False,
                "reason": "disease not mapped to an Open Targets identifier"}
    ot_id = M.loc[disease, "ot_id"]
    A = associations()
    hit = A[(A.diseaseId == ot_id) & (A.symbol == gene.upper())]
    if not len(hit):
        n = int(A[A.diseaseId == ot_id].n_scored.iloc[0]) if (A.diseaseId == ot_id).any() else 0
        return {"gene": gene, "disease": disease, "ot_id": ot_id,
                "ot_name": M.loc[disease, "ot_name"], "known": False,
                "n_scored_for_disease": n,
                "reason": "no association recorded for this gene and disease"}
    r = hit.iloc[0]
    return {"gene": gene, "disease": disease, "ot_id": ot_id,
            "ot_name": M.loc[disease, "ot_name"], "known": True,
            "score": round(float(r.associationScore), 4),
            "percentile": round(float(r.percentile), 4),
            "n_scored_for_disease": int(r.n_scored)}


# ==========================================================================
# Europe PMC — the one network call, cached
# ==========================================================================

def _cache_path(query, limit, result_type):
    key = hashlib.sha1(f"{query}|{limit}|{result_type}".encode()).hexdigest()[:16]
    return os.path.join(PMC_CACHE, f"{key}.json")


ABSTRACT_CHARS = 600


def europepmc(query, limit=5, offline_ok=True, abstracts=True):
    """Search Europe PMC. Returns real identifiers, or an empty list.

    An empty list means the search found nothing, and the caller is expected to
    say so. It is never evidence that a relationship is new — only that this
    query did not match, which is a fact about the query and the index.
    """
    result_type = "core" if abstracts else "lite"
    path = _cache_path(query, limit, result_type)
    if os.path.isfile(path):
        return json.load(open(path, encoding="utf-8"))

    import requests
    # Relevance, not citation count. Sorting by citations answers a gene-and-
    # disease query with the field's mega-reviews and consensus guidelines,
    # which mention thousands of genes and settle nothing about this one.
    params = {"query": query, "format": "json", "pageSize": limit,
              "resultType": result_type}
    # The service goes slow under load, and a read timeout looks exactly like
    # "nothing published" once it reaches the agent. That confusion matters
    # more here than anywhere else in the pipeline, so a failed query is
    # retried; if it still fails it is recorded as a failure rather than as an
    # empty result, and is not cached, so a later run asks again.
    data, last = None, None
    for attempt in range(PMC_RETRIES):
        try:
            time.sleep(POLITE_DELAY if attempt == 0 else 2 ** attempt)
            r = requests.get(EUROPEPMC, params=params, timeout=PMC_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last = e
    if data is None:
        if offline_ok:
            return {"query": query, "hit_count": None, "results": [],
                    "error": f"{type(last).__name__}: {last}"}
        raise last

    out = {"query": query,
           "hit_count": int(data.get("hitCount", 0)),
           "results": [{"pmid": h.get("pmid"), "doi": h.get("doi"),
                        "title": h.get("title"), "journal": h.get("journalTitle"),
                        "year": h.get("pubYear"),
                        "cited_by": h.get("citedByCount"),
                        # a title rarely names the factor; without the abstract
                        # the screening step can only ever answer "uncertain"
                        "abstract": (h.get("abstractText") or "")[:ABSTRACT_CHARS]}
                       for h in data.get("resultList", {}).get("result", [])
                       if h.get("pmid")]}
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def _phrase(disease):
    """The term a literature index will actually match.

    Our labels are file names, not vocabulary: `Alzheimers_Disease` retrieves a
    hundredth of what `Alzheimer disease` does for the same factor, because the
    possessive form is not how the literature writes it. Open Targets holds the
    curated name for every disease we mapped, so that is used where it exists
    and the label is only a fallback.
    """
    try:
        M = disease_map()
        if disease in M.index:
            return str(M.loc[disease, "ot_name"])
    except Exception:
        pass
    s = re.sub(r"_+", " ", str(disease))
    return re.sub(r"\s*\([^)]*\)", "", s).strip()


def literature(gene, disease_a, disease_b, limit=4):
    """Three searches: the factor in each disease, and the two diseases together.

    The joint search is the one that matters for a shared-mechanism claim, and
    it is also the one most likely to return nothing.

    The factor is required in the title or abstract, the disease anywhere. A
    default Europe PMC search covers full text, and a short gene symbol then
    matches whatever else those three letters stand for: `CIC` with Alzheimer's
    returned 536 papers, led by an echocardiography score and a
    near-infrared-spectroscopy monitor. Requiring the symbol where a paper
    states its subject leaves 10, and for EZH2 it returns the one paper that is
    actually about EZH2 in the disease. Requiring both terms in the abstract is
    stricter still and mostly returns nothing, so the disease is left
    unrestricted.
    """
    a, b = _phrase(disease_a), _phrase(disease_b)
    return {
        "gene_in_a": europepmc(f'TITLE_ABS:"{gene}" AND "{a}"', limit),
        "gene_in_b": europepmc(f'TITLE_ABS:"{gene}" AND "{b}"', limit),
        "gene_in_both": europepmc(
            f'TITLE_ABS:"{gene}" AND "{a}" AND "{b}"', limit),
    }


# ==========================================================================
# ChEA3 — where a factor is observed bound
# ==========================================================================

def chea3(library="ReMap_ChIP-seq"):
    """TF -> the genes it is reported to bind, from one ChIP library."""
    if library not in _chea3:
        path = os.path.join(CHEA3, f"{library}.gmt")
        if not os.path.isfile(path):
            raise SystemExit(f"no ChEA3 library at {path}")
        d = {}
        for line in open(path, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if len(f) > 2:
                d.setdefault(f[0].split("_")[0].upper(), set()).update(
                    g.upper() for g in f[2:] if g)
        _chea3[library] = d
    return _chea3[library]


def chip_support(gene, targets, library="ReMap_ChIP-seq"):
    """How many of a factor's proposed targets it is observed bound to."""
    d = chea3(library)
    key = gene.upper()
    if key not in d:
        return {"gene": gene, "library": library, "in_library": False}
    bound = d[key]
    t = {g.upper() for g in targets}
    return {"gene": gene, "library": library, "in_library": True,
            "n_targets": len(t), "n_bound": len(t & bound),
            "fraction_bound": round(len(t & bound) / max(len(t), 1), 3),
            "library_size": len(bound)}


# ==========================================================================
# assembling one pair's evidence
# ==========================================================================

def pack_for(record, limit=4, with_literature=True):
    """Attach retrieved evidence to one disease pair's read-out.

    The read-out arrives with what this study measured — how often each factor
    recurred, against what background, over which targets. This adds what is on
    record elsewhere about the same factors, and returns the two side by side.
    Nothing is filtered out on the strength of the retrieved evidence: a factor
    nobody has written about stays in the pack, marked as such.
    """
    a, b = record["disease_a"], record["disease_b"]
    out = dict(record)
    out["evidence"] = []
    for t in record["tfs"]:
        gene = t["tf"]
        e = {"tf": gene,
             "open_targets": {a: open_targets(gene, a), b: open_targets(gene, b)},
             "chip": chip_support(gene, t.get("targets", []))}
        if with_literature:
            lit = literature(gene, a, b, limit=limit)
            e["literature"] = {k: {"query": v.get("query"),
                                   "hit_count": v.get("hit_count"),
                                   "papers": v.get("results", [])}
                               for k, v in lit.items()}
        out["evidence"].append(e)
    return out


def build_packs(source=None, out_dir=None, limit=4, with_literature=True,
                only=None):
    """One evidence pack per disease pair, written where the agents read them."""
    source = source or os.path.join(paths.SCAFFOLD_OUT, "agent_prompts_cond.jsonl")
    out_dir = out_dir or os.path.join(paths.AGENT_OUT, "evidence")
    os.makedirs(out_dir, exist_ok=True)
    records = [json.loads(l) for l in open(source, encoding="utf-8")]
    if only:
        records = [r for r in records if r["pair"] in set(only)]
    written = []
    for i, r in enumerate(records, 1):
        safe = re.sub(r"[^A-Za-z0-9_.|-]", "_", r["pair"]).replace("||", "__")
        path = os.path.join(out_dir, f"{safe}.json")
        if os.path.isfile(path):
            written.append(path)
            continue
        pack = pack_for(r, limit=limit, with_literature=with_literature)
        json.dump(pack, open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        written.append(path)
        n_lit = sum(1 for e in pack["evidence"]
                    if any(v["hit_count"] for v in e.get("literature", {}).values()))
        n_ot = sum(1 for e in pack["evidence"]
                   if any(v.get("known") for v in e["open_targets"].values()))
        print(f"  [{i}/{len(records)}] {r['pair'][:52]:<52} "
              f"{len(pack['evidence'])} factors, {n_ot} in Open Targets, "
              f"{n_lit} with literature", flush=True)
    print(f"\n{len(written)} packs in {out_dir}")
    return written


def run(**kw):
    """Entry point for the pipeline runner."""
    build_packs(**kw)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="probe the evidence sources")
    ap.add_argument("--build-packs", action="store_true",
                    help="write one evidence pack per disease pair")
    ap.add_argument("--only", nargs="+", help="restrict to these pair ids")
    ap.add_argument("--no-literature", action="store_true")
    ap.add_argument("--build-cache", action="store_true",
                    help="build the Open Targets association cache and stop")
    ap.add_argument("--gene", default="STAT2")
    ap.add_argument("--disease-a", default="COVID-19")
    ap.add_argument("--disease-b", default="Sjögrens_Syndrome")
    ap.add_argument("--no-network", action="store_true")
    args = ap.parse_args()

    if args.build_cache:
        associations()
        raise SystemExit(0)

    if args.build_packs:
        build_packs(only=args.only, with_literature=not args.no_literature)
        raise SystemExit(0)

    M = disease_map()
    print(f"disease map: {len(M)} of our labels carry an Open Targets id\n")
    for d in (args.disease_a, args.disease_b):
        print(json.dumps(open_targets(args.gene, d), ensure_ascii=False))
    print()
    print(json.dumps(chip_support(args.gene, ["IFIT1", "MX1", "OAS2", "IRF9"]),
                     ensure_ascii=False))
    if not args.no_network:
        print()
        lit = literature(args.gene, args.disease_a, args.disease_b, limit=3)
        for k, v in lit.items():
            n = v.get("hit_count")
            print(f"{k:14s} hits={n}")
            for r in v.get("results", [])[:3]:
                print(f"    PMID {r['pmid']}  {str(r['title'])[:70]}")

"""Stage 3, first half: the assistant agent.

It answers "what is already on record about these factors and these diseases",
and hands that to the biologist agent as an evidence pack.

Retrieval is mechanical; judgement is not. The agent is given a first, literal
pass over Open Targets and Europe PMC and asked two things a lookup cannot
decide for itself:

  what else to search   our disease labels are not the terms the literature
                        uses. `Sjögrens_Syndrome` retrieves a different set
                        from `Sjogren syndrome` or `sicca syndrome`, and the
                        count that comes back is only as good as the term. Open
                        Targets records the synonyms, so the agent chooses among
                        real alternatives rather than inventing them.

  what came back is on point   a paper matching both disease names usually
                        mentions one in passing. Separating that from a paper
                        actually about a shared axis is reading, and it is the
                        distinction the whole read-out rests on: co-mention is
                        not mechanism.

The agent never fetches anything itself and never writes an identifier. It
proposes queries, which are run here; it marks identifiers, which are checked
here against what was retrieved. A pack therefore cannot contain a citation
that does not resolve, whatever the model does.
"""
import glob
import json
import os
import re

import pandas as pd

from grn import paths
from grn.agents import llm
from grn.tools import evidence

PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "prompts", "assistant.md")
OUT_DIR = os.path.join(paths.AGENT_OUT, "evidence")
MAX_EXTRA_QUERIES = 12


def synonyms_for(disease, limit=8):
    """What Open Targets calls this disease, so the agent picks a real term."""
    M = evidence.disease_map()
    if disease not in M.index:
        return []
    ot_id = M.loc[disease, "ot_id"]
    D = pd.read_parquet(glob.glob(f"{evidence.OT}/disease/*.parquet")[0])
    row = D[D.id == ot_id]
    if not len(row):
        return []
    row = row.iloc[0]
    names = [row["name"]]
    for c in ("exactSynonyms", "narrowSynonyms", "relatedSynonyms"):
        v = row.get(c)
        if v is not None and len(v):
            names += [str(x) for x in v]
    seen, out = set(), []
    for n in names:
        k = n.lower()
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out[:limit]


def _pmids(pack):
    """Every identifier the first pass actually retrieved for this pair."""
    got = set()
    for e in pack["evidence"]:
        for v in e.get("literature", {}).values():
            for p in v.get("papers", []):
                if p.get("pmid"):
                    got.add(str(p["pmid"]))
    return got


def brief(pack):
    """What the agent is shown: the record so far, and nothing of our statistics.

    Recurrence, p-values and rankings are deliberately withheld. The assistant's
    job is to describe the literature, and it should not be able to tune that
    description to how strongly we scored a factor.
    """
    a, b = pack["disease_a"], pack["disease_b"]
    lines = [f"pair_id: {pack['pair']}",
             f"disease A: {a}",
             f"  Open Targets synonyms: {'; '.join(synonyms_for(a)) or '(none)'}",
             f"disease B: {b}",
             f"  Open Targets synonyms: {'; '.join(synonyms_for(b)) or '(none)'}",
             "", "factors, and the first pass over the record:"]
    for e in pack["evidence"]:
        lines.append(f"\n  {e['tf']}")
        for d, key in ((a, a), (b, b)):
            ot = e["open_targets"][key]
            if ot.get("known"):
                lines.append(f"    Open Targets in {d}: score {ot['score']}, "
                             f"percentile {ot['percentile']} of "
                             f"{ot['n_scored_for_disease']} scored genes")
            else:
                lines.append(f"    Open Targets in {d}: no association recorded "
                             f"({ot.get('n_scored_for_disease', 0)} genes scored "
                             f"for this disease)")
        chip = e.get("chip", {})
        if chip.get("in_library"):
            lines.append(f"    ChIP (ReMap): bound at {chip['n_bound']} of "
                         f"{chip['n_targets']} of its proposed targets")
        for label, v in e.get("literature", {}).items():
            n = v.get("hit_count")
            shown = n if n is not None else "query failed"
            lines.append(f"    {label}: {shown} hits"
                         f"   [query: {v.get('query') or ''}]")
            for p in v.get("papers", []):
                lines.append(f"       PMID {p['pmid']} ({p.get('year','?')}) "
                             f"{str(p.get('title',''))[:120]}")
                abst = (p.get("abstract") or "").strip()
                if abst:
                    lines.append(f"         {abst}")
    return "\n".join(lines)


def validator(pack):
    """Structure, plus the two things a reader could not check for themselves."""
    known = _pmids(pack)
    factors = [e["tf"] for e in pack["evidence"]]

    def check(obj):
        llm.check_keys(obj, ["pair_id", "extra_queries", "per_factor", "coverage"])
        if obj["pair_id"] != pack["pair"]:
            raise llm.LLMError(f"pair_id should be {pack['pair']!r}")
        for q in obj["extra_queries"][:MAX_EXTRA_QUERIES]:
            llm.check_keys(q, ["factor", "query", "why"])
            if q["factor"] and q["factor"].upper() not in {f.upper() for f in factors}:
                raise llm.LLMError(
                    f"extra_queries names a factor that was not supplied: {q['factor']}")
        for f in obj["per_factor"]:
            llm.check_keys(f, ["factor", "supported_in_a", "supported_in_b",
                               "joint", "uncertain", "note"])
            if f["factor"].upper() not in {x.upper() for x in factors}:
                raise llm.LLMError(
                    f"per_factor names a factor that was not supplied: {f['factor']}")
            for field in ("supported_in_a", "supported_in_b", "joint", "uncertain"):
                bad = [p for p in f[field] if str(p) not in known]
                if bad:
                    raise llm.LLMError(
                        f"{f['factor']}.{field} lists identifiers that were not "
                        f"retrieved: {', '.join(map(str, bad[:5]))}")
        return obj
    return check


def run_extra_queries(pack, plan, limit=4):
    """Run what the agent asked for, and attach it. The agent never fetches."""
    added = 0
    by_tf = {e["tf"].upper(): e for e in pack["evidence"]}
    for q in plan.get("extra_queries", [])[:MAX_EXTRA_QUERIES]:
        res = evidence.europepmc(q["query"], limit=limit)
        target = by_tf.get(str(q.get("factor", "")).upper())
        if target is None:
            pack.setdefault("pair_level_queries", []).append(
                {"why": q.get("why", ""), **res})
        else:
            target.setdefault("literature_extra", []).append(
                {"why": q.get("why", ""), **res})
        added += 1
    return added


def accept(pack):
    """Validate the agent's plan, act on it, and return the finished pack.

    The batch writes whatever this returns, so what lands on disk is the
    evidence pack the biologist will read — the retrieved record plus the
    agent's reading of it — never the agent's reply on its own.
    """
    check = validator(pack)

    def f(obj):
        plan = check(obj)
        n_added = run_extra_queries(pack, plan)
        by_tf = {e["tf"].upper(): e for e in pack["evidence"]}
        for got in plan.get("per_factor", []):
            e = by_tf.get(got["factor"].upper())
            if e is not None:
                e["assessed"] = {k: got[k] for k in
                                 ("supported_in_a", "supported_in_b", "joint",
                                  "uncertain", "note")}
        pack["assistant"] = {"coverage": plan.get("coverage", ""),
                             "extra_queries_run": n_added}
        return pack
    return f


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=os.path.join(
        paths.SCAFFOLD_OUT, "agent_prompts_cond.jsonl"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--fixed-only", action="store_true",
                    help="run the literal first pass and stop, with no agent")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--jobs", type=int, default=6,
                    help="pairs handled at once; each is minutes of waiting")
    args = ap.parse_args(argv)

    records = [json.loads(l) for l in open(args.source, encoding="utf-8")]
    if args.only:
        records = [r for r in records if r["pair"] in set(args.only)]
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} disease pairs from {os.path.basename(args.source)}")

    os.makedirs(args.out_dir, exist_ok=True)

    def key_of(r):
        return re.sub(r"[^A-Za-z0-9_.-]", "_", r["pair"].replace("||", "__"))

    # The literal pass first: it is offline for Open Targets, cached for Europe
    # PMC, and it is what the agent is asked to improve on.
    packs = {}
    for i, r in enumerate(records, 1):
        packs[r["pair"]] = evidence.pack_for(r)
        print(f"  [{i}/{len(records)}] retrieved  {r['pair'][:56]}", flush=True)

    if args.fixed_only:
        for r in records:
            p = os.path.join(args.out_dir, key_of(r) + ".json")
            json.dump(packs[r["pair"]], open(p, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        print(f"\n{len(records)} packs written without the agent")
        return 0

    print()
    return llm.run_batch(
        records, key_of,
        prompt_of=lambda r: (open(PROMPT, encoding="utf-8").read()
                             + "\n\n---\n\n" + brief(packs[r["pair"]])),
        validate_of=lambda r: accept(packs[r["pair"]]),
        out_dir=args.out_dir, label="pair",
        model=args.model, backend=args.backend, redo=args.redo,
        jobs=args.jobs)


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
    import sys
    sys.exit(main())

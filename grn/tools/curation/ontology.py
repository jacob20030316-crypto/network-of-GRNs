"""Reading UBERON, and walking a term up to the system it belongs to.

The hierarchy is why the ontology is worth using at all: a tissue anchored to a
class comes with its organ system for free, from a public identifier anyone can
check rather than from a grouping we invented.

This is what remains of a larger module that also held a hand-written map from
our tissue labels to UBERON classes. That map is now the tissue agent's, and
what it proposes is resolved against the ontology here.
"""
SYSTEM_WEAK = {"entire sense organ system", "non-connected functional system"}


def parse_obo(path):
    """-> {id: {name, is_a, part_of}}.

    is_a and part_of are kept apart on purpose. `is_a` answers "what kind of
    thing is this" and `part_of` answers "what is it inside of" — conflating
    them makes blood a kind of anatomical system, because blood is part_of one.
    """
    terms, cur = {}, None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"is_a": [], "part_of": [], "name": None, "obsolete": False}
        elif line.startswith("["):
            cur = None                       # [Typedef] etc.
        elif cur is None:
            continue
        elif line.startswith("id: UBERON"):
            cur["id"] = line[4:]
            terms[cur["id"]] = cur
        elif line.startswith("name: "):
            cur["name"] = line[6:]
        elif line.startswith("is_obsolete: true"):
            cur["obsolete"] = True
        elif line.startswith("is_a: UBERON"):
            cur["is_a"].append(line[6:].split(" ")[0])
        elif line.startswith("relationship: part_of UBERON"):
            cur["part_of"].append(line.split()[2])
    return {i: t for i, t in terms.items() if not t["obsolete"] and t["name"]}


def ancestors(terms, start, rels=("is_a", "part_of")):
    """Closure over the given relations, inclusive of `start`.

    rels=("is_a",)            -> classification: what kind of thing is it
    rels=("is_a","part_of")   -> containment:    what is it inside of
    """
    seen, q = {start}, deque([start])
    while q:
        cur = terms[q.popleft()]
        for rel in rels:
            for p in cur[rel]:
                if p in terms and p not in seen:
                    seen.add(p)
                    q.append(p)
    return seen

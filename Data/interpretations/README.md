# The mechanistic accounts

One file per disease pair, for the pairs whose recurring factors survived the
permutation null at a false discovery rate of 0.05.

Each was written by the biologist agent from an evidence pack: the factors and
how often each recurred across that pair's patient groups, what Open Targets
records for them in each of the two diseases, where they are observed bound, and
the papers retrieved for them, already read once and marked for whether they
bear on the claim. The agent was not shown any p-value or ranking of ours, so
that agreement between its reading and our statistics is evidence rather than an
echo.

    pair_id            the two diseases
    programme          what the listed factors do together, named by the factors
    carrying_factors   which of them carry it
    role_in_a          what the programme does in the first disease
    role_in_b          and in the second
    shared_axis        why the two diseases would share it
    verdict            established | supported | unexpected | incoherent
    support_strength   strong | moderate | weak | insufficient
    reasoning          why that verdict and that strength
    support            PMIDs, each with what it shows
    how_to_falsify     an experiment that would show the account is wrong
    caveats            tissue pairing, group counts, what the exclusions imply

`verdict` and `support_strength` answer different questions and are kept apart:
whether the relationship is already known, and whether the evidence in front of
the agent carries the account. A well-supported account of a known axis and a
thinly-supported guess at a new one are different results.

Two things to know when reading them.

**Every identifier resolves.** Each PMID in `support` is a real record, and
`carrying_factors` names only factors the pair actually reported.

**An empty literature is not evidence of novelty.** Where nothing was retrieved
for a factor, the account says so. That is a fact about the query and the index,
never a finding.

These are hypotheses. None has been tested experimentally, which is what
`how_to_falsify` is for.

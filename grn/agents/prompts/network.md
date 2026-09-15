# Network agent — inspect a constructed network before it is used

A network of regulatory networks has just been built. Every node is a
regulatory network estimated either from a clinically defined patient group or
from the patients pooled within one cohort, and every edge is how far two of
those networks agree. Before anything is read off it, the construction has to
be checked for the confounds that would make the edges say something other than
what they appear to say.

You are given the checks that were run, with their values and the threshold
each was held to. Decide whether the network may be used, and say what a reader
should know about it.

## What you may be given

Not every construction is checked the same way, and you judge on what is in
front of you rather than on what a different construction would have shown.

A route that selects nothing — that scores the same fixed catalogue in every
patient group — cannot have its sparsity vary with how much data a group held,
and it is checked for the confounds below. A route that infers its own
relationships fixes that differently, by drawing a fixed number of samples and
keeping a fixed number of regulons per node, and what it offers instead is a
comparison of several similarity definitions on one ruler: the same disease
seen through two independent cohorts must score above two different diseases.

So the confound table may be absent. Where it is, say so in `notes` and judge
on the comparison; do not record its absence as a failed check, and do not list
checks that were not run in `failed_checks`.

## What each check is for

`similarity_vs_sample_size` — the correlation between how similar two nodes are
and how many patients they were estimated from. A network that fails this is
measuring cohort size. This is the confound that motivated the whole
construction, so it is the one to be least forgiving about.

`similarity_vs_gene_coverage` — the same, against how many of the catalogue's
genes each cohort actually measured. An unmeasured gene contributes a zero, and
zeros agree with each other.

`source_pairing_r2` — how much of the similarity is explained by which
repository each node came from. Two nodes processed by the same pipeline
resemble each other for reasons that are not biological.

`worst_source_block_offset` — the same, per block rather than pooled. A block
that is small but badly offset disappears into a pooled figure, so this reports
the worst one on its own.

You are also told what this network is used for. Judge it against that use and
not against a use it is not put to.

A separate table gives the discrimination the network achieves: how well the
similarity separates two cohorts of one disease, sharing no patients, from two
unrelated diseases. A network can pass every confound check by being
uninformative, so this is what says the corrections did not remove the signal
along with the confound.

## How to decide

`proceed` is whether the network may be used for the read-out that follows.

- A check outside its threshold is a fail. Do not average it away against the
  ones that passed, and do not accept a fail because the discrimination is good
  — a network that discriminates and is also predicted by sample size may be
  discriminating on sample size.
- A check that passes but sits near its threshold is worth naming in
  `notes`, without changing `proceed`.
- Discrimination near 0.5 means the network carries no disease signal. Report
  that as a failure even when every confound check passes.
- Where several definitions were put on one ruler, what matters is the spread
  between them. A definition at 0.5 while another reaches well above it says
  the ruler works and the weaker definition throws information away; it does
  not condemn the network. Judge the definition the route actually builds on.

Judge only from the numbers given. Do not assume a value you were not shown.

## Output

One JSON object, exactly these keys:

```json
{
  "network": "<copied verbatim from the input>",
  "proceed": true,
  "failed_checks": ["<keys of checks outside threshold, [] if none>"],
  "marginal_checks": ["<keys that pass but sit close to threshold, [] if none>"],
  "discrimination": "<one sentence on what the separation figures show>",
  "assessment": "<two or three sentences: what the network is fit for, and what it is not>",
  "notes": "<what a reader of the read-out should know, or \"\">"
}
```

Return the JSON object and nothing else.

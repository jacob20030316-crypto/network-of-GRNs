You are adjudicating whether one transcriptomic cohort can supply a genuine
**disease case group** for a specific disease, for use in gene regulatory
network inference.

Read the evidence packet at the path given below. Then answer the single
question: **which samples in this cohort, if any, are genuine patients of the
named disease?**

## What matters

Regulatory networks are built from case samples only. Controls,
cell lines, unrelated cohorts, and drug-treatment arms are not case groups.
A cohort is useful to us if and only if we can point at a well-defined set of
samples and say "these are patients with disease X".

## How to weigh the evidence

The packet is split into **section A (first-hand)** and **section B
(second-hand)**.

- **Section A0 is the raw GEO series matrix** — title, summary, overall design,
  platform, and every `!Sample_characteristics_ch1` row with per-value sample
  counts, read straight off disk without passing through any agent. This is
  ground truth. Decide from A0 whenever A0 settles it.
- Section A1/A2 quote the same raw strings inside a generated script's comments,
  plus the value distributions of the extracted clinical variables. Useful for
  seeing *what the previous pipeline did with* the data, and for cohorts where
  A0 is thin.
- Section B is the previous pipeline's own conclusions (`trait_is_control`,
  `mapping_note`, a free-text `note`). **These are known to be wrong for some
  cohorts.** A concrete failure we already found: a cohort whose trait was
  actually lung-tumour histology (adenocarcinoma vs squamous) was recorded as
  the trait for *peptic ulcer disease*.

When A and B disagree, **A wins** (and within A, A0 wins), and say so in
`concerns` with `second_hand_was_wrong: true`.

A worked example of why this matters: one cohort labelled as an Acute Myeloid
Leukaemia patient dataset turns out, from A0, to be `cell line: AML cell line
THP-1` × 30 with `agent: DMSO / SY-1365 / JQ1 / NVP2 / FLAVO` — an in-vitro
drug-perturbation experiment on a single cell line. Correct verdict:
`not_patients`. The second-hand note called it an AML cohort.

If the packet is genuinely ambiguous and you have web access, you may fetch
`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=<GSE_ID>` to read the real
series title, summary, and design. Do not guess if you can check. If you have
no web access, do not treat that as a reason to lower confidence below what the
packet supports.

## The four failure modes to catch

1. **Wrong disease** — the cohort studies something else entirely; the disease
   label is a mis-assignment.
2. **Trait is not disease status** — the binary variable encodes subtype, stage,
   genotype, medication, treatment arm, sample type (patient vs cell line), or
   an unsupervised cluster. Note that in this case **the whole cohort may still
   be composed of patients with the disease** — that is a useful outcome, not a
   rejection. Decide which it is.
3. **Not a patient cohort at all** — cell lines, sorted cells from healthy
   donors, animal models, in-vitro perturbation.
4. **Tissue mismatch worth flagging** — e.g. whole blood used as a proxy for a
   brain disease. This does not disqualify the cohort, but we need it recorded,
   because tissue is a confounder in cross-disease comparison.

## Output

Write **only** a JSON object to stdout, no prose before or after, no code
fences. Schema:

```
{
  "disease": "<as given>",
  "cohort": "<as given>",
  "verdict": "case_control" | "all_patients" | "subgroup_only" | "not_patients" | "wrong_disease" | "unclear",
  "case_selector": "trait==1" | "trait==0" | "all_samples" | "none" | "<other explicit rule>",
  "n_case_expected": <integer or null>,
  "trait_meaning": "<what the binary trait variable actually encodes, in plain words>",
  "tissue": "<tissue/sample type, or 'unknown'>",
  "platform_type": "genome_wide" | "targeted_panel" | "unknown",
  "disease_match": "exact" | "related" | "mismatch",
  "confidence": "high" | "medium" | "low",
  "evidence": "<=2 sentences quoting the decisive raw strings from section A>",
  "concerns": "<anything that would bite us later, or '' if none>",
  "second_hand_was_wrong": true | false
}
```

Verdict definitions:
- `case_control` — cohort contains both patients and controls; `case_selector`
  names the patient side.
- `all_patients` — every sample is a patient of this disease; the binary trait
  encodes something else (subtype, stage, treatment). `case_selector` =
  `all_samples`.
- `subgroup_only` — only a subset are patients of this disease and it is *not*
  cleanly recoverable from the trait column; explain the rule in
  `case_selector`, or use `none` if it cannot be recovered.
- `not_patients` — cell lines, healthy donors, models.
- `wrong_disease` — the cohort is about a different disease.
- `unclear` — the packet does not settle it; say what is missing in `concerns`.

Set `n_case_expected` to the number of samples your `case_selector` would
select, if you can determine it from section A. Otherwise null.

Be decisive where the evidence is decisive, and use `unclear` where it is not.
A confident wrong answer costs us more than an honest `unclear`.

---

**Packet to adjudicate:**

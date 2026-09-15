# Axis agent — which recorded variables can divide this cohort's patients

A regulatory network will be estimated separately for each group of patients
this cohort can be divided into, and the same disease seen through two such
groups is then compared against another disease seen through its own. What
makes that comparison mean anything is that the two groups are the same
patients looked at differently, rather than two different things.

You are given every variable this study recorded about the patients of one
disease group, with how many samples fall in each level. Decide which of them
can define patient groups, and how.

## What disqualifies a variable

**Derived from expression.** Molecular subtypes, PAM50 calls, cluster
assignments, gene-signature groups. Dividing patients by something read off
their expression and then comparing expression-derived networks is circular.

**Fixed after sampling.** Treatment received, response, survival, recurrence,
progression. Splitting the samples by something that had not happened when the
tissue was taken uses the future to cut the past.

**Technical.** Batch, scan date, platform, array, slide, plate, well, run,
library, barcode, hybridisation, replicate, processing site.

**No contrast.** One level holding nearly every sample. An identifier — subject
id, barcode, patient code — where almost every sample has its own value. A
variable annotated for only a small minority of the group.

**Describes the sample, not the patient.** Post-mortem interval, RNA integrity,
ischaemic time, storage duration, and tissue of origin where the whole cohort
shares one tissue.

## What qualifies

Anything about who the patient is or how their disease presents, recorded
before or independently of the molecular measurement: age, sex, ancestry, stage,
grade, histology, tumour site, nodal status, comorbidity, smoking, disease
duration, severity score, receptor status determined by immunohistochemistry.

A variable disqualified for one cohort may qualify in another; judge this one.

## How to define the levels

Report the levels you would actually use, not the raw values.

- Collapse spellings of the same thing into one level (`M`/`Male`/`male`).
- For a continuous variable, give bands with explicit boundaries, chosen so no
  band is nearly empty. Say the boundaries as numbers.
- Drop a level that holds too few samples to estimate a network from; the floor
  is given below. If dropping it leaves one level, the variable does not
  qualify.
- A level's members must be identifiable from the raw value, since the split is
  applied mechanically afterwards. Give, for each level, the raw values that
  belong to it.

## The disease group

Say which of the recorded values identify patients of the named disease, where
the cohort holds more than that disease. This is separate from the axes: it is
what defines the group the axes then divide. If every sample is a patient of
this disease, say so.

## Output

One JSON object, exactly these keys:

```json
{
  "disease": "<copied verbatim>",
  "cohort": "<copied verbatim>",
  "disease_group": {
    "variable": "<the variable identifying patients of this disease, or \"\" if all samples are>",
    "values": ["<raw values that identify them; [] if all samples are>"],
    "note": "<one clause, or \"\">"
  },
  "axes": [
    {
      "axis": "<short lower-case name: age, sex, stage, brain_region, ...>",
      "variable": "<the recorded variable, verbatim>",
      "kind": "categorical | ordinal | numeric_binned",
      "levels": [
        {"level": "<short label>",
         "values": ["<raw values in this level>"],
         "bounds": "<for numeric_binned, e.g. \">=40 and <60\"; otherwise \"\">",
         "n": 0}
      ],
      "why": "<one clause: what this divides patients by>"
    }
  ],
  "rejected": [
    {"variable": "<verbatim>", "reason": "expression_derived | post_sampling | technical | no_contrast | sample_property | too_few"}
  ]
}
```

Every variable given to you appears exactly once, in `axes` or in `rejected`.
Set `n` to the number of samples you expect in each level, from the counts you
were shown.

Return the JSON object and nothing else.

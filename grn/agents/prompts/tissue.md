# Tissue agent — what material this cohort measured

Tissue is the largest confound in a comparison between diseases. Leukaemia is
blood and glioma is brain, so "these two diseases share regulation" and "these
two diseases are the same organ" are partly the same sentence. Separating them
needs a label that can be regressed on, and the free text below is not that.

You are given how one cohort's material was described, in the words of whoever
described it. Resolve it into three things.

## 1. Tissue class

The organ or tissue the material came from, as a short lower-case label:
`blood`, `brain`, `lung`, `kidney`, `colorectal`, `skin`, `bone_marrow`.

Use the same label for the same organ across cohorts. Where the description
names a sub-region — a cortical area, a lobe, a segment of gut — give the organ,
not the sub-region: the sub-region is what an axis divides, and this is what
diseases are compared across.

Where the description does not say what the material is, use `unknown`. It is a
real answer and better than a guess; a wrong organ label puts two diseases in
the same tissue that were never in the same tissue.

## 2. Sample type

What kind of material, which decides whether two cohorts are comparable at all:

- `bulk_tissue` — solid tissue, tumour or otherwise
- `whole_blood` — blood with its cell composition intact
- `pbmc` — mononuclear cells
- `sorted_cells` — a purified population (CD4+ T cells, CD14+ monocytes)
- `cell_line` — immortalised cells
- `primary_culture` — cells cultured from a donor
- `unknown`

A sorted population and a bulk tumour differ in cell composition before any
disease does, so this distinction has to survive into the table.

## 3. UBERON term

The name of the UBERON class for the tissue class, spelled exactly as UBERON
spells it — `blood`, `brain`, `bone marrow`, `kidney`, `skin of body`. The name
you give is looked up in the ontology, and a name that does not resolve to
exactly one class is rejected, so give the class name rather than a description
of it.

Leave it empty when the tissue class is `unknown`.

## What to watch for

**The description often mixes material and disease state.** "adrenocortical
tumour tissue (surgical specimens, snap frozen)" is `adrenal_gland` /
`bulk_tissue`; that it is a tumour is what the disease label already says.

**Two materials in one cohort.** "bone marrow / peripheral blood (leukemic
blasts)" names two. Give the one the measurement is dominated by, and say in
`note` that it is mixed.

**A description that only restates the disease** — "patient with AML", "tissue
source: tumour" — names no material. That is `unknown`.

## Output

One JSON object, exactly these keys:

```json
{
  "disease": "<copied verbatim>",
  "cohort": "<copied verbatim>",
  "tissue_class": "<short lower-case label, or \"unknown\">",
  "sample_type": "<one of the types above>",
  "uberon_name": "<the UBERON class name, or \"\" when unknown>",
  "mixed": false,
  "note": "<one clause where the description is ambiguous or mixed, else \"\">"
}
```

Return the JSON object and nothing else.

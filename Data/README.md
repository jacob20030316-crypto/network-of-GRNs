# The results this pipeline reports

The cohorts the analysis starts from are the GEO and TCGA transcriptomes linked
in the repository README. What is kept here is what the pipeline reported from
them: the regulators of each disease pair, and the mechanistic account written
for it.

    reported_factors.csv               the regulators reported on the curated scaffold
    denovo_reported_factors.csv        the regulators reported without a catalogue
    denovo_shared_target_enrichment.csv   their shared targets against two references
    interpretations/                   one mechanistic account per disease pair

## reported_factors.csv

Twelve regulators for each disease pair carried forward, one row per regulator.

    pair                          the two diseases
    regulator                     transcription factor
    direction                     concordant or discordant coupling
    recurrence                    fraction of the pair's patient-group combinations
                                  in which the regulator cleared the 90th percentile
                                  of its tissue-matched background
    recur_fold                    that fraction over the 0.10 expected by chance
    n_combinations                patient-group combinations available for the pair
    median_agreement              median agreement of the regulator's target weights
                                  across those combinations
    background_percentile         where that median falls in the regulator's own
                                  tissue-matched background
    n_targets                     targets recorded for the regulator in CollecTRI
    tissue_pairing                the two tissues compared
    agreement_top100              overlap of the pair's 100 leading regulators
    agreement_top12               the same for its 12 leading regulators, both
                                  measured between two group-disjoint halves
    tissue_generality_percentile  how ordinary the regulator is for these tissues,
                                  computed with the two diseases held out
    share_of_pairs_reporting      fraction of reported pairs listing this regulator

## denovo_reported_factors.csv

The regulators retained where each network was inferred from expression alone.
One row per regulator and disease pair, with the genes the two independently
inferred regulons share.

    median_jaccard                overlap of the two regulons
    background_percentile, bg_n   that overlap against the same regulator's overlaps
                                  among unrelated disease pairs, and how many were
                                  available
    p, q_within_pair              empirical probability, corrected within the pair
    n_shared_targets, shared_targets   the genes both regulons assign to the regulator
    nes_a, nes_b                  motif enrichment of the regulon in each network

`denovo_shared_target_enrichment.csv` tests those shared targets against the
targets CollecTRI records for the same regulator and against ReMap ChIP
evidence, with a null that permutes regulator identity between regulons of
comparable size.

## interpretations/

One JSON file per disease pair, written by the biologist agent from the
regulators above and the evidence retrieved for them. Field definitions are in
`interpretations/README.md`. These are hypotheses; none has been tested
experimentally.

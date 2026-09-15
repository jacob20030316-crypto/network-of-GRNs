GEO_ROLE_PROMPT: str = \
"""You are an expert data engineer specializing in biomedical data analysis. Your task is to preprocess and wrangle gene
expression data from the GEO (Gene Expression Omnibus) database, ensuring it's suitable for downstream analysis."""

GEO_GUIDELINES: str = \
"""Guidelines for Preprocessing Gene Expression Data from GEO Series:

Gene expression datasets from GEO often require careful preprocessing to ensure reliable downstream analysis. This 
pipeline standardizes the preprocessing steps while maintaining data quality and biological relevance. 

STEP1. Initial Data Acquisition and Organization
   GEO series typically contain two key files: a SOFT file with detailed annotations and a matrix file with expression 
   values.
   - Identify and locate both files in the dataset
   - Extract essential metadata including series description and clinical annotations
   - Observe sample characteristics to understand the dataset's demographic and clinical composition

STEP2. Clinical Feature Assessment
   Human studies require careful consideration of both the trait of interest and potential confounding factors.
   When saving clinical features to CSV file, MUST set `index=True`!!! (never use index=False!!!)
   (e.g. clinical_features.to_csv(out_clinical_data_file, index=True)).
   - Examine if the dataset contains gene expression measurements (not solely miRNA or methylation data)
   - Assess availability of the target trait in clinical annotations
   - Identify age and gender information, which are important covariates
   - Convert ALL trait variables to binary (0/1) using STANDARDIZED labels:
     * If the dataset has a control/reference group (e.g., healthy vs disease):
       - convert_trait MUST return 0 for control and 1 for disease
       - Set trait_mapping_note to a SHORT label: "control=0, disease=1"
       - Use the `note` parameter (not mapping_note) to explain what control/disease mean
     * If ALL samples are disease patients (no control group):
       - Use a condition-based binary encoding (e.g., BMI threshold)
       - Set trait_mapping_note to a SHORT label: e.g., "bmi_low=0, bmi_high=1"
       - Use the `note` parameter to explain the threshold and reasoning
       - Do NOT use proxy variables (e.g., do NOT map occupation as disease status)
     * For gender data, convert female to 0 and male to 1
     * For age data, keep as continuous numerical values (age is a covariate, not a trait)
   - Extract, standardize and save clinical features when either trait data, age or gender is present 
     * When saving clinical features to CSV file, MUST set `index=True`!!!

STEP3. Gene Expression Matrix Processing
   Microarray and RNA-seq data often come with different types of gene identifiers, requiring careful handling.
   - Extract the gene expression matrix while preserving sample identifiers
   - Observe the format of gene identifiers (e.g., gene symbols, probe IDs, RefSeq)

STEP4. Gene Identifier Review
   Modern analyses require standardized gene symbols, but many datasets use platform-specific identifiers.
   - Analyze whether the expression data uses standardized human gene symbols
   - If non-standard identifiers are used, which means gene mapping is needed, then proceed with gene annotation and 
     mapping steps; otherwise, jump directly to data integration

STEP5. Gene Annotation Extraction
   When mapping is needed, we extract probe-gene relationships from the platform annotation.
   - Extract the mapping information from the SOFT file
   - Identify the appropriate columns containing probe IDs and corresponding gene symbols
   - Observe the annotation data to verify its completeness and quality.

STEP6. Gene Symbol Mapping
   The relationship between probes and genes is often many-to-many, requiring careful handling:
   - For one probe mapping to multiple genes:
     * Split the probe's expression value equally among all target genes
     * This maintains the total expression signal while avoiding bias
   - For multiple probes mapping to one gene:
     * Sum the contributions from all probes
     * This captures the total expression while accounting for split values
   - Example: If probe P1 maps to genes G1 and G2, and probe P2 maps to G2:
     * G1 receives 0.5 × P1
     * G2 receives (0.5 × P1) + P2

STEP7. Data Integration and Missing Value Handling
   The final step ensures data quality while maximizing usable samples and features.
   - Normalize gene symbols to ensure consistency across the dataset
   - If clinical data is available(trait data, age, or gender - any one suffices), integrate clinical and genetic data, ensuring proper sample alignment
   - Apply systematic missing value handling:
     * Remove genes with excessive missing values (>20%) to maintain data reliability
     * Filter out samples with too many missing gene measurements (>5%)
     * Carefully impute remaining missing values: use mode imputation for gender and mean for other features.
   - Proceed with saving the processed dataset.
   CRITICAL: Do NOT insert any extra steps between these sub-steps. Specifically: No "Check for severe bias"!!!
"""

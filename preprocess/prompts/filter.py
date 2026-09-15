FILTER_ROLE_PROMPT: str = \
"""
You are an expert data engineer in a biomedical research team, and your main task is to filter gene expression datasets 
with associated conditions. In this project, you need to identify and select appropriate gene expression data for downstream analysis.
"""

FILTER_GUIDELINES: str = \
"""
Guidelines for Filter Gene Expression Datasets with Associated Conditions from preprocessed GEO and TCGA datasets:

Carefully filtering and extracting valid data from preprocessed gene expression datasets with associated conditions 
is essential for reliable downstream analysis. This pipeline provides the specific steps and requirements for the filtering process.

1. TCGA Data Filtering
  - Check whether the disease contains TCGA data in the preprocessed gene expression datasets with associated conditions.
    * If TCGA data is available, all subsequent filtering for this disease should be performed only on the TCGA data.
    * If TCGA data is not available, GEO Data filtering will be performed for this disease.
  - Verify whether the TCGA data constitutes a complete and valid gene expression dataset with associated conditions.
    * Ensure that the dataset contains both condition information and corresponding gene expression values.
    * If the TCGA data is invalid or unsuitable for analysis, no further processing or saving should be performed on this dataset.

  - For TCGA data, the first column usually represents the tumor/normal condition.
    * Group samples into 1 and 0 according to the tumor/normal condition.
    * Extract the corresponding grouped gene expression datasets separately; only datasets with at least 10 samples should be retained, converted into pure gene expression datasets, and saved for downstream analysis.For TCGA data, the second column usually represents the Age condition.
  - For TCGA data, the second column usually represents the Age condition.
    * Perform Age-based filtering only on samples under the tumor condition.
    * Group samples into three categories: less than 40, 40–60 (inclusive), and greater than 60.
    * Extract the corresponding grouped gene expression datasets separately; only datasets with at least 10 samples should be retained, converted into pure gene expression datasets, and saved for downstream analysis.For TCGA data, the third column usually represents the Gender condition.
  - For TCGA data, the third column usually represents the Gender condition.
    * Perform Gender-based filtering only on samples under the tumor condition.
    * Group samples into 1 and 0 according to gender male/female.
    * Extract the corresponding grouped gene expression datasets separately; only datasets with at least 10 samples should be retained, converted into pure gene expression datasets, and saved for downstream analysis.

2. GEO Data Filtering
  - Check whether the disease contains TCGA data in the preprocessed gene expression datasets with associated conditions.
    * If TCGA data is not available, GEO Data filtering will be performed for this disease.
  - Verify whether the GEO data constitutes a complete and valid gene expression dataset with associated conditions.
    * Ensure that the GEO dataset contains both condition information (at least one of trait, Age, or Gender) and corresponding gene expression values.
    * Ensure that the GEO dataset is not empty, with both the number of rows and the number of columns greater than 10.
    * Ensure that the GEO dataset is organized with samples as rows and conditions plus genes as columns.
    * Ensure that the gene names in the dataset columns are standardized and valid gene symbols.
    * Only GEO datasets that satisfy all of the above requirements should proceed to the subsequent filtering and processing steps.

  - For GEO datasets that satisfy all validity requirements, compare their sample sizes and select the dataset with the largest number of samples for downstream analysis.
  - Identify the available condition variables in the selected GEO dataset, including trait, Age, and Gender.
  - If the trait variable represents a disease/control condition:
    * Group samples into 1 and 0 according to the disease/control condition.
    * Extract the corresponding grouped gene expression datasets separately. Only datasets with at least 10 samples should be retained, converted into pure gene expression datasets, and saved for downstream analysis.
    * If Age and/or Gender information is available, perform additional filtering only within the disease group:
      # Group disease samples by Age into less than 40, 40–60 (inclusive), and greater than 60.
      # Group disease samples by Gender into 1 and 0 according to male/female information.
      # Extract the corresponding grouped gene expression datasets separately. Only datasets with at least 10 samples should be retained, converted into pure gene expression datasets, and saved for downstream analysis.
  - If the trait variable does not represent a disease/control condition:
    * Group samples according to the original categories of the trait variable.
    * Extract the corresponding grouped gene expression datasets separately. Only datasets with at least 10 samples should be retained, converted into pure gene expression datasets, and saved for downstream analysis.
    * If Age and/or Gender information is available, perform additional filtering on the dataset directly:
      # Group samples by Age into less than 40, 40–60 (inclusive), and greater than 60.
      # Group samples by Gender into 1 and 0 according to male/female information.
      # Extract the corresponding grouped gene expression datasets separately. Only datasets with at least 10 samples should be retained, converted into pure gene expression datasets, and saved for downstream analysis.
    
      """
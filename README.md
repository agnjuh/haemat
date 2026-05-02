# HAEMAT: Cross-dataset immune cell classification and ambiguity analysis
_A framework for uncertainty-aware analysis of immune cell states and diagnostic patterns_

This repository implements an interpretable machine learning framework for cell type classification in single-cell RNA-seq (scRNA-seq) data, with a focus on cross-dataset generalisation and uncertainty-aware analysis.

The workflow is designed to evaluate how well a model trained on one PBMC dataset transfers to another independent dataset, while explicitly analysing prediction uncertainty and systematic failure modes.

---

## Data

- Training dataset: PBMC 5k (10x Genomics)
- External validation dataset: PBMC 10k (10x Genomics)

Cell identities in the validation dataset are treated as **proxy labels**.  
These labels are used as a reference for evaluation but are not assumed to be perfectly accurate ground truth.

---

## Methods

The pipeline includes:

- Logistic regression classifier (L1-regularised)
- Marker-informed feature space
- Cross-dataset evaluation (train on PBMC5k → test on PBMC10k)
- Uncertainty quantification:
  - Entropy
  - Top-2 probability gap
- Post hoc decision layer
- Benchmark comparison with CellTypist
- Gene-level interpretability (L1 coefficients)

---

## Summary figure

![Summary](figures/HAEMAT_summary_figure.png)

---

## Key results

### 1. Classification agreement by cell type

Agreement between HAEMAT predictions and proxy labels:

| Cell type | Agreement rate |
|----------|----------------|
| B        | 0.985 |
| DC       | 0.895 |
| Mono     | 0.972 |
| NK       | 0.585 |
| Platelet | 0.844 |
| T        | 0.993 |

Performance is high for most lineages, while NK cells show substantially lower agreement.

---

### 2. NK cell misclassification patterns

True NK cells are distributed across multiple predicted classes:

- NK → NK: 551  
- NK → Platelet: 609  
- NK → Monocyte: 383  
- NK → T: 59  

This indicates a systematic ambiguity between NK cells and other immune populations, particularly platelets and monocytes.

A similar pattern is observed in CellTypist, suggesting that this reflects biological or representation-level overlap rather than a model-specific issue.

---

### 3. Comparison with CellTypist

Both HAEMAT and CellTypist show:

- Strong performance for B cells, T cells, and monocytes
- Reduced agreement for NK cells

The consistency of these patterns across models indicates that the main limitations arise from dataset characteristics rather than implementation details.

---

### 4. Uncertainty under dataset shift

Prediction entropy differs between correct and incorrect predictions:

- Correct predictions are concentrated at low entropy
- Incorrect predictions show broader and higher entropy distribution

This demonstrates that uncertainty metrics provide meaningful signals of prediction reliability under domain shift.

---

### 5. Gene-level interpretability

The model identifies biologically meaningful marker genes:

- B cells: MS4A1, BANK1, CD79A  
- Dendritic cells: FCER1A, CLEC10A, CD1C  
- Monocytes: S100A8, AIF1  

This confirms that the classifier captures lineage-specific transcriptional programs.

---

## Interpretation

The results show that:

- Cell type classification generalises well across datasets for major lineages
- NK cells represent a consistent failure mode across independent models
- Prediction uncertainty correlates with classification correctness
- Interpretable models retain biologically relevant signal under domain shift

---

## Reproducibility

The full workflow is implemented in Snakemake:

```bash
snakemake --cores 4
```

---

## Repository Structure

```
workflow/
    Snakefile

scripts/
    run_pipeline.py
    cross_dataset.py
    decision_layer.py
    analyze_thresholds_per_class.py
    plot_benchmark_comparison.py
    hierarchical_models.py

results/
    benchmark_plots/
    cross_dataset/
    hierarchical/
```

---

## Scope and future direction

This repository focuses on rigorous evaluation and interpretability of PBMC cell type classification under dataset shift. Rather than treating classification errors only as technical failures, the analysis uses disagreement, uncertainty, and cell-type ambiguity as signals that may reveal biologically meaningful structure.

A key future direction is to extend this framework to disease-associated PBMC datasets and ask:

- whether immune cell composition differs systematically between disease groups
- whether ambiguous or transitional cell states are enriched in specific conditions
- whether uncertainty-aware cell type profiles can improve predictive modelling
- whether cytotoxic, myeloid, or platelet-associated signatures contribute to disease stratification

In this sense, HAEMAT is intended as a foundation for interpretable immune profiling, where classification outputs are not only labels, but structured features for downstream biological and predictive analysis.


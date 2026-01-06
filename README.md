# Machine learning-based classification of immune cell types in scRNA-seq data (PBMC) with gene-level interpretability and uncertainty analysis ![workflow](https://img.shields.io/badge/Development-Active-orange) ![status](https://img.shields.io/badge/Status-Work_in_progress-blue)

Overview
Core pipeline for PBMC scRNA-seq classification with:
	•	leak-free train/test split and preprocessing,
	•	proxy labels (Leiden + curated marker scoring) when annotations are missing,
	•	three baseline models (kNN, multinomial LR, Random Forest),
	•	gene-level interpretability (PCA back-projection; technical genes filtered),
	•	uncertainty analysis (coverage–accuracy, calibration curve, Brier score),
	•	optional per-class probability thresholds.

Repository layout:
	•	run_pipeline.py — main pipeline (reads data/5k_pbmc_10x.h5ad, optional markers_curated.csv, writes to results/).
	•	analyze_thresholds_per_class.py — finds class-specific probability thresholds from results/ arrays.
	•	environment.yml — conda environment spec.
	•	.gitignore

Requirements:
	•	Python 3.10+ (tested with 3.13)
	•	Conda or mamba (recommended)

Environment:
conda env create -f environment.yml
conda activate pbmc-ml

Data: place the 10x 5k PBMC .h5ad at: data/5k_pbmc_10x.h5ad

Notes:
	•	All preprocessing that can leak information (HVGs, scaling, PCA) is fitted on train only and applied to test.
	•	Gene-level summaries filter generic signals (HLA-, ribosomal RPL/RPS, mitochondrial MT-) to improve interpretability.

Cite:
if you use this code in coursework or reports, please cite the repository:
Juhasz, A. J. (2025). pbmc-haem-core: Machine learning-based classification of immune cell types in scRNA-seq data with gene-level interpretability and uncertainty analysis [Source code]. GitHub. https://github.com/agnjuh/haemat

This repository is licensed under a custom All Rights Reserved license (see LICENSE file)


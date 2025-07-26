# StateCNV

**A Bayesian State-Space Method for Copy Number Variation Detection and Thalassemia Profiling**

## Summary

This repository provides a full implementation of the method described in the paper:

> *"Classifying Copy Number Variations Using State Space Modeling of Targeted Sequencing Data: A Case Study in Thalassemia"*

We introduce a probabilistic framework for detecting and classifying copy number variation (CNV) profiles from targeted amplicon sequencing data. The method combines state-space modeling, auxiliary particle filtering, and Bayesian decision theory to robustly estimate copy number ratios, assign genetic profiles (e.g., thalassemia subtypes), and detect low-quality clinical samples using Bayesian evidence.

📄 **Paper:** [https://arxiv.org/abs/2504.10338](https://arxiv.org/abs/2504.10338)

---

## Features

- Accurate CNV estimation from targeted sequencing (amplicon-level resolution)
- State-space model with Laplace likelihood and change-point detection
- Auxiliary particle filtering for robust inference
- Bayesian profile assignment with built-in priors
- Automated sample quality control via Bayesian evidence
- Includes unit tests and reproducible demo scripts

---

## Installation

### 🔧 Requirements

- Python ≥ 3.8
- NumPy, SciPy, pandas, matplotlib
- `pytest` (for testing)
- particles

You can install everything using pip:

```bash
# Clone the repository
git clone https://github.com/yourusername/CNVBayesProfiler.git
cd CNVBayesProfiler

# Install dependencies and the package
pip install -e .

<p align="center">
  <img src="assets/statecnv_logo.png" alt="StateCNV logo" width="400"/>
</p>

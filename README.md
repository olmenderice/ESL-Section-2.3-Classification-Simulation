# ESL Section 2.3 — Classification Simulation

This repository contains the R code used for my assignment reproducing
Figures 2.1–2.4 from Section 2.3 of *The Elements of Statistical Learning*
by Hastie, Tibshirani, and Friedman.

## Objective

The objective is to reproduce the main classification examples using
new simulated datasets generated with fixed random seeds rather than
using the original textbook observations.

Two simulation scenarios are considered:

### Scenario 1
Each class is generated from a single bivariate Gaussian distribution.

- Training sample: 200 observations
  - 100 Class 0
  - 100 Class 1
- Test sample: 10,000 observations
  - 5,000 Class 0
  - 5,000 Class 1

### Scenario 2
Each class is generated from a mixture of 10 low-variance Gaussian
distributions.

- 10 Gaussian component centers per class
- Training sample: 200 observations
- Test sample: 10,000 observations

## Figures reproduced

The R script reproduces textbook-style versions of:

- Figure 2.1 — Linear regression of a 0/1 response
- Figure 2.2 — 15-nearest-neighbor classifier
- Figure 2.3 — 1-nearest-neighbor classifier
- Figure 2.4 — Training and test error versus model flexibility

The analysis is repeated for both simulation scenarios.

## Methods

The assignment compares:

- Least-squares classification
- K-nearest neighbors
- Training misclassification error
- Test misclassification error
- Bayes classification error
- Model flexibility using approximately N/k degrees of freedom

## Reproducibility

Fixed random seeds are used throughout the simulation so that the
results can be reproduced.

Main seeds:

- Scenario 1 training data: `20261008`
- Scenario 1 test data: `20261009`
- Scenario 2 Gaussian centers: `20261010`
- Scenario 2 training data: `20261011`
- Scenario 2 test data: `20261012`

## R packages

The analysis uses:

- `MASS`
- `class`

## Code

The complete commented R script is available in:

`ESL Assignment 1.R`

## Reference

Hastie, T., Tibshirani, R., & Friedman, J.  
*The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd Edition.

# Stroke Prediction: Machine Learning Analysis

A machine learning project for predicting stroke occurrence from patient health records, combining supervised classification with unsupervised clustering to identify at-risk patient profiles.

## Overview

This notebook analyzes the Kaggle Healthcare Stroke Dataset to:

1. Explore and clean patient clinical and demographic data
2. Train and compare multiple supervised classification models to predict stroke risk
3. Apply unsupervised learning to discover natural patient groupings and validate them against known stroke outcomes

## Dataset

- **Source:** [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **File:** `healthcare-dataset-stroke-data.csv`
- **Records:** 5,110 patients, 12 columns (before cleaning)
- **Target variable:** `stroke` (binary: 1 = stroke, 0 = no stroke)
- **Features:** gender, age, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, and smoking status
- **Class balance:** The dataset is highly imbalanced, with stroke cases representing a small minority of records

## Methodology

### 1. Data Cleaning and Preprocessing
- Removed the non-informative `id` column
- Excluded the single record with gender "Other" to keep modeling scope consistent
- Encoded binary categorical variables (gender, marital status, residence type)
- One-hot encoded multi-category variables (work type, smoking status)
- Split data into training and test sets (75/25) with stratification **before** fitting any preprocessing step, to prevent test-set leakage
- Imputed missing BMI values using the median, fit only on the training set
- Applied IQR-based outlier capping to age, average glucose level, and BMI, with bounds learned only from the training set

### 2. Exploratory Data Analysis
- Missing value inspection and BMI distribution analysis
- Class distribution and correlation analysis
- Visual comparison of key clinical variables (age, glucose level, BMI) against stroke outcome
- Relationship between stroke and categorical risk factors (hypertension, heart disease)
- Age density comparison by stroke outcome

### 3. Supervised Learning
The following classifiers were trained and evaluated, with class imbalance addressed via class weighting or sample weighting:

| Model | Notes |
|---|---|
| Decision Tree | max_depth=5, balanced class weights |
| Support Vector Machine (RBF kernel) | Trained on standardized features |
| Random Forest | 300 estimators, max_depth=8, balanced class weights |
| Bagging | Decision tree base estimator, 150 estimators |
| AdaBoost | 200 estimators, balanced sample weights |
| Gradient Boosting | 200 estimators, max_depth=3, balanced sample weights |

Each model is evaluated using accuracy, precision, recall, F1-score, ROC-AUC, a confusion matrix, and a full classification report. Feature importance is visualized for tree-based models. Final comparison includes a multi-metric bar chart and combined ROC curves across all models.

### 4. Unsupervised Learning
- **K-Means clustering** on a reduced clinical feature set (age, glucose level, BMI, hypertension, heart disease), with cluster count selected via the elbow method and silhouette score
- **Cluster profiling** to examine average feature values and stroke rate per cluster
- **PCA** (2 components) for visualizing cluster structure and true stroke labels in reduced dimensions
- **Hierarchical (Agglomerative) clustering** with Ward linkage, visualized via dendrogram on a sample of patients
- **Adjusted Rand Index (ARI)** to quantify agreement between clustering assignments and true stroke labels

## Results

### Class Distribution
The target is highly imbalanced: 4,860 non-stroke cases (95.13%) vs. 249 stroke cases (4.87%).

### Supervised Model Performance (Test Set)

Metrics below are for the "Stroke" (positive) class unless noted, computed on the held-out test set (1,278 patients, 62 stroke cases):

| Model | Accuracy | Precision (Stroke) | Recall (Stroke) | F1 (Stroke) |
|---|---|---|---|---|
| Decision Tree | 0.66 | 0.10 | 0.74 | 0.18 |
| SVM (RBF kernel) | 0.77 | 0.13 | 0.65 | 0.21 |
| Random Forest | 0.85 | 0.15 | 0.45 | 0.22 |
| Bagging (Decision Tree base) | 0.77 | 0.14 | 0.73 | 0.24 |
| AdaBoost | 0.71 | 0.12 | 0.77 | 0.21 |
| Gradient Boosting | 0.84 | 0.15 | 0.52 | 0.23 |

**Key takeaway:** Random Forest and Gradient Boosting achieve the highest overall accuracy, but AdaBoost, Decision Tree, and Bagging recover the most actual stroke cases (recall), reflecting the classic precision/recall trade-off under severe class imbalance. Given the clinical cost of missing a stroke case, models with higher recall (AdaBoost, Decision Tree, Bagging) may be preferable despite their lower precision and accuracy.

### Unsupervised Clustering

- **K-Means (k=2)** separated patients into a younger, lower-risk cluster (mean age ~38, stroke rate 2.69%) and an older, higher-risk cluster (mean age ~63, stroke rate 13.39%), showing that clustering on clinical features alone recovers a meaningful risk split without using the stroke label.
- **PCA (2 components)** explained 54.5% of variance in the clustering feature set.
- **Agglomerative clustering (Ward linkage, k=2)** achieved a silhouette score of 0.487.
- **Adjusted Rand Index** against true stroke labels: 0.111 (K-Means) and 0.115 (Agglomerative) — indicating the unsupervised clusters align only loosely with actual stroke outcomes, as expected given that clustering optimizes for feature similarity rather than label separation.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

## Usage

1. Place `healthcare-dataset-stroke-data.csv` in the expected data directory (the notebook currently reads from `/content/`, matching a Google Colab environment; update the path if running elsewhere).
2. Run the notebook cells sequentially from top to bottom.
3. Review the generated visualizations and the final model comparison table to assess model performance.

## Notes and Limitations

- The dataset is imbalanced, so recall and F1-score are more informative than raw accuracy when evaluating clinical usefulness.
- IQR capping and imputation are fit exclusively on the training set to avoid data leakage into the test set.
- Points beyond the boxplot whiskers represent statistical outliers rather than necessarily invalid or erroneous patient records.
- This project is intended for educational and exploratory purposes and is not validated for clinical decision-making.

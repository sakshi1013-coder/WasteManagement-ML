# Waste Generation Analysis Using Machine Learning

**Case Study No. 73 — B.Tech CSE Machine Learning Final Examination Project**

---

## Project Overview

This project investigates factors associated with changes in municipal solid waste (MSW)
generation, as specified in Case Study No. 73.

> **Official Problem Statement:**
> "A municipal organization wants to investigate factors associated with changes in waste generation."

---

## Dataset

- **Source:** Kaggle — *What A Waste Global Dataset* (`mannmann2/what-a-waste-global-dataset`), World Bank
- **File:** `data/country_level_data_0.csv` (217 rows, 51 columns)
- **Target:** `total_msw_total_msw_generated_tons_year` — annual MSW in metric tons

### Key Predictors

| Feature (original name) | Short name | Type |
|------------------------|------------|------|
| `gdp` | `gdp` | Numeric |
| `population_population_number_of_people` | `population` | Numeric |
| `income_id` | `income` | Categorical |
| `region_id` | `region` | Categorical |
| `composition_food_organic_waste_percent` | `organic_pct` | Numeric |
| `composition_paper_cardboard_percent` | `paper_pct` | Numeric |
| `composition_plastic_percent` | `plastic_pct` | Numeric |
| `composition_glass_percent` | `glass_pct` | Numeric |
| `composition_metal_percent` | `metal_pct` | Numeric |
| `composition_other_percent` | `other_pct` | Numeric |

---

## ML Models (Case Study 73 — 5 Core Models)

| # | Model | Task |
|---|-------|------|
| 1 | Linear Regression | Regression |
| 2 | K-Nearest Neighbors (KNN) | Regression |
| 3 | Random Forest | Regression |
| 4 | Logistic Regression | Classification (Waste Tiers: Low/Med/High) |
| 5 | Hierarchical Clustering | Unsupervised Pattern Discovery |

---

## Results Summary

### Regression (Target: Annual Waste Generation in Tons)

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Random Forest** | **1.23M** | **2.51M** | **0.9446** |
| Linear Regression | 2.46M | 5.53M | 0.7322 |
| KNN Regressor | 4.22M | 10.38M | 0.0562 |

### Classification (Target: Waste Tier — Low, Medium, High)

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|-------|----------|-------------------|-----------------|-------------|
| **Logistic Regression** | **69.77%** | **77.38%** | **69.77%** | **68.97%** |

### Unsupervised Clustering

- **Hierarchical Clustering (Agglomerative)**: 3 distinct developmental clusters identified via Ward linkage dendrogram, mapping directly to developmental waste tiers.

---

## Project Files

```
ML final project/
├── data/
│   └── country_level_data_0.csv
├── Waste_Generation_Analysis.ipynb   ← Primary deliverable
├── app.py                            ← Streamlit application
├── requirements.txt
└── README.md
```

---

## How to Run

### Jupyter Notebook

```bash
jupyter notebook Waste_Generation_Analysis.ipynb
```

*(The notebook is fully pre-executed with all outputs, charts, and tables embedded.)*

### Streamlit Application

```bash
streamlit run app.py
```

---

## Requirements

```bash
pip install -r requirements.txt
```

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

## ML Models (Case Study 73 — 7 Required Models)

| # | Model | Task |
|---|-------|------|
| 1 | Linear Regression | Regression |
| 2 | K-Nearest Neighbors (KNN) | Regression + Classification |
| 3 | Decision Tree | Regression + Classification |
| 4 | Random Forest | Regression + Classification |
| 5 | Logistic Regression | Classification |
| 6 | Naive Bayes | Classification |
| 7 | Hierarchical Clustering | Unsupervised |

---

## Results Summary

### Regression (R² Score, higher is better)

| Model | R² Score |
|-------|----------|
| Random Forest | ~0.95 |
| Decision Tree | ~0.90 |
| Linear Regression | ~0.72 |
| KNN Regressor | ~0.50 |

### Classification (Accuracy on Low/Medium/High Waste Tier)

| Model | Accuracy |
|-------|----------|
| Decision Tree | ~86% |
| Random Forest | ~84% |
| Logistic Regression | ~60% |
| KNN Classifier | ~58% |
| Naive Bayes | ~51% |

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

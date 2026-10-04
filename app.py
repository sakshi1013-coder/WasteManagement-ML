
"""
app.py — Waste Generation Analysis
Streamlit application for Case Study No. 73

Run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, accuracy_score

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Waste Generation Analysis",
    page_icon="♻️",
    layout="wide",
)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("♻️ Waste Generation Analysis Using Machine Learning")
st.markdown(
    "**Case Study No. 73 — B.Tech CSE Machine Learning Final Examination Project**"
)
st.markdown(
    "*This application lets you predict annual municipal solid waste generation "
    "and the waste tier (Low / Medium / High) for a country using the trained ML models.*"
)
st.divider()

# ──────────────────────────────────────────────
# Load & prepare data (cached so it runs once)
# ──────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df_raw = pd.read_csv("data/country_level_data_0.csv")

    cols = {
        "country_name": "country",
        "region_id": "region",
        "income_id": "income",
        "gdp": "gdp",
        "population_population_number_of_people": "population",
        "total_msw_total_msw_generated_tons_year": "waste_tons",
        "composition_food_organic_waste_percent": "organic_pct",
        "composition_glass_percent": "glass_pct",
        "composition_metal_percent": "metal_pct",
        "composition_other_percent": "other_pct",
        "composition_paper_cardboard_percent": "paper_pct",
        "composition_plastic_percent": "plastic_pct",
    }
    df = df_raw[list(cols.keys())].rename(columns=cols)
    df = df.dropna(subset=["waste_tons"]).reset_index(drop=True)

    df["waste_tier"], cutoffs = pd.qcut(
        df["waste_tons"], q=3, labels=["Low", "Medium", "High"], retbins=True
    )

    num_features = [
        "gdp", "population", "organic_pct", "glass_pct",
        "metal_pct", "other_pct", "paper_pct", "plastic_pct",
    ]
    cat_features = ["income", "region"]

    X_raw = df[num_features + cat_features]
    y_reg = df["waste_tons"]
    y_clf = df["waste_tier"]

    X_train_raw, X_test_raw, y_reg_train, y_reg_test, y_clf_train, y_clf_test = (
        train_test_split(
            X_raw, y_reg, y_clf,
            test_size=0.20, random_state=42, stratify=y_clf,
        )
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(imputer.fit_transform(X_train_raw[num_features]))
    X_test_num = scaler.transform(imputer.transform(X_test_raw[num_features]))

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train_raw[cat_features])
    X_test_cat = encoder.transform(X_test_raw[cat_features])

    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    # Train all regression models
    reg_models = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    }
    reg_r2 = {}
    for name, model in reg_models.items():
        model.fit(X_train, y_reg_train)
        reg_r2[name] = round(r2_score(y_reg_test, model.predict(X_test)), 4)

    # Train all classification models
    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "Naive Bayes": GaussianNB(),
    }
    clf_acc = {}
    for name, model in clf_models.items():
        model.fit(X_train, y_clf_train)
        clf_acc[name] = round(accuracy_score(y_clf_test, model.predict(X_test)), 4)

    return (
        imputer, scaler, encoder,
        num_features, cat_features,
        reg_models, reg_r2,
        clf_models, clf_acc,
        cutoffs,
    )


with st.spinner("Loading data and training models..."):
    (
        imputer, scaler, encoder,
        num_features, cat_features,
        reg_models, reg_r2,
        clf_models, clf_acc,
        cutoffs,
    ) = load_and_train()

best_reg_name = max(reg_r2, key=reg_r2.get)
best_clf_name = max(clf_acc, key=clf_acc.get)

# ──────────────────────────────────────────────
# Sidebar — model performance summary
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("Model Performance")
    st.subheader("Regression (R² Score)")
    for name, val in sorted(reg_r2.items(), key=lambda x: -x[1]):
        flag = " ★" if name == best_reg_name else ""
        st.metric(label=name + flag, value=f"{val:.4f}")

    st.divider()
    st.subheader("Classification (Accuracy)")
    for name, val in sorted(clf_acc.items(), key=lambda x: -x[1]):
        flag = " ★" if name == best_clf_name else ""
        st.metric(label=name + flag, value=f"{val:.4f}")

    st.divider()
    st.markdown(
        "**★ = Best model** selected for prediction on this page."
    )

# ──────────────────────────────────────────────
# Main — input form
# ──────────────────────────────────────────────
st.header("Predict Waste Generation for a Country")
st.markdown(
    "Enter the country's features below. "
    "The app will predict the **annual waste volume** and the **waste tier**."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Economic & Demographic")
    gdp = st.number_input(
        "GDP (USD)",
        min_value=1e8, max_value=2.5e13, value=5e10,
        step=1e9, format="%.2e",
        help="Gross Domestic Product in US Dollars.",
    )
    population = st.number_input(
        "Population",
        min_value=10_000, max_value=1_500_000_000, value=20_000_000,
        step=1_000_000,
        help="Total national population.",
    )
    income = st.selectbox(
        "World Bank Income Group",
        options=["LIC", "LMC", "UMC", "HIC"],
        index=2,
        help="LIC=Low Income, LMC=Lower-Middle, UMC=Upper-Middle, HIC=High Income",
    )
    region = st.selectbox(
        "World Bank Region",
        options=["EAS", "ECS", "LCN", "MEA", "NAC", "SAS", "SSF"],
        index=1,
        help="World Bank geographic region code.",
    )

with col2:
    st.subheader("Waste Composition (%)")
    organic_pct = st.slider("Organic / Food Waste %", 0.0, 100.0, 48.0, 0.5)
    paper_pct   = st.slider("Paper / Cardboard %",    0.0, 100.0, 12.0, 0.5)
    plastic_pct = st.slider("Plastic %",               0.0, 100.0, 10.0, 0.5)
    glass_pct   = st.slider("Glass %",                 0.0, 100.0,  4.0, 0.5)
    metal_pct   = st.slider("Metal %",                 0.0, 100.0,  3.5, 0.5)
    other_pct   = st.slider("Other %",                 0.0, 100.0, 15.0, 0.5)

st.divider()

# ──────────────────────────────────────────────
# Preprocess input & predict
# ──────────────────────────────────────────────
if st.button("Predict", type="primary", use_container_width=True):
    new_num = pd.DataFrame(
        [[gdp, population, organic_pct, glass_pct, metal_pct, other_pct, paper_pct, plastic_pct]],
        columns=num_features,
    )
    new_cat = pd.DataFrame([[income, region]], columns=cat_features)

    new_num_prep = scaler.transform(imputer.transform(new_num))
    new_cat_prep = encoder.transform(new_cat)
    new_X = np.hstack([new_num_prep, new_cat_prep])

    # Regression
    reg_pred = reg_models[best_reg_name].predict(new_X)[0]

    # Classification
    clf_pred = clf_models[best_clf_name].predict(new_X)[0]

    st.header("Prediction Results")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(
            label=f"Predicted Annual Waste  [{best_reg_name}]",
            value=f"{reg_pred:,.0f} tons/year",
            delta=f"({reg_pred / 1e6:.2f} million tons/year)",
        )

    with res_col2:
        tier_colour = {"Low": "green", "Medium": "orange", "High": "red"}
        colour = tier_colour.get(clf_pred, "gray")
        st.markdown(
            f"**Waste Tier  [{best_clf_name}]**"
        )
        st.markdown(
            f"<h2 style='color:{colour};'>{clf_pred} Waste Generator</h2>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Waste Tier Reference")
    tier_df = pd.DataFrame({
        "Tier": ["Low", "Medium", "High"],
        "Annual Waste Threshold (tons/year)": [
            f"up to {cutoffs[1]:,.0f}",
            f"{cutoffs[1]:,.0f} to {cutoffs[2]:,.0f}",
            f"above {cutoffs[2]:,.0f}",
        ],
    })
    st.table(tier_df)

# ──────────────────────────────────────────────
# About section
# ──────────────────────────────────────────────
with st.expander("About this project"):
    st.markdown(
        """
**Case Study No. 73 — Waste Generation Analysis Using Machine Learning**

**Problem Statement:**  
"A municipal organization wants to investigate factors associated with changes in waste generation."

**Dataset:** World Bank *What A Waste* Global Dataset  
(Kaggle: `mannmann2/what-a-waste-global-dataset`)  
217 countries, 51 original columns.

**Models implemented:**
1. Linear Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Logistic Regression
6. Naive Bayes
7. Hierarchical Clustering (in notebook)

**Target variable:** `total_msw_total_msw_generated_tons_year` — annual MSW in metric tons.
        """
    )

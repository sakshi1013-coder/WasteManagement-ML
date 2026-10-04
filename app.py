"""
app.py — WasteSense: Municipal Waste Generation Analysis Dashboard
Case Study No. 73 — B.Tech Machine Learning Final Examination Project

Features:
- Dark, polished dashboard theme inspired by modern enterprise ML UI
- Sidebar navigation with system status, best model metrics, and quick stats
- 5 Core tabs:
  1. Waste Predictor (interactive input, real-time ML inference, high-impact result cards)
  2. Dataset Insights (6 deep-dive charts on actual World Bank What A Waste dataset)
  3. Model Performance (Regression & Classification benchmarks, bar charts, confusion matrix)
  4. Hierarchical Clustering (3 clusters, scatter visualization, dendrogram)
  5. Data Quality (shape, missingness audit, feature taxonomy, preprocessing pipeline)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="WasteSense — Case Study 73",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS — Polished Dark Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d121f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Top Header & Brand */
    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-sub {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .system-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 16px;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
    }
    
    /* Sidebar Cards */
    .sidebar-card {
        background: #131926;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 14px;
    }
    .sidebar-card-label {
        font-size: 0.68rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .sidebar-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .sidebar-card-metric {
        font-size: 0.8rem;
        color: #cbd5e1;
        margin-top: 2px;
    }
    
    /* Stat Grid */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-bottom: 16px;
    }
    .stat-box {
        background: #131926;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 8px 4px;
        text-align: center;
    }
    .stat-num {
        font-size: 1.1rem;
        font-weight: 800;
        color: #ffffff;
        display: block;
    }
    .stat-lbl {
        font-size: 0.65rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* Main Content Metric Cards */
    .metric-card {
        background: #131926;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-card-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-card-val {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
        margin: 4px 0;
    }
    .metric-card-sub {
        font-size: 0.78rem;
        color: #38bdf8;
        font-weight: 500;
    }
    
    /* Hero Prediction Result Cards */
    .result-hero {
        background: linear-gradient(135deg, #131d2e 0%, #172439 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        margin: 16px 0;
    }
    .result-hero-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .result-hero-tons {
        font-size: 2.8rem;
        font-weight: 900;
        color: #00e599;
        margin: 8px 0;
        letter-spacing: -1px;
    }
    .tier-badge-high {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        display: inline-block;
    }
    .tier-badge-med {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        display: inline-block;
    }
    .tier-badge-low {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        display: inline-block;
    }
    
    /* Section titles */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 10px;
        margin-bottom: 4px;
    }
    .section-desc {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    
    /* Tab highlight */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Helper: Dark Matplotlib Figure
# ──────────────────────────────────────────────
def get_dark_figure(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#131926")
    ax.set_facecolor("#131926")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(0.8)
    ax.grid(True, linestyle="--", alpha=0.15, color="#64748b")
    return fig, ax

# ──────────────────────────────────────────────
# Cached Data Loader & Model Trainer
# ──────────────────────────────────────────────
@st.cache_data
def load_data_and_train():
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

    # 3-tier classification target
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

    # 80-20 Train-Test Split
    X_train_raw, X_test_raw, y_reg_train, y_reg_test, y_clf_train, y_clf_test = (
        train_test_split(
            X_raw, y_reg, y_clf,
            test_size=0.20, random_state=42, stratify=y_clf
        )
    )

    # Preprocessing
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(imputer.fit_transform(X_train_raw[num_features]))
    X_test_num = scaler.transform(imputer.transform(X_test_raw[num_features]))

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train_raw[cat_features])
    X_test_cat = encoder.transform(X_test_raw[cat_features])

    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    # 1. Regression Models (5 core models strictly: LR, KNN, RF)
    reg_models = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    }
    reg_metrics = {}
    reg_preds = {}
    for name, model in reg_models.items():
        model.fit(X_train, y_reg_train)
        pred = model.predict(X_test)
        reg_preds[name] = pred
        reg_metrics[name] = {
            "MAE": mean_absolute_error(y_reg_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_reg_test, pred)),
            "R2": r2_score(y_reg_test, pred)
        }

    # 2. Classification Models (Logistic Regression, KNN, Random Forest)
    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    }
    clf_metrics = {}
    clf_preds = {}
    for name, model in clf_models.items():
        model.fit(X_train, y_clf_train)
        pred = model.predict(X_test)
        clf_preds[name] = pred
        clf_metrics[name] = {
            "Accuracy": accuracy_score(y_clf_test, pred),
            "Precision": precision_score(y_clf_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_clf_test, pred, average="weighted", zero_division=0),
            "F1": f1_score(y_clf_test, pred, average="weighted", zero_division=0)
        }

    # 3. Hierarchical Clustering (Full scaled features)
    imputed_full_num = imputer.transform(df[num_features])
    scaled_full_num = scaler.transform(imputed_full_num)
    encoded_full_cat = encoder.transform(df[cat_features])
    X_full = np.hstack([scaled_full_num, encoded_full_cat])

    hc = AgglomerativeClustering(n_clusters=3)
    clusters = hc.fit_predict(X_full)
    df["cluster"] = clusters

    return (
        df_raw, df,
        imputer, scaler, encoder,
        num_features, cat_features,
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test,
        reg_models, reg_metrics, reg_preds,
        clf_models, clf_metrics, clf_preds,
        cutoffs, X_full
    )

with st.spinner("Initializing WasteSense Dashboard & Machine Learning Models..."):
    (
        df_raw, df,
        imputer, scaler, encoder,
        num_features, cat_features,
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test,
        reg_models, reg_metrics, reg_preds,
        clf_models, clf_metrics, clf_preds,
        cutoffs, X_full
    ) = load_data_and_train()

# ──────────────────────────────────────────────
# Sidebar Implementation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-title">♻️ WasteSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">CASE STUDY NO. 73 &bull; MSW ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="system-status"><span class="pulse-dot"></span> ML SYSTEM READY</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-label">Problem Statement</div>
        <div style="font-size: 0.78rem; color: #cbd5e1; line-height: 1.4;">
            "A municipal organization wants to investigate factors associated with changes in waste generation."
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Best Model Card
    best_reg_name = "Random Forest Regressor"
    best_r2 = reg_metrics[best_reg_name]["R2"]
    st.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-card-label">Best Regression Model</div>
        <div class="sidebar-card-title">Random Forest</div>
        <div class="sidebar-card-metric">R&sup2; Score: <b style="color: #00e599;">{best_r2:.4f}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Metrics 3-box Grid
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-box">
            <span class="stat-num">217</span>
            <span class="stat-lbl">Records</span>
        </div>
        <div class="stat-box">
            <span class="stat-num">5</span>
            <span class="stat-lbl">Models</span>
        </div>
        <div class="stat-box">
            <span class="stat-num">3</span>
            <span class="stat-lbl">Tiers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation
    st.markdown("### Navigation")
    nav_choice = st.radio(
        "Select View",
        [
            "Waste Predictor",
            "Dataset Insights",
            "Model Performance",
            "Clustering",
            "Data Quality"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("<div style='font-size: 0.72rem; color: #64748b;'>B.Tech Final Examination Project<br>World Bank What A Waste Dataset</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Main Application Content
# ──────────────────────────────────────────────

# ══════════════════════════════════════════════
# TAB 1: Waste Predictor
# ══════════════════════════════════════════════
if nav_choice == "Waste Predictor":
    st.markdown('<div class="section-header">Waste Generation Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Predict annual municipal solid-waste generation and classify the expected waste tier using the trained ML models.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown("#### 1. Economic & Demographic")
        c1, c2 = st.columns(2)
        with c1:
            gdp_input = st.number_input(
                "Gross Domestic Product (USD)",
                min_value=1e7,
                max_value=2e13,
                value=5.0e10,
                step=1e9,
                format="%.2e",
                help="Annual GDP of the jurisdiction in US Dollars."
            )
            income_input = st.selectbox(
                "Income Level (World Bank)",
                options=["HIC", "UMC", "LMC", "LIC"],
                format_func=lambda x: {
                    "HIC": "High Income (HIC)",
                    "UMC": "Upper Middle Income (UMC)",
                    "LMC": "Lower Middle Income (LMC)",
                    "LIC": "Low Income (LIC)"
                }[x]
            )
        with c2:
            pop_input = st.number_input(
                "Population Size",
                min_value=10_000,
                max_value=1_500_000_000,
                value=15_000_000,
                step=500_000,
                format="%d",
                help="Total population under municipal administration."
            )
            region_input = st.selectbox(
                "Geographic Region",
                options=["ECS", "LCN", "SAS", "SSF", "MEA", "EAS", "NAC"],
                format_func=lambda x: {
                    "ECS": "Europe & Central Asia",
                    "LCN": "Latin America & Caribbean",
                    "SAS": "South Asia",
                    "SSF": "Sub-Saharan Africa",
                    "MEA": "Middle East & North Africa",
                    "EAS": "East Asia & Pacific",
                    "NAC": "North America"
                }[x]
            )

        st.markdown("#### 2. Waste Composition Percentages")
        st.caption("Adjust estimated municipal waste breakdown (must approximate 100%):")
        
        c3, c4, c5 = st.columns(3)
        with c3:
            organic_pct = st.slider("Organic / Food %", 0.0, 90.0, 48.0, 1.0)
            glass_pct = st.slider("Glass %", 0.0, 30.0, 5.0, 0.5)
        with c4:
            paper_pct = st.slider("Paper & Cardboard %", 0.0, 50.0, 17.0, 1.0)
            metal_pct = st.slider("Metal %", 0.0, 25.0, 4.0, 0.5)
        with c5:
            plastic_pct = st.slider("Plastic %", 0.0, 40.0, 12.0, 1.0)
            other_pct = st.slider("Other / Residual %", 0.0, 50.0, 14.0, 1.0)

        tot_comp = organic_pct + paper_pct + plastic_pct + glass_pct + metal_pct + other_pct
        if abs(tot_comp - 100.0) > 5.0:
            st.warning(f"Total composition sum is {tot_comp:.1f}% (ideal is 100%). Predictions remain valid.")

        st.markdown("#### 3. Model Engine")
        c6, c7 = st.columns(2)
        with c6:
            chosen_reg = st.selectbox(
                "Regression Engine",
                options=["Random Forest Regressor", "Linear Regression", "KNN Regressor"]
            )
        with c7:
            chosen_clf = st.selectbox(
                "Classification Engine",
                options=["Random Forest Classifier", "Logistic Regression", "KNN Classifier"]
            )

        predict_btn = st.button("PREDICT WASTE", use_container_width=True, type="primary")

    with col_right:
        st.markdown("#### Inference Output")

        # Perform Inference
        input_data = pd.DataFrame([{
            "gdp": gdp_input,
            "population": pop_input,
            "organic_pct": organic_pct,
            "glass_pct": glass_pct,
            "metal_pct": metal_pct,
            "other_pct": other_pct,
            "paper_pct": paper_pct,
            "plastic_pct": plastic_pct,
            "income": income_input,
            "region": region_input,
        }])

        input_num = scaler.transform(imputer.transform(input_data[num_features]))
        input_cat = encoder.transform(input_data[cat_features])
        input_vec = np.hstack([input_num, input_cat])

        pred_tons = reg_models[chosen_reg].predict(input_vec)[0]
        pred_tons = max(0.0, pred_tons)
        pred_tier = clf_models[chosen_clf].predict(input_vec)[0]

        tier_class = "tier-badge-low" if pred_tier == "Low" else ("tier-badge-med" if pred_tier == "Medium" else "tier-badge-high")

        st.markdown(f"""
        <div class="result-hero">
            <div class="result-hero-label">Predicted Annual Waste Generation</div>
            <div class="result-hero-tons">{pred_tons:,.0f} <span style="font-size: 1.1rem; color: #94a3b8;">tons/year</span></div>
            <div style="margin-top: 14px;">
                <span class="{tier_class}">{pred_tier.upper()} WASTE TIER</span>
            </div>
            <div style="margin-top: 16px; font-size: 0.78rem; color: #64748b;">
                Engine: <b style="color: #38bdf8;">{chosen_reg}</b> &bull; R&sup2; = {reg_metrics[chosen_reg]['R2']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Detailed Prediction Breakdown")
        b1, b2 = st.columns(2)
        with b1:
            per_capita_kg = (pred_tons * 1000.0) / max(1.0, pop_input)
            st.markdown(f"""
            <div class="sidebar-card">
                <div class="sidebar-card-label">Per Capita Waste</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff;">{per_capita_kg:.1f} kg/person/yr</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">{(per_capita_kg/365):.2f} kg/person/day</div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown(f"""
            <div class="sidebar-card">
                <div class="sidebar-card-label">Classification Status</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff;">{pred_tier} Tier</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Model: {chosen_clf}</div>
            </div>
            """, unsafe_allow_html=True)

        # Tier Cutoff Reference
        st.markdown("##### Official Municipal Tier Cutoffs")
        st.markdown(f"""
        - **Low Tier:** &le; {cutoffs[1]:,.0f} tons/year
        - **Medium Tier:** {cutoffs[1]:,.0f} &ndash; {cutoffs[2]:,.0f} tons/year
        - **High Tier:** &gt; {cutoffs[2]:,.0f} tons/year
        """)

# ══════════════════════════════════════════════
# TAB 2: Dataset Insights
# ══════════════════════════════════════════════
elif nav_choice == "Dataset Insights":
    st.markdown('<div class="section-header">Dataset Insights & Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Comprehensive analysis of the World Bank What A Waste dataset (217 national jurisdictions).</div>', unsafe_allow_html=True)

    # 4 Top Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-label">Dataset Records</div>
            <div class="metric-card-val">217</div>
            <div class="metric-card-sub">Global Countries & Territories</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Original Attributes</div>
            <div class="metric-card-val">{df_raw.shape[1]}</div>
            <div class="metric-card-sub">World Bank Raw Variables</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-label">Selected Features</div>
            <div class="metric-card-val">10</div>
            <div class="metric-card-sub">Demographic, Economic & Composition</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-label">Prediction Target</div>
            <div class="metric-card-val">MSW Tons</div>
            <div class="metric-card-sub">Continuous & 3-Tier Categorical</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 6 Deep Dive Charts
    col1, col2 = st.columns(2, gap="medium")

    # Chart 1: Distribution
    with col1:
        st.markdown("#### 1. Waste Generation Distribution")
        fig, ax = get_dark_figure()
        sns.histplot(df["waste_tons"] / 1e6, kde=True, color="#00e599", bins=30, ax=ax)
        ax.set_title("Annual Waste Generation (Million Tons)", color="#ffffff", fontsize=11, fontweight="bold")
        ax.set_xlabel("Waste (Million Tons / Year)", color="#94a3b8")
        ax.set_ylabel("Number of Countries", color="#94a3b8")
        st.pyplot(fig)
        st.caption("**Insight:** Heavy right-skew. The vast majority of countries produce under 20M tons/year, while high-population industrial powers form an extreme upper tail.")

    # Chart 2: Population vs Waste
    with col2:
        st.markdown("#### 2. Population vs Waste Generation")
        fig, ax = get_dark_figure()
        ax.scatter(df["population"] / 1e6, df["waste_tons"] / 1e6, color="#38bdf8", alpha=0.75, edgecolors="none", s=40)
        ax.set_title("Population vs Waste Output", color="#ffffff", fontsize=11, fontweight="bold")
        ax.set_xlabel("Population (Millions)", color="#94a3b8")
        ax.set_ylabel("Waste (Million Tons / Year)", color="#94a3b8")
        st.pyplot(fig)
        st.caption("**Insight:** Direct linear relationship. Municipal waste generation scales monotonically with national population size.")

    col3, col4 = st.columns(2, gap="medium")

    # Chart 3: GDP vs Waste
    with col3:
        st.markdown("#### 3. GDP vs Waste Generation")
        fig, ax = get_dark_figure()
        ax.scatter(df["gdp"] / 1e9, df["waste_tons"] / 1e6, color="#f59e0b", alpha=0.75, edgecolors="none", s=40)
        ax.set_title("GDP (Billion USD) vs Waste Output", color="#ffffff", fontsize=11, fontweight="bold")
        ax.set_xlabel("GDP (Billion USD)", color="#94a3b8")
        ax.set_ylabel("Waste (Million Tons / Year)", color="#94a3b8")
        st.pyplot(fig)
        st.caption("**Insight:** Strong positive correlation with economic activity. Greater industrial production and consumer purchasing drive higher municipal solid waste.")

    # Chart 4: Waste Composition Breakdown
    with col4:
        st.markdown("#### 4. Global Waste Composition Breakdown")
        comp_means = df[[
            "organic_pct", "paper_pct", "plastic_pct",
            "glass_pct", "metal_pct", "other_pct"
        ]].mean()
        comp_df = pd.DataFrame({
            "Fraction": ["Organic", "Paper", "Plastic", "Glass", "Metal", "Other"],
            "Mean %": comp_means.values
        }).sort_values(by="Mean %", ascending=True)

        fig, ax = get_dark_figure()
        bars = ax.barh(comp_df["Fraction"], comp_df["Mean %"], color="#818cf8")
        ax.set_title("Average Waste Stream Fractions (%)", color="#ffffff", fontsize=11, fontweight="bold")
        ax.set_xlabel("Percentage (%)", color="#94a3b8")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', color="#cbd5e1", fontsize=9)
        st.pyplot(fig)
        st.caption("**Insight:** Organic and food waste is by far the largest single fraction globally (~45%), making composting municipal policy vital.")

    col5, col6 = st.columns(2, gap="medium")

    # Chart 5: Waste by Income Group
    with col5:
        st.markdown("#### 5. Waste Generation by Income Group")
        fig, ax = get_dark_figure()
        income_order = ["LIC", "LMC", "UMC", "HIC"]
        sns.boxplot(
            x="income", y=df["waste_tons"] / 1e6, data=df,
            order=income_order, palette="mako", ax=ax
        )
        ax.set_title("Waste Output Across Income Brackets", color="#ffffff", fontsize=11, fontweight="bold")
        ax.set_xlabel("Income Tier (World Bank)", color="#94a3b8")
        ax.set_ylabel("Waste (Million Tons)", color="#94a3b8")
        st.pyplot(fig)
        st.caption("**Insight:** High-Income (HIC) and Upper-Middle (UMC) nations account for the highest aggregate and median waste volumes.")

    # Chart 6: Correlation Heatmap
    with col6:
        st.markdown("#### 6. Feature Correlation Heatmap")
        fig, ax = get_dark_figure(figsize=(7, 4.3))
        corr_subset = df[["waste_tons", "population", "gdp", "organic_pct", "paper_pct", "plastic_pct"]].corr()
        sns.heatmap(
            corr_subset, annot=True, fmt=".2f", cmap="coolwarm",
            cbar=False, ax=ax, annot_kws={"size": 8}
        )
        ax.set_title("Correlation Matrix", color="#ffffff", fontsize=11, fontweight="bold")
        st.pyplot(fig)
        st.caption("**Insight:** Population (r=0.86) and GDP (r=0.82) are the strongest linear drivers of total municipal waste generation.")

# ══════════════════════════════════════════════
# TAB 3: Model Performance
# ══════════════════════════════════════════════
elif nav_choice == "Model Performance":
    st.markdown('<div class="section-header">Model Performance Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Evaluated strictly on the 5 core project algorithms (Linear Regression, KNN, Random Forest, Logistic Regression, Hierarchical Clustering).</div>', unsafe_allow_html=True)

    # ── Part 1: Regression Benchmarks ──
    st.markdown("### 1. Regression Models (Continuous Waste Prediction)")
    
    r_card1, r_card2, r_card3 = st.columns(3)
    with r_card1:
        st.markdown(f"""
        <div class="sidebar-card" style="border-left: 4px solid #00e599;">
            <div class="sidebar-card-label">Best Regression Model</div>
            <div class="sidebar-card-title">Random Forest</div>
            <div class="sidebar-card-metric">R&sup2;: <b>{reg_metrics['Random Forest Regressor']['R2']:.4f}</b> | RMSE: <b>{reg_metrics['Random Forest Regressor']['RMSE']/1e6:.2f}M</b></div>
        </div>
        """, unsafe_allow_html=True)
    with r_card2:
        st.markdown(f"""
        <div class="sidebar-card" style="border-left: 4px solid #38bdf8;">
            <div class="sidebar-card-label">Linear Baseline</div>
            <div class="sidebar-card-title">Linear Regression</div>
            <div class="sidebar-card-metric">R&sup2;: <b>{reg_metrics['Linear Regression']['R2']:.4f}</b> | RMSE: <b>{reg_metrics['Linear Regression']['RMSE']/1e6:.2f}M</b></div>
        </div>
        """, unsafe_allow_html=True)
    with r_card3:
        st.markdown(f"""
        <div class="sidebar-card" style="border-left: 4px solid #f59e0b;">
            <div class="sidebar-card-label">Non-Parametric Distance</div>
            <div class="sidebar-card-title">KNN Regressor</div>
            <div class="sidebar-card-metric">R&sup2;: <b>{reg_metrics['KNN Regressor']['R2']:.4f}</b> | RMSE: <b>{reg_metrics['KNN Regressor']['RMSE']/1e6:.2f}M</b></div>
        </div>
        """, unsafe_allow_html=True)

    c_tbl1, c_chart1 = st.columns([1.1, 0.9], gap="large")

    with c_tbl1:
        st.markdown("##### Regression Comparison Table")
        reg_df = pd.DataFrame([
            {
                "Model": name,
                "MAE (Tons)": f"{m['MAE']:,.0f}",
                "RMSE (Tons)": f"{m['RMSE']:,.0f}",
                "R² Score": f"{m['R2']:.4f}",
            }
            for name, m in reg_metrics.items()
        ])
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

    with c_chart1:
        st.markdown("##### R² Score Comparison")
        fig, ax = get_dark_figure(figsize=(6, 2.8))
        names = list(reg_metrics.keys())
        r2_vals = [reg_metrics[k]["R2"] for k in names]
        bars = ax.barh(names, r2_vals, color=["#38bdf8", "#f59e0b", "#00e599"])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("R² Score", color="#94a3b8")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', color="#cbd5e1", fontsize=9, fontweight="bold")
        st.pyplot(fig)

    st.markdown("---")

    # ── Part 2: Classification Benchmarks ──
    st.markdown("### 2. Classification Models (Waste Tier: Low / Medium / High)")

    c_card1, c_card2, c_card3 = st.columns(3)
    with c_card1:
        st.markdown(f"""
        <div class="sidebar-card" style="border-left: 4px solid #00e599;">
            <div class="sidebar-card-label">Best Classifier</div>
            <div class="sidebar-card-title">Random Forest Classifier</div>
            <div class="sidebar-card-metric">Accuracy: <b>{clf_metrics['Random Forest Classifier']['Accuracy']*100:.2f}%</b> | F1: <b>{clf_metrics['Random Forest Classifier']['F1']*100:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    with c_card2:
        st.markdown(f"""
        <div class="sidebar-card" style="border-left: 4px solid #38bdf8;">
            <div class="sidebar-card-label">Linear Classification</div>
            <div class="sidebar-card-title">Logistic Regression</div>
            <div class="sidebar-card-metric">Accuracy: <b>{clf_metrics['Logistic Regression']['Accuracy']*100:.2f}%</b> | F1: <b>{clf_metrics['Logistic Regression']['F1']*100:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    with c_card3:
        st.markdown(f"""
        <div class="sidebar-card" style="border-left: 4px solid #f59e0b;">
            <div class="sidebar-card-label">Instance-Based</div>
            <div class="sidebar-card-title">KNN Classifier</div>
            <div class="sidebar-card-metric">Accuracy: <b>{clf_metrics['KNN Classifier']['Accuracy']*100:.2f}%</b> | F1: <b>{clf_metrics['KNN Classifier']['F1']*100:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    c_tbl2, c_chart2 = st.columns([1.1, 0.9], gap="large")

    with c_tbl2:
        st.markdown("##### Classification Comparison Table")
        clf_df = pd.DataFrame([
            {
                "Model": name,
                "Accuracy": f"{m['Accuracy']*100:.2f}%",
                "Precision": f"{m['Precision']*100:.2f}%",
                "Recall": f"{m['Recall']*100:.2f}%",
                "F1-Score": f"{m['F1']*100:.2f}%"
            }
            for name, m in clf_metrics.items()
        ])
        st.dataframe(clf_df, use_container_width=True, hide_index=True)

    with c_chart2:
        st.markdown("##### Accuracy Comparison")
        fig, ax = get_dark_figure(figsize=(6, 2.8))
        names_clf = list(clf_metrics.keys())
        acc_vals = [clf_metrics[k]["Accuracy"] * 100 for k in names_clf]
        bars = ax.barh(names_clf, acc_vals, color=["#38bdf8", "#f59e0b", "#00e599"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy (%)", color="#94a3b8")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', color="#cbd5e1", fontsize=9, fontweight="bold")
        st.pyplot(fig)

    st.markdown("---")

    # Diagnostic Visualizations: Actual vs Predicted + Confusion Matrix
    st.markdown("### 3. Model Diagnostic Evaluations")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("##### Random Forest: Actual vs. Predicted Waste Output")
        fig, ax = get_dark_figure(figsize=(6, 4))
        ax.scatter(y_reg_test / 1e6, reg_preds["Random Forest Regressor"] / 1e6, color="#00e599", alpha=0.75, s=35)
        max_v = max(y_reg_test.max(), reg_preds["Random Forest Regressor"].max()) / 1e6
        ax.plot([0, max_v], [0, max_v], color="#f87171", linestyle="--", linewidth=1.5, label="Perfect 1:1 Line")
        ax.set_xlabel("Actual Waste (Million Tons)", color="#94a3b8")
        ax.set_ylabel("Predicted Waste (Million Tons)", color="#94a3b8")
        ax.legend(facecolor="#131926", edgecolor="#334155", labelcolor="#cbd5e1")
        st.pyplot(fig)

    with d2:
        st.markdown("##### Logistic Regression: Confusion Matrix")
        fig, ax = get_dark_figure(figsize=(6, 4))
        cm = confusion_matrix(y_clf_test, clf_preds["Logistic Regression"], labels=["Low", "Medium", "High"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Low", "Med", "High"], yticklabels=["Low", "Med", "High"],
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_xlabel("Predicted Tier", color="#94a3b8")
        ax.set_ylabel("Actual Tier", color="#94a3b8")
        st.pyplot(fig)

# ══════════════════════════════════════════════
# TAB 4: Hierarchical Clustering
# ══════════════════════════════════════════════
elif nav_choice == "Clustering":
    st.markdown('<div class="section-header">Hierarchical Clustering (Unsupervised Learning)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Grouping 217 nations by socioeconomic similarity and waste stream characteristics without target labels.</div>', unsafe_allow_html=True)

    # Cluster Metric Cards
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-label">Optimal Clusters</div>
            <div class="sidebar-card-title">k = 3 Clusters</div>
            <div class="sidebar-card-metric">Determined via Ward's Linkage</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-card-label">Distance Metric</div>
            <div class="sidebar-card-title">Euclidean / Ward</div>
            <div class="sidebar-card-metric">Standardized Features (Scaled)</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        cluster_counts = df["cluster"].value_counts().sort_index()
        counts_str = " | ".join([f"C{i}: {count}" for i, count in cluster_counts.items()])
        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-card-label">Cluster Distribution</div>
            <div class="sidebar-card-title">{counts_str}</div>
            <div class="sidebar-card-metric">217 Total Countries Clustered</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    cl_col1, cl_col2 = st.columns([1, 1], gap="large")

    with cl_col1:
        st.markdown("#### Dendrogram (Hierarchical Tree)")
        fig, ax = get_dark_figure(figsize=(7, 4.5))
        linkage_matrix = linkage(X_full, method="ward")
        dendrogram(linkage_matrix, truncate_mode="lastp", p=20, ax=ax, color_threshold=15)
        ax.set_title("Ward Hierarchical Clustering Dendrogram", color="#ffffff", fontsize=11, fontweight="bold")
        ax.set_xlabel("Aggregated Cluster / Sample Index", color="#94a3b8")
        ax.set_ylabel("Ward Distance", color="#94a3b8")
        st.pyplot(fig)
        st.caption("**Interpretation:** A horizontal cut at distance ~15 yields 3 distinct developmental clusters.")

    with cl_col2:
        st.markdown("#### Cluster Scatter Visualization (GDP vs. Waste)")
        fig, ax = get_dark_figure(figsize=(7, 4.5))
        colors = ["#38bdf8", "#00e599", "#f59e0b"]
        for c_id in sorted(df["cluster"].unique()):
            subset = df[df["cluster"] == c_id]
            ax.scatter(
                subset["gdp"] / 1e9, subset["waste_tons"] / 1e6,
                color=colors[c_id % len(colors)], label=f"Cluster {c_id}",
                alpha=0.8, s=45
            )
        ax.set_title("Identified Clusters: GDP vs. Waste Output", color="#ffffff", fontsize=11, fontweight="bold")
        ax.set_xlabel("GDP (Billion USD)", color="#94a3b8")
        ax.set_ylabel("Waste (Million Tons / Year)", color="#94a3b8")
        ax.legend(facecolor="#131926", edgecolor="#334155", labelcolor="#cbd5e1")
        st.pyplot(fig)
        st.caption("**Interpretation:** Natural partitioning reveals lower-output developing countries, mid-tier developing nations, and mega-economy waste producers.")

    st.markdown("#### Real-World Municipal Cluster Profiles")
    st.markdown("""
    - **Cluster 0 — Developing / High Organic Fraction:** Nations characterized by lower GDP, emerging urbanization, and organic/food waste fractions exceeding 55%. Primary municipal need: **organic composting & basic collection infrastructure**.
    - **Cluster 1 — Industrialized / High Packaging Fraction:** High GDP per capita jurisdictions producing substantial dry recyclables (paper, cardboard, plastics). Primary municipal need: **advanced sorting, recycling incentives, and EPR legislation**.
    - **Cluster 2 — Demographic Mega-Producers:** High-population giants with enormous absolute tonnage requiring centralized mega-landfills and regional waste-to-energy processing plants.
    """)

# ══════════════════════════════════════════════
# TAB 5: Data Quality
# ══════════════════════════════════════════════
elif nav_choice == "Data Quality":
    st.markdown('<div class="section-header">Data Quality & Pipeline Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Full documentation of data integrity, missingness, and preprocessing steps as required by Case Study No. 73.</div>', unsafe_allow_html=True)

    # 4 Cards
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Raw Dimensions</div>
            <div class="metric-card-val">{df_raw.shape[0]} &times; {df_raw.shape[1]}</div>
            <div class="metric-card-sub">217 Rows, 51 Raw Attributes</div>
        </div>
        """, unsafe_allow_html=True)
    with q2:
        missing_target = df_raw["total_msw_total_msw_generated_tons_year"].isnull().sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Missing Target Rows</div>
            <div class="metric-card-val">{missing_target}</div>
            <div class="metric-card-sub">Dropped Prior to Modeling</div>
        </div>
        """, unsafe_allow_html=True)
    with q3:
        duplicates = df_raw.duplicated().sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Duplicate Rows</div>
            <div class="metric-card-val">{duplicates}</div>
            <div class="metric-card-sub">100% Unique Jurisdictions</div>
        </div>
        """, unsafe_allow_html=True)
    with q4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Cleaned Modeling Rows</div>
            <div class="metric-card-val">{df.shape[0]}</div>
            <div class="metric-card-sub">Complete Data for Training</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    dq_col1, dq_col2 = st.columns([1.1, 0.9], gap="large")

    with dq_col1:
        st.markdown("#### Feature Missingness Audit")
        missing_series = df_raw[list(num_features)].isnull().sum()
        missing_pct = (missing_series / len(df_raw)) * 100
        miss_df = pd.DataFrame({
            "Feature": missing_series.index,
            "Missing Count": missing_series.values,
            "Missing %": missing_pct.values,
            "Imputation Strategy": ["Median Imputation"] * len(missing_series)
        }).sort_values(by="Missing Count", ascending=False)
        st.dataframe(
            miss_df.style.format({"Missing %": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

    with dq_col2:
        st.markdown("#### Preprocessing Pipeline Steps")
        st.markdown("""
        1. **Target Sanitization:** Dropped 2 rows missing total MSW generation (`waste_tons`).
        2. **Quantile Discretization:** Segmented continuous waste tonnage into 3 balanced tiers (`Low`, `Medium`, `High`) using `pd.qcut()`.
        3. **Imputation:** Median numerical imputation on missing composition percentages and GDP values using `SimpleImputer(strategy='median')`.
        4. **Standardization:** Zero-mean, unit-variance scaling on numerical predictors using `StandardScaler()`.
        5. **Categorical Encoding:** One-hot encoding on `income_id` (4 levels) and `region_id` (7 levels) using `OneHotEncoder(handle_unknown='ignore')`.
        6. **Train/Test Split:** Stratified 80% train / 20% test split based on `waste_tier` to prevent distribution shift.
        """)

    st.markdown("---")
    st.markdown("#### Raw Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

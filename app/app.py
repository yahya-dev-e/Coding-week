import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CervAI · Risk Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #13161d;
    --surface2: #1c2030;
    --border: #2a2e3f;
    --accent: #e8c547;
    --accent2: #5bc8af;
    --danger: #f06a6a;
    --safe: #5bc8af;
    --text: #e8eaf0;
    --muted: #7a7f99;
    --radius: 12px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent);
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

/* Header */
.cerv-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.cerv-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: var(--accent);
    margin: 0;
    line-height: 1;
}
.cerv-header span {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 4px;
}

/* Section label */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
    padding-left: 2px;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1rem 0.85rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card .val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: var(--accent2);
    line-height: 1;
}
.metric-card .lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 6px;
}

/* Result box */
.result-box {
    border-radius: var(--radius);
    padding: 1.6rem 2rem;
    margin: 1.5rem 0;
    border: 1.5px solid;
}
.result-box.high {
    background: rgba(240,106,106,0.08);
    border-color: var(--danger);
}
.result-box.low {
    background: rgba(91,200,175,0.08);
    border-color: var(--safe);
}
.result-box h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    margin: 0 0 0.3rem;
}
.result-box.high h2 { color: var(--danger); }
.result-box.low h2 { color: var(--safe); }
.result-box p {
    font-size: 0.85rem;
    color: var(--muted);
    margin: 0;
    line-height: 1.6;
}

/* Probability bar */
.prob-bar-wrap {
    background: var(--surface2);
    border-radius: 100px;
    height: 8px;
    margin: 0.8rem 0 0.3rem;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s ease;
}
.prob-bar-fill.high { background: linear-gradient(90deg, #f06a6a88, #f06a6a); }
.prob-bar-fill.low  { background: linear-gradient(90deg, #5bc8af88, #5bc8af); }
.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
}

/* Streamlit widgets override */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,197,71,0.15) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Button */
.stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Info box */
.info-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.6;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Feature definitions ───────────────────────────────────────────────────────
FEATURES = {
    "Demographics": [
        ("Age", "Age", 13, 84, 25, 1, "Patient age in years"),
        ("Number of sexual partners", "Num_sexual_partners", 0, 28, 2, 1, "Lifetime number of sexual partners"),
        ("First sexual intercourse (age)", "First_sexual_intercourse", 10, 32, 17, 1, "Age at first sexual intercourse"),
        ("Num of pregnancies", "Num_of_pregnancies", 0, 11, 1, 1, "Number of pregnancies"),
    ],
    "Smoking": [
        ("Smokes", "Smokes", 0, 1, 0, 1, "Does the patient smoke? (0=No, 1=Yes)"),
        ("Smokes (years)", "Smokes_years", 0.0, 37.0, 0.0, 0.5, "Years of smoking"),
        ("Smokes (packs/year)", "Smokes_packs_year", 0.0, 40.0, 0.0, 0.5, "Packs smoked per year"),
    ],
    "Contraceptives": [
        ("Hormonal Contraceptives", "Hormonal_Contraceptives", 0, 1, 0, 1, "Uses hormonal contraceptives? (0=No, 1=Yes)"),
        ("Hormonal Contraceptives (years)", "Hormonal_Contraceptives_years", 0.0, 30.0, 0.0, 0.5, "Years of hormonal contraceptive use"),
        ("IUD", "IUD", 0, 1, 0, 1, "Has an intrauterine device? (0=No, 1=Yes)"),
        ("IUD (years)", "IUD_years", 0.0, 19.0, 0.0, 0.5, "Years of IUD use"),
    ],
    "STDs": [
        ("STDs", "STDs", 0, 1, 0, 1, "Has any STD? (0=No, 1=Yes)"),
        ("STDs (number)", "STDs_number", 0, 4, 0, 1, "Number of different STDs"),
        ("STDs: condylomatosis", "STDs_condylomatosis", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: cervical condylomatosis", "STDs_cervical_condylomatosis", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: vaginal condylomatosis", "STDs_vaginal_condylomatosis", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: vulvo-perineal condylomatosis", "STDs_vulvo_perineal_condylomatosis", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: syphilis", "STDs_syphilis", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: pelvic inflammatory disease", "STDs_pelvic_inflammatory_disease", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: genital herpes", "STDs_genital_herpes", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: molluscum contagiosum", "STDs_molluscum_contagiosum", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: AIDS", "STDs_AIDS", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: HIV", "STDs_HIV", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: Hepatitis B", "STDs_Hepatitis_B", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: HPV", "STDs_HPV", 0, 1, 0, 1, "(0=No, 1=Yes)"),
        ("STDs: Number of diagnosis", "STDs_Number_of_diagnosis", 0, 3, 0, 1, "Number of STD diagnoses"),
        ("STDs: Time since first diagnosis", "STDs_Time_since_first_diagnosis", 0.0, 29.0, 0.0, 0.5, "Years since first STD diagnosis"),
        ("STDs: Time since last diagnosis", "STDs_Time_since_last_diagnosis", 0.0, 29.0, 0.0, 0.5, "Years since last STD diagnosis"),
    ],
    "Diagnoses & Tests": [
        ("Dx: Cancer", "Dx_Cancer", 0, 1, 0, 1, "Previous cancer diagnosis (0=No, 1=Yes)"),
        ("Dx: CIN", "Dx_CIN", 0, 1, 0, 1, "Previous CIN diagnosis (0=No, 1=Yes)"),
        ("Dx: HPV", "Dx_HPV", 0, 1, 0, 1, "Previous HPV diagnosis (0=No, 1=Yes)"),
        ("Dx", "Dx", 0, 1, 0, 1, "General diagnosis flag (0=No, 1=Yes)"),
        ("Hinselmann", "Hinselmann", 0, 1, 0, 1, "Hinselmann test result (0=Neg, 1=Pos)"),
        ("Schiller", "Schiller", 0, 1, 0, 1, "Schiller test result (0=Neg, 1=Pos)"),
        ("Citology", "Citology", 0, 1, 0, 1, "Cytology result (0=Neg, 1=Pos)"),
    ],
}

# Column order must match what the model was trained on
COLUMN_ORDER = [
    "Age", "Number of sexual partners", "First sexual intercourse",
    "Num of pregnancies", "Smokes", "Smokes (years)", "Smokes (packs/year)",
    "Hormonal Contraceptives", "Hormonal Contraceptives (years)", "IUD", "IUD (years)",
    "STDs", "STDs (number)", "STDs:condylomatosis", "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes",
    "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV", "STDs:Hepatitis B",
    "STDs:HPV", "STDs: Number of diagnosis", "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis", "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx",
    "Hinselmann", "Schiller", "Citology",
]

# Map our internal keys to exact column names
KEY_TO_COL = {
    "Age": "Age",
    "Num_sexual_partners": "Number of sexual partners",
    "First_sexual_intercourse": "First sexual intercourse",
    "Num_of_pregnancies": "Num of pregnancies",
    "Smokes": "Smokes",
    "Smokes_years": "Smokes (years)",
    "Smokes_packs_year": "Smokes (packs/year)",
    "Hormonal_Contraceptives": "Hormonal Contraceptives",
    "Hormonal_Contraceptives_years": "Hormonal Contraceptives (years)",
    "IUD": "IUD",
    "IUD_years": "IUD (years)",
    "STDs": "STDs",
    "STDs_number": "STDs (number)",
    "STDs_condylomatosis": "STDs:condylomatosis",
    "STDs_cervical_condylomatosis": "STDs:cervical condylomatosis",
    "STDs_vaginal_condylomatosis": "STDs:vaginal condylomatosis",
    "STDs_vulvo_perineal_condylomatosis": "STDs:vulvo-perineal condylomatosis",
    "STDs_syphilis": "STDs:syphilis",
    "STDs_pelvic_inflammatory_disease": "STDs:pelvic inflammatory disease",
    "STDs_genital_herpes": "STDs:genital herpes",
    "STDs_molluscum_contagiosum": "STDs:molluscum contagiosum",
    "STDs_AIDS": "STDs:AIDS",
    "STDs_HIV": "STDs:HIV",
    "STDs_Hepatitis_B": "STDs:Hepatitis B",
    "STDs_HPV": "STDs:HPV",
    "STDs_Number_of_diagnosis": "STDs: Number of diagnosis",
    "STDs_Time_since_first_diagnosis": "STDs: Time since first diagnosis",
    "STDs_Time_since_last_diagnosis": "STDs: Time since last diagnosis",
    "Dx_Cancer": "Dx:Cancer",
    "Dx_CIN": "Dx:CIN",
    "Dx_HPV": "Dx:HPV",
    "Dx": "Dx",
    "Hinselmann": "Hinselmann",
    "Schiller": "Schiller",
    "Citology": "Citology",
}


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_paths = [
        "models/catboost_model.joblib",
        "catboost_model.joblib",
        "../models/catboost_model.joblib",
    ]
    asset_paths = [
        "models/catboost_assets.joblib",
        "catboost_assets.joblib",
        "../models/catboost_assets.joblib",
    ]
    for mp, ap in zip(model_paths, asset_paths):
        if Path(mp).exists() and Path(ap).exists():
            model = joblib.load(mp)
            assets = joblib.load(ap)  # dict: {imputer, scaler, columns}
            return model, assets, mp
    return None, None, None

model, assets, model_path = load_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cerv-header">
    <h1>CervAI</h1>
    <span>Biopsy Risk Classifier · CatBoost</span>
</div>
""", unsafe_allow_html=True)

# Model status
if model is None:
    st.markdown("""
    <div class="info-box">
    ⚠️ <strong>Model not found.</strong> Place <code>catboost_model.joblib</code> and
    <code>catboost_assets.joblib</code> inside a <code>models/</code> folder next to this script,
    then refresh. Train and save them using the training script.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-box">
    ✅ Model loaded from <code>{model_path}</code>. Fill in the patient details below and click <strong>Run Prediction</strong>.
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar: model metrics ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Metrics")
    st.markdown("""
    <div style='margin-bottom:1rem;'>
    <p style='font-size:0.75rem;color:#7a7f99;line-height:1.7;'>
    Reported on held-out test set (20% stratified split, SMOTE-balanced training).
    </p>
    </div>
    """, unsafe_allow_html=True)

    metrics = {
        "Accuracy": "—",
        "Precision": "—",
        "Recall": "—",
        "F1": "—",
        "ROC-AUC": "—",
    }

    for name, val in metrics.items():
        st.markdown(f"""
        <div style="background:#1c2030;border:1px solid #2a2e3f;border-radius:10px;
                    padding:0.7rem 1rem;margin-bottom:0.5rem;display:flex;
                    justify-content:space-between;align-items:center;">
          <span style="font-size:0.8rem;color:#7a7f99;font-family:'DM Sans',sans-serif;">{name}</span>
          <span style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#5bc8af;">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    <p style='font-size:0.75rem;color:#7a7f99;line-height:1.7;'>
    This tool uses a <strong style='color:#e8eaf0;'>CatBoost</strong> classifier trained on the
    UCI Cervical Cancer (Risk Factors) dataset to predict biopsy outcome.
    It is intended for <em>research and educational use only</em> and does not constitute
    medical advice.
    </p>
    """, unsafe_allow_html=True)


# ── Main: input tabs ──────────────────────────────────────────────────────────
tab_names = list(FEATURES.keys())
tabs = st.tabs(tab_names)

input_values = {}

for tab, (group_name, fields) in zip(tabs, FEATURES.items()):
    with tab:
        cols = st.columns(2)
        for i, (label, key, mn, mx, default, step, help_text) in enumerate(fields):
            col = cols[i % 2]
            with col:
                if isinstance(step, int):
                    input_values[key] = col.number_input(
                        label, min_value=int(mn), max_value=int(mx),
                        value=int(default), step=step, help=help_text
                    )
                else:
                    input_values[key] = col.number_input(
                        label, min_value=float(mn), max_value=float(mx),
                        value=float(default), step=step, help=help_text
                    )

st.markdown("<br>", unsafe_allow_html=True)


# ── Predict button ────────────────────────────────────────────────────────────
col_btn, col_reset = st.columns([1, 3])

with col_btn:
    predict_clicked = st.button("🔬 Run Prediction", disabled=(model is None))


# ── Prediction output ─────────────────────────────────────────────────────────
if predict_clicked and model is not None:
    # Build input row
    row = {KEY_TO_COL[k]: v for k, v in input_values.items()}
    input_df = pd.DataFrame([row])

    # Apply preprocessing from training pipeline (imputer → scaler → column alignment)
    try:
        trained_cols = assets["columns"]
        imputer = assets["imputer"]
        scaler = assets["scaler"]

        # Align columns to training order, fill any missing with 0
        input_df = input_df.reindex(columns=trained_cols, fill_value=0)
        input_df = pd.DataFrame(imputer.transform(input_df), columns=trained_cols)
        input_df = pd.DataFrame(scaler.transform(input_df), columns=trained_cols)
    except Exception as e:
        # Fallback: try raw column alignment if assets are unavailable
        try:
            input_df = input_df.reindex(columns=COLUMN_ORDER, fill_value=0)
        except Exception:
            if hasattr(model, "feature_names_"):
                input_df = input_df.reindex(columns=model.feature_names_, fill_value=0)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    pct = round(prob * 100, 1)
    risk_class = "high" if pred == 1 else "low"
    label = "High Risk — Biopsy Indicated" if pred == 1 else "Low Risk — Biopsy Unlikely"
    detail = (
        "The model predicts a positive biopsy result. "
        "This patient's risk profile warrants further clinical evaluation."
        if pred == 1 else
        "The model predicts a negative biopsy result. "
        "Continue routine screening per clinical guidelines."
    )

    st.markdown(f"""
    <div class="result-box {risk_class}">
        <h2>{'⚠️' if pred==1 else '✅'} &nbsp;{label}</h2>
        <p>{detail}</p>
        <div class="prob-bar-wrap" style="margin-top:1rem;">
            <div class="prob-bar-fill {risk_class}" style="width:{pct}%;"></div>
        </div>
        <div class="prob-label">
            <span>Biopsy probability</span>
            <span style="color:{'#f06a6a' if pred==1 else '#5bc8af'};">{pct}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature contribution summary
    st.markdown('<div class="section-label">Top contributing features</div>', unsafe_allow_html=True)

    if hasattr(model, "get_feature_importance"):
        try:
            importances = model.get_feature_importance()
            feat_names = assets.get("columns", COLUMN_ORDER)[:len(importances)]
            top_idx = np.argsort(importances)[::-1][:8]
            rows = [(feat_names[i], importances[i]) for i in top_idx]
            feat_df = pd.DataFrame(rows, columns=["Feature", "Importance"])
            feat_df["Importance"] = feat_df["Importance"].round(2)

            cols_f = st.columns(4)
            for i, (_, row_f) in enumerate(feat_df.iterrows()):
                with cols_f[i % 4]:
                    bar_w = int(row_f["Importance"] / feat_df["Importance"].max() * 100)
                    st.markdown(f"""
                    <div style="background:#1c2030;border:1px solid #2a2e3f;border-radius:10px;
                                padding:0.8rem;margin-bottom:0.5rem;">
                      <div style="font-size:0.7rem;color:#7a7f99;font-family:'DM Mono',monospace;
                                  margin-bottom:0.4rem;white-space:nowrap;overflow:hidden;
                                  text-overflow:ellipsis;">{row_f['Feature']}</div>
                      <div style="background:#2a2e3f;border-radius:100px;height:5px;">
                        <div style="width:{bar_w}%;height:100%;background:#e8c547;border-radius:100px;"></div>
                      </div>
                      <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;
                                  color:#e8c547;margin-top:0.3rem;">{row_f['Importance']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            pass

    st.markdown("""
    <div class="info-box" style="margin-top:1rem;">
    ⚕️ <strong>Clinical Disclaimer:</strong> This prediction is generated by a machine learning model
    and is not a substitute for professional medical judgment. Always consult a qualified healthcare
    provider before making clinical decisions.
    </div>
    """, unsafe_allow_html=True)

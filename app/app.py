import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CervAI · Risk Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #f5f4f0;
    --surface: #ffffff;
    --surface2: #f0ede8;
    --border: #ddd9d0;
    --accent: #1a4f6e;
    --accent-light: #e8f2f8;
    --accent2: #2d7a5e;
    --danger: #b83232;
    --danger-light: #fdf0f0;
    --safe: #2d7a5e;
    --safe-light: #edf7f3;
    --warning: #8a5a00;
    --warning-light: #fdf6e3;
    --text: #1a1816;
    --text2: #3d3a35;
    --muted: #7a7568;
    --radius: 10px;
    --shadow: 0 1px 4px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: var(--bg); }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent);
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 0.75rem;
}

.cerv-header {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.cerv-logo {
    width: 52px; height: 52px;
    background: var(--accent);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
}
.cerv-header-text h1 {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.65rem; color: var(--accent);
    margin: 0 0 0.15rem; line-height: 1.1;
}
.cerv-header-text p { font-size: 0.8rem; color: var(--muted); margin: 0; }
.cerv-badge {
    margin-left: auto;
    background: var(--accent-light); color: var(--accent);
    border: 1px solid #bed8ea;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 5px 12px; border-radius: 20px; white-space: nowrap;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.75rem; padding-left: 2px;
}

.info-box {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent2); border-radius: var(--radius);
    padding: 0.85rem 1.1rem; margin-bottom: 1rem;
    font-size: 0.82rem; color: var(--text2); line-height: 1.6;
    box-shadow: var(--shadow);
}
.info-box.warning { border-left-color: var(--warning); background: var(--warning-light); }
.info-box code {
    background: var(--surface2); padding: 1px 6px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78em; color: var(--accent);
}

.result-box {
    border-radius: var(--radius); padding: 1.4rem 1.8rem;
    margin: 1.25rem 0; border: 1.5px solid; box-shadow: var(--shadow);
}
.result-box.high { background: var(--danger-light); border-color: #e08080; }
.result-box.low  { background: var(--safe-light);   border-color: #80c9aa; }
.result-box-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
.result-box h2 { font-family: 'Libre Baskerville', serif; font-size: 1.35rem; margin: 0; line-height: 1.2; }
.result-box.high h2 { color: var(--danger); }
.result-box.low  h2 { color: var(--safe); }
.result-icon { font-size: 1.6rem; line-height: 1; }
.result-box p { font-size: 0.83rem; color: var(--text2); margin: 0; line-height: 1.65; }

.prob-section { margin-top: 1rem; }
.prob-bar-wrap {
    background: rgba(0,0,0,0.06); border-radius: 100px;
    height: 9px; margin: 0.5rem 0 0.3rem; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 100px; }
.prob-bar-fill.high { background: linear-gradient(90deg, #e07070, #b83232); }
.prob-bar-fill.low  { background: linear-gradient(90deg, #6bbf9a, #2d7a5e); }
.prob-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--muted);
    display: flex; justify-content: space-between; margin-top: 0.15rem;
}

[data-testid="stNumberInput"] input, .stSelectbox > div > div {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 7px !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.87rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(26,79,110,0.1) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important; color: var(--text2) !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 500 !important;
}

.stButton > button {
    background: #e8c547 !important; color: #1a1816 !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.65rem 2rem !important; font-size: 0.88rem !important;
    width: 100%; box-shadow: 0 2px 8px rgba(232,197,71,0.35) !important;
}
.stButton > button:hover { background: #d4b23a !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: var(--radius) var(--radius) 0 0;
    padding: 0 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.78rem !important; font-weight: 500 !important;
    color: var(--muted) !important; background: transparent !important;
    border: none !important; padding: 0.7rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-top: none !important; border-radius: 0 0 var(--radius) var(--radius);
    padding: 1.25rem !important; box-shadow: var(--shadow);
}

.metric-row {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.65rem 0.9rem; margin-bottom: 0.45rem;
    display: flex; justify-content: space-between; align-items: center;
}
.metric-name { font-size: 0.78rem; color: var(--text2); }
.metric-val  { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--accent2); font-weight: 500; }

.disclaimer {
    background: var(--warning-light); border: 1px solid #e0c060;
    border-left: 3px solid var(--warning); border-radius: var(--radius);
    padding: 0.85rem 1.1rem; margin-top: 1rem;
    font-size: 0.78rem; color: var(--text2); line-height: 1.6;
}
hr { border-color: var(--border) !important; }
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


# ── Load model ─────────────────────────────────────────────────────────────────
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
            assets = joblib.load(ap)
            return model, assets, mp
    return None, None, None

model, assets, model_path = load_model()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cerv-header">
    <div class="cerv-logo">&#x1F52C;</div>
    <div class="cerv-header-text">
        <h1>CervAI</h1>
        <p>Cervical Cancer Biopsy Risk Classifier &nbsp;&middot;&nbsp; CatBoost ML Model</p>
    </div>
    <div class="cerv-badge">Research Use Only</div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.markdown("""
    <div class="info-box warning">
    <strong>Model not found.</strong> Place <code>catboost_model.joblib</code> and
    <code>catboost_assets.joblib</code> inside a <code>models/</code> folder next to this script,
    then refresh.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-box">
    <strong>Model loaded</strong> from <code>{model_path}</code>.
    Complete the patient profile below and click <strong>Run Prediction</strong>.
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Performance")
    st.markdown("""
    <p style='font-size:0.73rem;color:#7a7568;line-height:1.65;margin-bottom:0.9rem;'>
    Held-out test set &middot; 20% stratified split &middot; SMOTE-balanced training
    </p>
    """, unsafe_allow_html=True)

    # Load real metrics from assets if available
    _metrics_display = {"Accuracy": "—", "Precision": "—", "Recall": "—", "F1-Score": "—", "ROC-AUC": "—"}
    if assets is not None:
        _key_map = {
            "Accuracy":  ["accuracy",  "Accuracy"],
            "Precision": ["precision", "Precision"],
            "Recall":    ["recall",    "Recall"],
            "F1-Score":  ["f1",        "f1_score", "F1", "F1-Score", "f1-score"],
            "ROC-AUC":   ["roc_auc",   "roc-auc",  "ROC-AUC", "auc", "AUC"],
        }
        for display_name, keys in _key_map.items():
            for k in keys:
                if k in assets:
                    v = assets[k]
                    _metrics_display[display_name] = f"{v:.3f}" if isinstance(v, float) else str(v)
                    break

    for name, val in _metrics_display.items():
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-name">{name}</span>
            <span class="metric-val">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    <p style='font-size:0.73rem;color:#7a7568;line-height:1.7;'>
    CervAI uses a <strong style='color:#1a1816;'>CatBoost</strong> gradient boosting classifier
    trained on the UCI Cervical Cancer (Risk Factors) dataset to predict biopsy outcome.<br><br>
    <em>Intended for research and educational purposes only. Not a substitute for clinical judgment.</em>
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("""
    <p style='font-size:0.73rem;color:#7a7568;line-height:1.7;'>
    UCI ML Repository &middot; Cervical Cancer (Risk Factors)<br>
    858 patients &middot; 36 clinical features &middot; Biopsy as target
    </p>
    """, unsafe_allow_html=True)


# ── Input tabs ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Patient Profile</div>', unsafe_allow_html=True)

tabs = st.tabs(list(FEATURES.keys()))
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

# ── Predict button ─────────────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button("Run Prediction", disabled=(model is None))


# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_clicked and model is not None:

    # Build and preprocess input
    row = {KEY_TO_COL[k]: v for k, v in input_values.items()}
    input_df = pd.DataFrame([row])

    try:
        trained_cols = assets["columns"]
        imputer      = assets["imputer"]
        scaler       = assets["scaler"]
        input_df = input_df.reindex(columns=trained_cols, fill_value=0)
        input_df = pd.DataFrame(imputer.transform(input_df), columns=trained_cols)
        input_df = pd.DataFrame(scaler.transform(input_df),  columns=trained_cols)
    except Exception:
        try:
            input_df = input_df.reindex(columns=COLUMN_ORDER, fill_value=0)
        except Exception:
            if hasattr(model, "feature_names_"):
                input_df = input_df.reindex(columns=model.feature_names_, fill_value=0)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    pct  = round(prob * 100, 1)
    risk_class = "high" if pred == 1 else "low"

    label  = "High Risk — Biopsy Indicated" if pred == 1 else "Low Risk — Biopsy Unlikely"
    detail = (
        "The model predicts a positive biopsy result. This patient's risk profile warrants "
        "further clinical evaluation and immediate specialist referral."
        if pred == 1 else
        "The model predicts a negative biopsy result. Continue routine cervical screening "
        "as per current clinical guidelines."
    )

    # Result card
    st.markdown(f"""
    <div class="result-box {risk_class}">
        <div class="result-box-header">
            <h2>{label}</h2>
        </div>
        <p>{detail}</p>
        <div class="prob-section">
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill {risk_class}" style="width:{pct}%;"></div>
            </div>
            <div class="prob-label">
                <span>Predicted biopsy probability</span>
                <span style="font-weight:600;color:{'#b83232' if pred==1 else '#2d7a5e'};">{pct}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Importance Chart (matplotlib) ─────────────────────────────────
    if hasattr(model, "get_feature_importance"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            importances = model.get_feature_importance()
            feat_names  = assets.get("columns", COLUMN_ORDER)[:len(importances)]

            top_n   = 10
            top_idx = np.argsort(importances)[::-1][:top_n]
            # Reverse so highest importance is at the top of the chart
            names   = [feat_names[i] for i in top_idx][::-1]
            vals    = [importances[i] for i in top_idx][::-1]

            # Teal-to-red gradient per bar (bottom = teal, top = red)
            bar_colors = []
            for i in range(top_n):
                t = i / max(top_n - 1, 1)
                bar_colors.append((
                    (45  + t * (184 - 45))  / 255,
                    (122 - t * (122 - 50))  / 255,
                    (94  - t * (94  - 50))  / 255,
                ))

            fig, ax = plt.subplots(figsize=(9, 4.8))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#fafaf8")

            y_pos = np.arange(top_n)
            bars  = ax.barh(y_pos, vals, color=bar_colors,
                            height=0.62, edgecolor="none", zorder=3)

            # Value labels at end of bars
            x_max = max(vals)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_width() + x_max * 0.013,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}",
                    va="center", ha="left",
                    fontsize=9, color="#7a7568", fontfamily="monospace"
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=10.5, color="#3d3a35")
            ax.set_xlabel("Feature Importance Score", fontsize=9,
                          color="#7a7568", labelpad=8, fontfamily="monospace")
            ax.tick_params(axis="x", colors="#bbb6aa", labelsize=9)
            ax.tick_params(axis="y", length=0)

            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#ddd9d0")
            ax.spines["bottom"].set_linewidth(0.8)

            ax.xaxis.grid(True, color="#ddd9d0", linewidth=0.7,
                          linestyle="--", zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(0, x_max * 1.18)

            ax.set_title("Top 10 Contributing Features",
                         fontsize=12, color="#1a1816", pad=12,
                         loc="left", fontweight="bold")

            plt.tight_layout(pad=1.4)

            st.markdown(
                '<div class="section-label" style="margin-top:1.5rem;">Feature importance analysis</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "Global feature importance from CatBoost · Higher score = stronger influence on prediction"
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        except Exception as e:
            st.warning(f"Feature importance chart unavailable: {e}")

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
    <strong>Clinical Disclaimer:</strong> This prediction is generated by a machine learning model
    and is <em>not a substitute</em> for professional medical judgment. All clinical decisions must
    be made by a qualified healthcare provider following established guidelines.
    </div>
    """, unsafe_allow_html=True)
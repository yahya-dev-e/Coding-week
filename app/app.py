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
    initial_sidebar_state="collapsed",
)

# ── Session state init ────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "input_values" not in st.session_state:
    st.session_state.input_values = {}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:           #0d1117;
    --surface:      #161b22;
    --surface2:     #1c2230;
    --surface3:     #21283a;
    --border:       #2a3248;
    --border2:      #334060;
    --accent:       #4f9cf9;
    --accent-glow:  rgba(79,156,249,0.18);
    --accent2:      #56d9a0;
    --danger:       #ff6b6b;
    --warning:      #f5a623;
    --text:         #e8eaf0;
    --text2:        #a8b4cc;
    --muted:        #5a6580;
    --radius:       14px;
    --radius-sm:    8px;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: var(--bg); }
.main .block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Shared nav ─────────────────────────────────────────────────── */
.cerv-nav {
    display: flex; align-items: center; gap: 1.5rem;
    padding: 0.9rem 1.8rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 2.5rem;
}
.cerv-nav-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem; color: #fff; letter-spacing: -0.02em; white-space: nowrap;
}
.cerv-nav-logo span { color: var(--accent); }
.cerv-nav-sep { width: 1px; height: 20px; background: var(--border2); flex-shrink: 0; }
.cerv-nav-crumb {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted);
}
.cerv-nav-crumb span { color: var(--text2); }
.cerv-nav-pill {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.12em; text-transform: uppercase;
    padding: 4px 12px; border-radius: 20px; white-space: nowrap;
    background: rgba(79,156,249,0.1); color: var(--accent);
    border: 1px solid rgba(79,156,249,0.2);
}

/* ── All buttons ────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; border-radius: 10px !important;
    font-size: 0.9rem !important; transition: all 0.2s !important;
    border: none !important; width: 100%;
}
/* Default = primary blue */
.stButton > button {
    background: linear-gradient(135deg, #4f9cf9 0%, #2563eb 100%) !important;
    color: #fff !important; padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 20px rgba(79,156,249,0.28) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(79,156,249,0.45) !important;
}
/* Back / ghost variant — wrap in .ghost-btn div */
.ghost-btn .stButton > button {
    background: var(--surface2) !important; color: var(--text2) !important;
    border: 1px solid var(--border2) !important;
    padding: 0.55rem 1.4rem !important; font-size: 0.82rem !important;
    box-shadow: none !important; width: auto !important;
}
.ghost-btn .stButton > button:hover {
    background: var(--surface3) !important; color: var(--text) !important;
    transform: none !important; box-shadow: none !important;
}

/* ── HOME ───────────────────────────────────────────────────────── */
.home-hero {
    position: relative;
    background: linear-gradient(145deg, #090e1a 0%, #0d1525 55%, #071220 100%);
    border: 1px solid var(--border2); border-radius: 24px;
    padding: 5.5rem 4rem; margin-bottom: 2rem; overflow: hidden; text-align: center;
}
.home-hero::before {
    content: ''; position: absolute; top: -100px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(79,156,249,0.09) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
}
.home-hero::after {
    content: ''; position: absolute; bottom: -80px; right: 8%;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(86,217,160,0.07) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
}
.home-hero-inner { position: relative; z-index: 1; }
.home-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.25em; text-transform: uppercase; color: var(--accent);
    display: inline-block; background: rgba(79,156,249,0.08);
    border: 1px solid rgba(79,156,249,0.2); padding: 5px 16px;
    border-radius: 20px; margin-bottom: 1.4rem;
}
.home-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 4rem; color: #fff; margin: 0 0 0.5rem;
    line-height: 1.05; letter-spacing: -0.03em;
}
.home-title span { color: var(--accent); font-style: italic; }
.home-subtitle {
    font-size: 1.05rem; color: var(--text2); max-width: 540px;
    margin: 0 auto 2.8rem; line-height: 1.75;
}
.home-stats-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem;
}
.stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.6rem 1.4rem; text-align: center;
}
.stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem; color: var(--accent); line-height: 1; margin-bottom: 0.4rem;
}
.stat-label { font-size: 0.78rem; color: var(--muted); }
.home-feats {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem;
}
.feat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.5rem;
    transition: border-color 0.2s, transform 0.2s;
}
.feat-card:hover { border-color: var(--border2); transform: translateY(-2px); }
.feat-icon { font-size: 1.6rem; margin-bottom: 0.75rem; }
.feat-title { font-size: 0.9rem; font-weight: 600; color: var(--text); margin-bottom: 0.4rem; }
.feat-desc  { font-size: 0.78rem; color: var(--muted); line-height: 1.65; }
.home-disclaimer {
    background: rgba(245,166,35,0.05); border: 1px solid rgba(245,166,35,0.18);
    border-left: 3px solid var(--warning); border-radius: var(--radius-sm);
    padding: 0.9rem 1.25rem; font-size: 0.78rem; color: var(--text2); line-height: 1.7;
    text-align: left;
}

/* ── CLASSIFIER ─────────────────────────────────────────────────── */
.page-header {
    display: flex; align-items: flex-start; gap: 1.5rem; margin-bottom: 1.75rem;
}
.page-header-icon {
    width: 52px; height: 52px; flex-shrink: 0;
    background: linear-gradient(135deg, #1a3a5c, #0e2240);
    border: 1px solid rgba(79,156,249,0.3); border-radius: 14px;
    display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
    box-shadow: 0 0 20px rgba(79,156,249,0.12);
}
.page-header-text h2 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.75rem; color: #fff; margin: 0 0 0.25rem; line-height: 1.2;
}
.page-header-text p { font-size: 0.84rem; color: var(--text2); margin: 0; line-height: 1.6; }
.col-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.6rem; padding-left: 2px;
}
.notice {
    display: flex; gap: 0.9rem; align-items: flex-start;
    background: var(--surface2); border: 1px solid var(--border);
    border-left: 3px solid var(--accent2); border-radius: var(--radius-sm);
    padding: 0.85rem 1.1rem; margin-bottom: 1.5rem;
    font-size: 0.8rem; color: var(--text2); line-height: 1.65;
}
.notice.error { border-left-color: var(--danger); }
.notice code {
    background: var(--surface3); padding: 2px 6px; border-radius: 4px;
    font-family: 'DM Mono', monospace; font-size: 0.78em; color: var(--accent);
}
[data-testid="stExpander"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; margin-bottom: 0.75rem !important;
}
[data-testid="stExpander"] summary {
    background: var(--surface2) !important; border-radius: var(--radius) !important;
    color: var(--text) !important; font-size: 0.86rem !important;
    font-weight: 500 !important; padding: 0.9rem 1.25rem !important;
}
details[open] summary { border-radius: var(--radius) var(--radius) 0 0 !important; }
[data-testid="stExpander"] > div:last-child {
    background: var(--surface) !important; border-top: 1px solid var(--border) !important;
    padding: 1.25rem !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.76rem !important; color: var(--text2) !important; font-weight: 500 !important;
}
[data-testid="stNumberInput"] input {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: var(--radius-sm) !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.88rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important; outline: none !important;
}
[data-testid="stNumberInput"] button {
    background: var(--surface3) !important; border: 1px solid var(--border) !important;
    color: var(--text2) !important;
}

/* ── RESULTS ────────────────────────────────────────────────────── */
.results-ttl {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.7rem; color: #fff; margin: 0 0 0.2rem;
}
.results-sub { font-size: 0.82rem; color: var(--muted); margin: 0 0 1.75rem; }
.result-card {
    border-radius: 20px; padding: 2.5rem; border: 1.5px solid;
    position: relative; overflow: hidden; margin-bottom: 1.5rem;
}
.result-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px; border-radius: 20px 20px 0 0;
}
.result-card.high {
    background: linear-gradient(135deg, rgba(255,107,107,0.08) 0%, rgba(13,17,23,0.5) 100%);
    border-color: rgba(255,107,107,0.3);
}
.result-card.high::before { background: linear-gradient(90deg, #ff6b6b, #ff4040); }
.result-card.low {
    background: linear-gradient(135deg, rgba(86,217,160,0.08) 0%, rgba(13,17,23,0.5) 100%);
    border-color: rgba(86,217,160,0.3);
}
.result-card.low::before { background: linear-gradient(90deg, #56d9a0, #22c55e); }
.result-inner { display: flex; align-items: flex-start; gap: 2rem; flex-wrap: wrap; }
.result-left  { flex: 1; min-width: 260px; }
.result-right { text-align: right; }
.result-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 4px 12px; border-radius: 20px;
    margin-bottom: 0.9rem; font-family: 'DM Mono', monospace;
}
.result-card.high .result-badge {
    background: rgba(255,107,107,0.15); color: var(--danger);
    border: 1px solid rgba(255,107,107,0.3);
}
.result-card.low  .result-badge {
    background: rgba(86,217,160,0.15); color: var(--accent2);
    border: 1px solid rgba(86,217,160,0.3);
}
.result-verdict {
    font-family: 'DM Serif Display', serif; font-size: 2rem;
    line-height: 1.15; margin: 0 0 0.6rem;
}
.result-card.high .result-verdict { color: var(--danger); }
.result-card.low  .result-verdict { color: var(--accent2); }
.result-detail { font-size: 0.86rem; color: var(--text2); line-height: 1.75; margin-bottom: 1.5rem; }
.prob-number {
    font-family: 'DM Serif Display', serif;
    font-size: 4.5rem; line-height: 1; font-style: italic;
}
.result-card.high .prob-number { color: rgba(255,107,107,0.75); }
.result-card.low  .prob-number { color: rgba(86,217,160,0.75); }
.prob-number-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--muted); margin-top: 0.3rem;
}
.prob-track {
    background: var(--surface3); border-radius: 100px;
    height: 10px; overflow: hidden; margin: 0.5rem 0 0.35rem; border: 1px solid var(--border);
}
.prob-fill { height: 100%; border-radius: 100px; }
.prob-fill.high {
    background: linear-gradient(90deg, #ff9a9a, #ff4040);
    box-shadow: 0 0 10px rgba(255,64,64,0.4);
}
.prob-fill.low {
    background: linear-gradient(90deg, #7ee8be, #22c55e);
    box-shadow: 0 0 10px rgba(34,197,94,0.4);
}
.prob-meta {
    display: flex; justify-content: space-between;
    font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--muted);
}
.summary-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 0.6rem; margin-bottom: 1.5rem;
}
.summary-item {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 0.7rem 1rem;
}
.summary-item-lbl { font-size: 0.68rem; color: var(--muted); margin-bottom: 0.2rem; }
.summary-item-val { font-family: 'DM Mono', monospace; font-size: 0.86rem; color: var(--text); font-weight: 500; }
.chart-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem;
}
.chart-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.3rem;
}
.chart-title { font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 0.2rem; }
.chart-sub   { font-size: 0.76rem; color: var(--muted); margin-bottom: 1.2rem; }
.disclaimer {
    background: rgba(245,166,35,0.05); border: 1px solid rgba(245,166,35,0.18);
    border-left: 3px solid var(--warning); border-radius: var(--radius-sm);
    padding: 0.9rem 1.25rem; font-size: 0.78rem; color: var(--text2); line-height: 1.7;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature definitions ───────────────────────────────────────────────────────
FEATURES = {
    "Demographics": {
        "icon": "👤", "desc": "Age, sexual history, pregnancies",
        "fields": [
            ("Age", "Age", 13, 84, 25, 1, "Patient age in years"),
            ("Sexual partners (lifetime)", "Num_sexual_partners", 0, 28, 2, 1, "Lifetime number"),
            ("Age at first intercourse", "First_sexual_intercourse", 10, 32, 17, 1, ""),
            ("Number of pregnancies", "Num_of_pregnancies", 0, 11, 1, 1, ""),
        ],
    },
    "Smoking": {
        "icon": "🚬", "desc": "Tobacco use history",
        "fields": [
            ("Smokes (0=No, 1=Yes)", "Smokes", 0, 1, 0, 1, ""),
            ("Smoking duration (years)", "Smokes_years", 0.0, 37.0, 0.0, 0.5, ""),
            ("Smoking intensity (packs/year)", "Smokes_packs_year", 0.0, 40.0, 0.0, 0.5, ""),
        ],
    },
    "Contraceptives": {
        "icon": "💊", "desc": "Hormonal & IUD use",
        "fields": [
            ("Hormonal contraceptives (0=No, 1=Yes)", "Hormonal_Contraceptives", 0, 1, 0, 1, ""),
            ("Hormonal contraceptive use (years)", "Hormonal_Contraceptives_years", 0.0, 30.0, 0.0, 0.5, ""),
            ("IUD (0=No, 1=Yes)", "IUD", 0, 1, 0, 1, ""),
            ("IUD use (years)", "IUD_years", 0.0, 19.0, 0.0, 0.5, ""),
        ],
    },
    "STDs": {
        "icon": "🦠", "desc": "Sexually transmitted disease history",
        "fields": [
            ("Has STD(s) (0=No, 1=Yes)", "STDs", 0, 1, 0, 1, ""),
            ("Number of STDs", "STDs_number", 0, 4, 0, 1, ""),
            ("Condylomatosis", "STDs_condylomatosis", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("Cervical condylomatosis", "STDs_cervical_condylomatosis", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("Vaginal condylomatosis", "STDs_vaginal_condylomatosis", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("Vulvo-perineal condylomatosis", "STDs_vulvo_perineal_condylomatosis", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("Syphilis", "STDs_syphilis", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("Pelvic inflammatory disease", "STDs_pelvic_inflammatory_disease", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("Genital herpes", "STDs_genital_herpes", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("Molluscum contagiosum", "STDs_molluscum_contagiosum", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("AIDS", "STDs_AIDS", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("HIV", "STDs_HIV", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("Hepatitis B", "STDs_Hepatitis_B", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("HPV", "STDs_HPV", 0, 1, 0, 1, "(0=No, 1=Yes)"),
            ("STD diagnoses (count)", "STDs_Number_of_diagnosis", 0, 3, 0, 1, ""),
            ("Years since first STD diagnosis", "STDs_Time_since_first_diagnosis", 0.0, 29.0, 0.0, 0.5, ""),
            ("Years since last STD diagnosis", "STDs_Time_since_last_diagnosis", 0.0, 29.0, 0.0, 0.5, ""),
        ],
    },
    "Prior Diagnoses & Tests": {
        "icon": "🔬", "desc": "Cancer, CIN, HPV & test results",
        "fields": [
            ("Previous cancer dx (0=No, 1=Yes)", "Dx_Cancer", 0, 1, 0, 1, ""),
            ("Previous CIN dx (0=No, 1=Yes)", "Dx_CIN", 0, 1, 0, 1, ""),
            ("Previous HPV dx (0=No, 1=Yes)", "Dx_HPV", 0, 1, 0, 1, ""),
            ("General diagnosis flag (0=No, 1=Yes)", "Dx", 0, 1, 0, 1, ""),
            ("Hinselmann test (0=Neg, 1=Pos)", "Hinselmann", 0, 1, 0, 1, ""),
            ("Schiller test (0=Neg, 1=Pos)", "Schiller", 0, 1, 0, 1, ""),
            ("Cytology (0=Neg, 1=Pos)", "Citology", 0, 1, 0, 1, ""),
        ],
    },
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
    "Age": "Age", "Num_sexual_partners": "Number of sexual partners",
    "First_sexual_intercourse": "First sexual intercourse",
    "Num_of_pregnancies": "Num of pregnancies", "Smokes": "Smokes",
    "Smokes_years": "Smokes (years)", "Smokes_packs_year": "Smokes (packs/year)",
    "Hormonal_Contraceptives": "Hormonal Contraceptives",
    "Hormonal_Contraceptives_years": "Hormonal Contraceptives (years)",
    "IUD": "IUD", "IUD_years": "IUD (years)", "STDs": "STDs",
    "STDs_number": "STDs (number)", "STDs_condylomatosis": "STDs:condylomatosis",
    "STDs_cervical_condylomatosis": "STDs:cervical condylomatosis",
    "STDs_vaginal_condylomatosis": "STDs:vaginal condylomatosis",
    "STDs_vulvo_perineal_condylomatosis": "STDs:vulvo-perineal condylomatosis",
    "STDs_syphilis": "STDs:syphilis",
    "STDs_pelvic_inflammatory_disease": "STDs:pelvic inflammatory disease",
    "STDs_genital_herpes": "STDs:genital herpes",
    "STDs_molluscum_contagiosum": "STDs:molluscum contagiosum",
    "STDs_AIDS": "STDs:AIDS", "STDs_HIV": "STDs:HIV",
    "STDs_Hepatitis_B": "STDs:Hepatitis B", "STDs_HPV": "STDs:HPV",
    "STDs_Number_of_diagnosis": "STDs: Number of diagnosis",
    "STDs_Time_since_first_diagnosis": "STDs: Time since first diagnosis",
    "STDs_Time_since_last_diagnosis": "STDs: Time since last diagnosis",
    "Dx_Cancer": "Dx:Cancer", "Dx_CIN": "Dx:CIN", "Dx_HPV": "Dx:HPV",
    "Dx": "Dx", "Hinselmann": "Hinselmann", "Schiller": "Schiller", "Citology": "Citology",
}

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_paths = ["models/xgboost_model.joblib", "xgboost_model.joblib", "../models/xgboost_model.joblib", "models/catboost_model.joblib", "catboost_model.joblib"]
    asset_paths = ["models/xgboost_assets.joblib", "xgboost_assets.joblib", "../models/xgboost_assets.joblib", "models/catboost_assets.joblib", "catboost_assets.joblib"]
    for mp, ap in zip(model_paths, asset_paths):
        if Path(mp).exists() and Path(ap).exists():
            return joblib.load(mp), joblib.load(ap), mp
    return None, None, None

model, assets, model_path = load_model()

# ── Nav helper ─────────────────────────────────────────────────────────────────
PAGE_LABELS = {"home": "Home", "classifier": "Patient Classifier", "results": "Prediction Results"}

def render_nav(current):
    crumb = PAGE_LABELS.get(current, current)
    st.markdown(f"""
    <div class="cerv-nav">
        <div class="cerv-nav-logo">Cerv<span>AI</span></div>
        <div class="cerv-nav-sep"></div>
        <div class="cerv-nav-crumb">
            <span style="color:var(--muted);">CervAI</span> &nbsp;/&nbsp; <span>{crumb}</span>
        </div>
        <div class="cerv-nav-pill">Research Use Only</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE · HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    render_nav("home")

    st.markdown("""
    <div class="home-hero">
      <div class="home-hero-inner">
        <div class="home-eyebrow">Cervical Cancer Risk Assessment Tool</div>
        <div class="home-title">Cerv<span>AI</span></div>
        <div class="home-subtitle">
          An AI-powered biopsy risk classifier trained on clinical data from 858 patients.
          Enter patient risk factors and receive an evidence-informed prediction in seconds.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.6, 1, 1.6])
    with c2:
        if st.button("Begin Patient Assessment →", key="home_cta"):
            st.session_state.page = "classifier"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats row
    st.markdown("""
    <div class="home-stats-row">
      <div class="stat-card"><div class="stat-val">858</div><div class="stat-label">Clinical patient records</div></div>
      <div class="stat-card"><div class="stat-val">36</div><div class="stat-label">Risk factor features</div></div>
      <div class="stat-card"><div class="stat-val">XGBoost</div><div class="stat-label">Gradient boosting classifier</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="home-feats">
      <div class="feat-card">
        <div class="feat-icon">📋</div>
        <div class="feat-title">Comprehensive Profiling</div>
        <div class="feat-desc">Covers demographics, smoking, contraceptives, STDs, and prior diagnoses across 5 clinical categories with 36 features.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">⚡</div>
        <div class="feat-title">Instant Prediction</div>
        <div class="feat-desc">XGBoost model trained with SMOTE oversampling on a stratified 80/20 split returns results in milliseconds.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">📊</div>
        <div class="feat-title">Explainable AI</div>
        <div class="feat-desc">Visualize the top contributing features driving each prediction — transparent, interpretable results for every case.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model metrics (if loaded)
    if assets is not None:
        _key_map = {
            "Accuracy":  ["accuracy", "Accuracy"],
            "Precision": ["precision", "Precision"],
            "Recall":    ["recall", "Recall"],
            "F1-Score":  ["f1", "f1_score", "F1"],
            "ROC-AUC":   ["roc_auc", "roc-auc", "auc", "AUC"],
        }
        metrics = {}
        for dn, keys in _key_map.items():
            for k in keys:
                if k in assets:
                    v = assets[k]
                    metrics[dn] = f"{v:.3f}" if isinstance(v, float) else str(v)
                    break
        if metrics:
            st.markdown('<p class="col-label">Model Performance · Held-out Test Set</p>', unsafe_allow_html=True)
            m_cols = st.columns(len(metrics))
            for col, (name, val) in zip(m_cols, metrics.items()):
                col.markdown(f"""
                <div class="stat-card" style="padding:1.1rem 1rem;">
                  <div class="stat-val" style="font-size:1.65rem;">{val}</div>
                  <div class="stat-label">{name}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="home-disclaimer">
    <strong>⚕ Research &amp; Educational Use Only.</strong> CervAI is a machine learning prototype
    trained on the UCI Cervical Cancer (Risk Factors) dataset. It is <em>not a medical device</em>
    and must not be used to guide clinical decisions. All results require interpretation by a
    qualified healthcare professional following established clinical guidelines.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE · CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
def page_classifier():
    render_nav("classifier")

    # Back button row
    back_col, _ = st.columns([1, 8])
    with back_col:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("← Home", key="clf_back"):
            st.session_state.page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="page-header">
      <div class="page-header-icon">📋</div>
      <div class="page-header-text">
        <h2>Patient Risk Profile</h2>
        <p>Complete all relevant sections, then click <strong style="color:#e8eaf0;">Run Biopsy Risk Prediction</strong> at the bottom.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.markdown("""<div class="notice error"><span>⚠️</span>
        <div><strong>Model not found.</strong> Place <code>xgboost_model.joblib</code> and
        <code>xgboost_assets.joblib</code> in a <code>models/</code> folder, then refresh.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="notice"><span>✅</span>
        <div>Model ready &nbsp;·&nbsp; <code>{model_path}</code></div>
        </div>""", unsafe_allow_html=True)

    input_values = {}
    col1, col2 = st.columns([1, 1], gap="medium")

    def render_section(name, data, container):
        fields = data["fields"]
        with container:
            with st.expander(f"{data['icon']}  {name}  ·  {data['desc']}",
                             expanded=(name in ["Demographics", "Smoking"])):
                n = 2 if len(fields) > 2 else len(fields)
                cols = st.columns(n)
                for i, (label, key, mn, mx, default, step, help_text) in enumerate(fields):
                    c = cols[i % n]
                    saved = st.session_state.input_values.get(key, default)
                    with c:
                        if isinstance(step, int):
                            input_values[key] = c.number_input(
                                label, min_value=int(mn), max_value=int(mx),
                                value=int(saved), step=step,
                                help=help_text if help_text else None, key=f"f_{key}"
                            )
                        else:
                            input_values[key] = c.number_input(
                                label, min_value=float(mn), max_value=float(mx),
                                value=float(saved), step=step,
                                help=help_text if help_text else None, key=f"f_{key}"
                            )

    with col1:
        st.markdown('<p class="col-label">Patient Characteristics</p>', unsafe_allow_html=True)
        for name in ["Demographics", "Contraceptives", "Prior Diagnoses & Tests"]:
            if name in FEATURES:
                render_section(name, FEATURES[name], col1)

    with col2:
        st.markdown('<p class="col-label">Risk Factors &amp; History</p>', unsafe_allow_html=True)
        for name in ["Smoking", "STDs"]:
            if name in FEATURES:
                render_section(name, FEATURES[name], col2)

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        if st.button("🔬  Run Biopsy Risk Prediction", disabled=(model is None), key="predict_btn"):
            st.session_state.input_values = dict(input_values)

            row = {KEY_TO_COL[k]: v for k, v in input_values.items()}
            input_df = pd.DataFrame([row])
            try:
                trained_cols = assets["columns"]
                imputer = assets["imputer"]
                scaler  = assets["scaler"]
                input_df = input_df.reindex(columns=trained_cols, fill_value=0)
                input_df = pd.DataFrame(imputer.transform(input_df), columns=trained_cols)
                input_df = pd.DataFrame(scaler.transform(input_df),  columns=trained_cols)
            except Exception:
                try:
                    input_df = input_df.reindex(columns=COLUMN_ORDER, fill_value=0)
                except Exception:
                    if hasattr(model, "feature_names_"):
                        input_df = input_df.reindex(columns=model.feature_names_, fill_value=0)

            pred = int(model.predict(input_df)[0])
            prob = float(model.predict_proba(input_df)[0][1])

            fi = None
            # XGBoost sklearn API uses feature_importances_
            _has_fi = hasattr(model, "feature_importances_") or hasattr(model, "get_feature_importance")
            if _has_fi:
                try:
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                    else:
                        importances = model.get_feature_importance()
                    feat_names = (assets.get("columns", COLUMN_ORDER) if assets else COLUMN_ORDER)[:len(importances)]
                    top_n   = 12
                    top_idx = np.argsort(importances)[::-1][:top_n]
                    fi = {
                        "names": [feat_names[i] for i in top_idx][::-1],
                        "vals":  [float(importances[i]) for i in top_idx][::-1],
                    }
                except Exception:
                    pass

            st.session_state.prediction = {
                "pred": pred, "prob": prob, "pct": round(prob * 100, 1),
                "feature_importance": fi,
                "input_summary": {KEY_TO_COL[k]: v for k, v in input_values.items()
                                  if v not in (0, 0.0) and k in KEY_TO_COL},
            }
            st.session_state.page = "results"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE · RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def page_results():
    render_nav("results")

    p = st.session_state.prediction
    if p is None:
        st.warning("No prediction available. Please complete the patient profile first.")
        if st.button("← Go to Classifier"):
            st.session_state.page = "classifier"
            st.rerun()
        return

    pred       = p["pred"]
    pct        = p["pct"]
    risk_class = "high" if pred == 1 else "low"
    verdict    = "Biopsy Indicated" if pred == 1 else "Biopsy Unlikely"
    badge_txt  = "⚠  High Risk" if pred == 1 else "✓  Low Risk"
    detail     = (
        "The model predicts a <strong>positive biopsy result</strong>. This patient's risk profile "
        "warrants further clinical evaluation and immediate specialist referral."
        if pred == 1 else
        "The model predicts a <strong>negative biopsy result</strong>. Continue routine cervical "
        "screening as per current clinical guidelines."
    )

    # Top bar: title + back button
    th, tb = st.columns([5, 1])
    with th:
        st.markdown(f"""
        <div class="results-ttl">Prediction Results</div>
        <p class="results-sub">XGBoost · Patient profile assessment</p>
        """, unsafe_allow_html=True)
    with tb:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("← Edit Profile", key="res_back_top"):
            st.session_state.page = "classifier"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Main result card
    st.markdown(f"""
    <div class="result-card {risk_class}">
      <div class="result-inner">
        <div class="result-left">
          <div class="result-badge">{badge_txt}</div>
          <div class="result-verdict">{verdict}</div>
          <div class="result-detail">{detail}</div>
          <div>
            <div class="prob-track">
              <div class="prob-fill {risk_class}" style="width:{pct}%;"></div>
            </div>
            <div class="prob-meta">
              <span>Predicted biopsy probability</span>
              <span style="font-weight:600;">{pct}%</span>
            </div>
          </div>
        </div>
        <div class="result-right">
          <div class="prob-number">{pct}%</div>
          <div class="prob-number-lbl">Risk Score</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Notable inputs
    summary = {k: v for k, v in p.get("input_summary", {}).items() if v not in (0, 0.0)}
    if summary:
        st.markdown('<p class="col-label" style="margin-top:0.5rem;">Notable Input Values</p>', unsafe_allow_html=True)
        grid_html = '<div class="summary-grid">'
        for k, v in list(summary.items())[:12]:
            grid_html += f'<div class="summary-item"><div class="summary-item-lbl">{k}</div><div class="summary-item-val">{v}</div></div>'
        grid_html += "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)

    # Feature importance
    fi = p.get("feature_importance")
    if fi:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            names = fi["names"]
            vals  = fi["vals"]
            top_n = len(names)

            fig, ax = plt.subplots(figsize=(9, 5.2))
            fig.patch.set_facecolor("#161b22")
            ax.set_facecolor("#161b22")

            y_pos = np.arange(top_n)
            bar_colors = []
            for i in range(top_n):
                t = i / max(top_n - 1, 1)
                r = int(79  + t * (249 - 79))
                g = int(156 - t * (156 - 107))
                b_ch = int(249 - t * (249 - 107))
                bar_colors.append((r/255, g/255, b_ch/255))

            bars = ax.barh(y_pos, vals, color=bar_colors, height=0.58, edgecolor="none", zorder=3)
            x_max = max(vals)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_width() + x_max * 0.015,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}", va="center", ha="left",
                        fontsize=9, color="#5a6580", fontfamily="monospace")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=10, color="#a8b4cc")
            ax.set_xlabel("Feature Importance Score", fontsize=9,
                          color="#5a6580", labelpad=8, fontfamily="monospace")
            ax.tick_params(axis="x", colors="#2a3248", labelsize=9, labelcolor="#5a6580")
            ax.tick_params(axis="y", length=0)
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#2a3248")
            ax.spines["bottom"].set_linewidth(0.8)
            ax.xaxis.grid(True, color="#1c2230", linewidth=0.8, linestyle="-", zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(0, x_max * 1.2)
            ax.set_title("Top Contributing Features", fontsize=12,
                         color="#e8eaf0", pad=14, loc="left",
                         fontweight="bold", fontfamily="sans-serif")
            plt.tight_layout(pad=1.2)

            st.markdown("""
            <div class="chart-wrap">
              <div class="chart-eyebrow">Explainability · Global Feature Importance</div>
              <div class="chart-title">Top 12 Contributing Features</div>
              <div class="chart-sub">Higher score = stronger influence on prediction · XGBoost built-in importance</div>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Feature importance chart unavailable: {e}")

    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("← Edit Profile", key="res_back_bottom"):
            st.session_state.page = "classifier"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with ac2:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("🏠  Home", key="res_home"):
            st.session_state.page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with ac3:
        if st.button("🔄  New Assessment", key="res_new"):
            st.session_state.input_values = {}
            st.session_state.prediction = None
            st.session_state.page = "classifier"
            st.rerun()

    st.markdown("""
    <div class="disclaimer">
    <strong>⚕ Clinical Disclaimer:</strong> This prediction is generated by a machine learning model
    and is <em>not a substitute</em> for professional medical judgment. All clinical decisions must
    be made by a qualified healthcare provider following established guidelines.
    This tool is intended for <em>research and educational purposes only</em>.
    </div>
    """, unsafe_allow_html=True)


# ── Router ─────────────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "classifier":
    page_classifier()
elif st.session_state.page == "results":
    page_results()
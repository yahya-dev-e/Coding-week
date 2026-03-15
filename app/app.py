import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import datetime

st.set_page_config(
    page_title="CervAI · Clinical Risk Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────
for k, v in {
    "page": "login",
    "logged_in": False,
    "prediction": None,
    "input_values": {},
    "appointments": [],
    "history": [],
    "notifications": [
        {"id": 1, "text": "New screening guidelines updated", "time": "2h ago", "read": False, "icon": "📋"},
        {"id": 2, "text": "Model v2.1 deployed successfully",  "time": "5h ago", "read": False, "icon": "🚀"},
        {"id": 3, "text": "3 appointments scheduled today",    "time": "1d ago", "read": True,  "icon": "📅"},
    ],
    "show_notif": False,
    "show_profile_menu": False,
    "contact_sent": False,
    "settings": {"theme": "dark", "alerts": True, "autosave": True},
    "active_faq": None,
    "show_result_overlay": False,
    "last_prediction_id": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

USERS = {
    "bakraouladomor@gmail.com":  {"password": "cerv2024", "name": "Bakr Aoulad Omar",  "role": "Lead Researcher",  "initials": "BO", "dept": "Clinical Research"},
    "yassirjbili@gmail.com":     {"password": "cerv2024", "name": "Yassir Jbili",       "role": "ML Engineer",      "initials": "YJ", "dept": "AI & Engineering"},
    "ilyaselhadad@gmail.com":    {"password": "cerv2024", "name": "Ilyas El Hadad",     "role": "Data Scientist",   "initials": "IE", "dept": "Data Science"},
    "mohammedelhadad@gmail.com": {"password": "cerv2024", "name": "Mohamed El Hadad",   "role": "Biostatistics",    "initials": "ME", "dept": "Statistics"},
    "yahyaelomari@gmail.com":    {"password": "cerv2024", "name": "Yahya El Omari",     "role": "Clinical Review",  "initials": "YO", "dept": "Clinical QA"},
    "admin@cerv.ai":             {"password": "admin123", "name": "Admin User",         "role": "Administrator",    "initials": "AU", "dept": "IT & Systems"},
}

def check_login(email, password):
    for k, v in USERS.items():
        if k.lower() == email.strip().lower() and v["password"] == password.strip():
            return v
    return None

@st.cache_resource
def load_model():
    mp_list = ["models/xgboost_model.joblib","xgboost_model.joblib","../models/xgboost_model.joblib",
               "models/catboost_model.joblib","catboost_model.joblib","../models/catboost_model.joblib"]
    ap_list = ["models/xgboost_assets.joblib","xgboost_assets.joblib","../models/xgboost_assets.joblib",
               "models/catboost_assets.joblib","catboost_assets.joblib","../models/catboost_assets.joblib"]
    for mp, ap in zip(mp_list, ap_list):
        if Path(mp).exists() and Path(ap).exists():
            return joblib.load(mp), joblib.load(ap), mp
    return None, None, None

model, assets, model_path = load_model()

COLUMN_ORDER = [
    "Age","Number of sexual partners","First sexual intercourse","Num of pregnancies",
    "Smokes","Smokes (years)","Smokes (packs/year)","Hormonal Contraceptives",
    "Hormonal Contraceptives (years)","IUD","IUD (years)","STDs","STDs (number)",
    "STDs:condylomatosis","STDs:cervical condylomatosis","STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis","STDs:syphilis","STDs:pelvic inflammatory disease",
    "STDs:genital herpes","STDs:molluscum contagiosum","STDs:AIDS","STDs:HIV",
    "STDs:Hepatitis B","STDs:HPV","STDs: Number of diagnosis",
    "STDs: Time since first diagnosis","STDs: Time since last diagnosis",
    "Dx:Cancer","Dx:CIN","Dx:HPV","Dx","Hinselmann","Schiller","Citology",
]
KEY_TO_COL = {
    "Age":"Age","Num_sexual_partners":"Number of sexual partners",
    "First_sexual_intercourse":"First sexual intercourse","Num_of_pregnancies":"Num of pregnancies",
    "Smokes":"Smokes","Smokes_years":"Smokes (years)","Smokes_packs_year":"Smokes (packs/year)",
    "Hormonal_Contraceptives":"Hormonal Contraceptives",
    "Hormonal_Contraceptives_years":"Hormonal Contraceptives (years)",
    "IUD":"IUD","IUD_years":"IUD (years)","STDs":"STDs","STDs_number":"STDs (number)",
    "STDs_condylomatosis":"STDs:condylomatosis",
    "STDs_cervical_condylomatosis":"STDs:cervical condylomatosis",
    "STDs_vaginal_condylomatosis":"STDs:vaginal condylomatosis",
    "STDs_vulvo_perineal_condylomatosis":"STDs:vulvo-perineal condylomatosis",
    "STDs_syphilis":"STDs:syphilis",
    "STDs_pelvic_inflammatory_disease":"STDs:pelvic inflammatory disease",
    "STDs_genital_herpes":"STDs:genital herpes",
    "STDs_molluscum_contagiosum":"STDs:molluscum contagiosum",
    "STDs_AIDS":"STDs:AIDS","STDs_HIV":"STDs:HIV","STDs_Hepatitis_B":"STDs:Hepatitis B",
    "STDs_HPV":"STDs:HPV","STDs_Number_of_diagnosis":"STDs: Number of diagnosis",
    "STDs_Time_since_first_diagnosis":"STDs: Time since first diagnosis",
    "STDs_Time_since_last_diagnosis":"STDs: Time since last diagnosis",
    "Dx_Cancer":"Dx:Cancer","Dx_CIN":"Dx:CIN","Dx_HPV":"Dx:HPV","Dx":"Dx",
    "Hinselmann":"Hinselmann","Schiller":"Schiller","Citology":"Citology",
}

# ═══════════════════════════════════════════════════════════════════
# CSS  (from document-7 base, kept intact)
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:#04080f; --bg2:#070d1a; --surf:#0b1220; --surf2:#0f1828;
    --surf3:#141f32; --surf4:#1a2740; --bord:#1c2d46; --bord2:#253a58; --bord3:#2f4870;
    --blue:#3b9eff; --blue2:#1a7de8; --blue-g:rgba(59,158,255,0.18); --blue-s:rgba(59,158,255,0.08);
    --teal:#00d9a0; --purple:#7b6fff; --rose:#ff5f7e; --amber:#ffb74d;
    --txt:#d8e4f5; --txt2:#7e97be; --txt3:#3d5270;
    --r:14px; --r2:10px; --r3:8px;
    --fd:'Syne',sans-serif; --fb:'Outfit',sans-serif; --fm:'JetBrains Mono',monospace;
    --shadow:0 8px 32px rgba(0,0,0,0.45); --shadow2:0 2px 12px rgba(0,0,0,0.3);
}

html,body,[class*="css"]{font-family:var(--fb)!important;background:var(--bg)!important;color:var(--txt)!important;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:var(--bg);}
.main .block-container{padding:1.5rem 2rem 4rem;max-width:1380px;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--bord2);border-radius:3px;}
.stApp::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
    background:radial-gradient(ellipse 1000px 800px at 10% 15%,rgba(59,158,255,0.04) 0%,transparent 60%),
               radial-gradient(ellipse 700px 600px at 90% 85%,rgba(123,111,255,0.04) 0%,transparent 60%),
               radial-gradient(ellipse 600px 500px at 55% 50%,rgba(0,217,160,0.025) 0%,transparent 60%);}
.stApp::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
    background-image:radial-gradient(circle 1px at 1px 1px,rgba(59,158,255,0.055) 1px,transparent 0);
    background-size:52px 52px;}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{background:var(--surf)!important;border-right:1px solid var(--bord)!important;min-width:260px!important;max-width:260px!important;}
[data-testid="stSidebar"]>div:first-child{padding:1.25rem 1rem 2rem;}

/* ── BUTTONS ── */
.stButton>button{font-family:var(--fb)!important;font-weight:600!important;border-radius:var(--r2)!important;
    font-size:0.875rem!important;transition:all 0.2s!important;border:none!important;width:100%;
    background:linear-gradient(135deg,var(--blue),var(--blue2))!important;color:#fff!important;
    padding:0.7rem 1.5rem!important;box-shadow:0 4px 18px rgba(59,158,255,0.28)!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(59,158,255,0.45)!important;}
.stButton>button:active{transform:translateY(0)!important;}
.ghost .stButton>button{background:var(--surf2)!important;color:var(--txt2)!important;
    border:1px solid var(--bord)!important;box-shadow:none!important;
    padding:0.5rem 1rem!important;font-size:0.78rem!important;width:auto!important;}
.ghost .stButton>button:hover{background:var(--surf3)!important;color:var(--txt)!important;transform:none!important;box-shadow:none!important;}
.nav-btn .stButton>button{background:transparent!important;color:var(--txt2)!important;
    border:none!important;box-shadow:none!important;text-align:left!important;
    padding:0.6rem 0.9rem!important;font-size:0.84rem!important;border-radius:var(--r2)!important;width:100%!important;}
.nav-btn .stButton>button:hover{background:var(--surf2)!important;color:var(--txt)!important;transform:none!important;box-shadow:none!important;}
.nav-btn-active .stButton>button{background:var(--blue-g)!important;color:var(--blue)!important;box-shadow:none!important;}
.nav-btn-active .stButton>button:hover{background:var(--blue-g)!important;transform:none!important;}

/* ── INPUTS ── */
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{
    background:var(--surf2)!important;border:1px solid var(--bord)!important;
    color:var(--txt)!important;border-radius:var(--r2)!important;font-family:var(--fb)!important;transition:all 0.2s!important;}
[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{
    border-color:var(--blue)!important;box-shadow:0 0 0 3px var(--blue-g)!important;background:var(--surf3)!important;outline:none!important;}
[data-testid="stSelectbox"] *{background:var(--surf2)!important;color:var(--txt)!important;border-color:var(--bord)!important;}
label[data-testid="stWidgetLabel"] p{font-size:0.75rem!important;color:var(--txt2)!important;font-weight:500!important;}

/* ── NUMBER INPUTS (classifier) ── */
[data-testid="stNumberInput"]{margin-bottom:0!important;}
[data-testid="stNumberInput"] input{
    background:var(--surf3)!important;border:1.5px solid var(--bord)!important;
    color:var(--txt)!important;border-radius:var(--r2)!important;
    font-family:var(--fm)!important;font-size:0.9rem!important;font-weight:500!important;
    padding:0.55rem 0.75rem!important;transition:all 0.2s!important;}
[data-testid="stNumberInput"] input:focus{
    border-color:var(--blue)!important;box-shadow:0 0 0 3px var(--blue-s)!important;
    background:var(--surf4)!important;outline:none!important;}
[data-testid="stNumberInput"] button{
    background:var(--surf4)!important;border-color:var(--bord)!important;
    color:var(--txt2)!important;border-radius:var(--r3)!important;}

/* ── YES/NO TOGGLE BUTTONS ── */
div[data-testid="stRadio"]>div{
    display:flex!important;flex-direction:row!important;gap:0!important;
    background:var(--surf3)!important;border:1px solid var(--bord)!important;
    border-radius:10px!important;padding:3px!important;width:fit-content!important;}
div[data-testid="stRadio"]>div>label{
    display:flex!important;align-items:center!important;justify-content:center!important;
    padding:0.45rem 1.4rem!important;border-radius:7px!important;cursor:pointer!important;
    font-family:var(--fm)!important;font-size:0.72rem!important;font-weight:600!important;
    letter-spacing:0.08em!important;transition:all 0.18s!important;
    color:var(--txt3)!important;background:transparent!important;min-width:68px!important;}
div[data-testid="stRadio"]>div>label:has(input:checked){
    background:var(--surf)!important;box-shadow:0 1px 4px rgba(0,0,0,0.35)!important;}
div[data-testid="stRadio"]>div>label:first-of-type:has(input:checked){color:var(--teal)!important;border:1px solid rgba(0,217,160,0.3)!important;}
div[data-testid="stRadio"]>div>label:last-of-type:has(input:checked){color:var(--rose)!important;border:1px solid rgba(255,95,126,0.3)!important;}
div[data-testid="stRadio"] input[type="radio"]{display:none!important;}
div[data-testid="stRadio"]>label{display:none!important;}

/* ── EXPANDER ── */
[data-testid="stExpander"]{background:transparent!important;border:none!important;margin-bottom:0!important;}
[data-testid="stExpander"] summary{background:var(--surf3)!important;border:1px solid var(--bord)!important;
    border-radius:var(--r2)!important;color:var(--txt2)!important;font-size:0.8rem!important;
    font-weight:500!important;padding:0.7rem 1rem!important;margin-bottom:0.5rem!important;}
[data-testid="stExpander"]>div:last-child{background:transparent!important;border-top:none!important;padding:0.25rem 0 0!important;}

/* ── ALERTS ── */
.stSuccess>div{background:rgba(0,217,160,0.08)!important;border:1px solid rgba(0,217,160,0.25)!important;border-radius:var(--r2)!important;color:var(--teal)!important;}
.stError>div{background:rgba(255,95,126,0.08)!important;border:1px solid rgba(255,95,126,0.25)!important;border-radius:var(--r2)!important;}
.stInfo>div{background:var(--blue-s)!important;border:1px solid rgba(59,158,255,0.2)!important;border-radius:var(--r2)!important;}
.stWarning>div{background:rgba(255,183,77,0.08)!important;border:1px solid rgba(255,183,77,0.2)!important;border-radius:var(--r2)!important;}

/* ── ANIMATIONS ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:translateY(0);}}
@keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
@keyframes glow{0%,100%{box-shadow:0 6px 28px rgba(59,158,255,0.35);}50%{box-shadow:0 6px 40px rgba(59,158,255,0.6);}}
@keyframes ecg{0%{background-position:0 0;}100%{background-position:-1440px 0;}}
@keyframes pulse2{0%,100%{opacity:1;}50%{opacity:0.4;}}
@keyframes spin2{to{transform:rotate(360deg);}}
.au{animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) both;}
.au2{animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.08s both;}
.au3{animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.16s both;}
.au4{animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.24s both;}

/* ── RING GAUGE ── */
.ring-svg{transform:rotate(-90deg);}
.ring-track{fill:none;stroke:var(--surf3);stroke-width:14;}
.ring-fill-high{fill:none;stroke:var(--rose);stroke-width:14;stroke-linecap:round;stroke-dasharray:339;}
.ring-fill-low{fill:none;stroke:var(--teal);stroke-width:14;stroke-linecap:round;stroke-dasharray:339;}

/* ── SECTION CARDS (classifier) ── */
.section-card{background:var(--surf);border:1px solid var(--bord);border-radius:16px;
    padding:1.2rem 1.4rem 0.9rem;margin-bottom:0.75rem;position:relative;overflow:hidden;}
.sc-top-blue{position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--blue),var(--purple),transparent 75%);}
.sc-top-smoke{position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#94a3b8,#64748b,transparent 75%);}
.sc-top-contra{position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--purple),var(--rose),transparent 75%);}
.sc-top-std{position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--teal),var(--blue),transparent 75%);}
.sc-top-dx{position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--amber),var(--rose),transparent 75%);}
.section-eyebrow{font-family:var(--fm);font-size:0.55rem;letter-spacing:0.22em;text-transform:uppercase;color:var(--txt3);margin-bottom:0.15rem;}
.section-title{font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);display:flex;align-items:baseline;gap:0.5rem;margin-bottom:0.1rem;}
.section-subtitle{font-size:0.72rem;font-weight:400;color:var(--txt3);}
.section-desc{font-size:0.72rem;color:var(--txt3);line-height:1.55;margin-bottom:0.9rem;margin-top:0.05rem;}

/* ── FIELD ROWS ── */
.field-row{display:grid;grid-template-columns:1fr auto;align-items:center;
    background:var(--surf2);border:1px solid var(--bord);border-radius:10px;
    padding:0.6rem 0.9rem;margin-bottom:0.45rem;gap:1rem;transition:border-color 0.18s;}
.field-row:hover{border-color:var(--bord2);}
.field-label{font-size:0.76rem;font-weight:500;color:var(--txt2);}
.field-label small{display:block;font-size:0.65rem;color:var(--txt3);font-family:var(--fm);margin-top:1px;}
.yn-wrap{background:var(--surf2);border:1px solid var(--bord);border-radius:10px;
    padding:0.55rem 0.9rem;margin-bottom:0.45rem;display:flex;align-items:center;
    justify-content:space-between;transition:border-color 0.18s;}
.yn-wrap:hover{border-color:var(--bord2);}
.yn-label{font-size:0.76rem;font-weight:500;color:var(--txt2);}

/* ── PROGRESS BAR ── */
.prog-wrap{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r2);
    padding:0.65rem 1.25rem;margin-bottom:1.25rem;display:flex;align-items:center;gap:1rem;}
.prog-lbl{font-family:var(--fm);font-size:0.58rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--txt3);white-space:nowrap;}
.prog-bar{flex:1;background:var(--surf3);border-radius:100px;height:4px;}
.prog-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--blue),var(--purple));transition:width 0.4s ease;}
.prog-pct{font-family:var(--fm);font-size:0.7rem;color:var(--blue);white-space:nowrap;font-weight:500;}

/* ── RUN BUTTON ── */
.run-btn-wrap .stButton>button{
    background:linear-gradient(135deg,#1566c8,#3b9eff,#6b5fff)!important;
    font-size:1rem!important;font-weight:700!important;padding:1rem 2.5rem!important;
    border-radius:14px!important;letter-spacing:0.02em!important;width:100%!important;
    animation:glow 3s ease infinite!important;}
.run-btn-wrap .stButton>button:hover{transform:translateY(-3px)!important;box-shadow:0 12px 40px rgba(59,158,255,0.55)!important;}

/* ── CARDS & SHARED ── */
.card{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);padding:1.5rem;}
.card-sm{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);padding:1rem 1.25rem;}
.topbar{display:flex;align-items:center;gap:1rem;padding:0.8rem 1.4rem;background:rgba(11,18,32,0.9);
    backdrop-filter:blur(20px);border:1px solid var(--bord);border-radius:var(--r);
    margin-bottom:1.75rem;position:sticky;top:0.5rem;z-index:99;box-shadow:var(--shadow2);animation:fadeIn 0.4s ease both;}
.topbar-logo{font-family:var(--fd);font-size:1.1rem;color:#fff;letter-spacing:-0.02em;white-space:nowrap;}
.topbar-logo em{color:var(--blue);font-style:normal;}
.topbar-sep{width:1px;height:16px;background:var(--bord2);}
.topbar-page{font-family:var(--fm);font-size:0.6rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--txt2);}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:0.6rem;}
.tb-pill{font-family:var(--fm);font-size:0.55rem;letter-spacing:0.12em;text-transform:uppercase;
    padding:3px 11px;border-radius:20px;background:rgba(59,158,255,0.1);color:var(--blue);border:1px solid rgba(59,158,255,0.2);}
.tb-ico{width:32px;height:32px;border-radius:8px;cursor:pointer;display:flex;align-items:center;justify-content:center;
    font-size:0.95rem;background:var(--surf2);border:1px solid var(--bord);transition:all 0.2s;position:relative;}
.tb-ico:hover{background:var(--surf3);border-color:var(--bord2);}
.tb-badge{position:absolute;top:-4px;right:-4px;width:16px;height:16px;border-radius:50%;
    background:var(--rose);color:#fff;font-size:0.5rem;font-weight:700;
    display:flex;align-items:center;justify-content:center;border:2px solid var(--bg);}
.tb-avatar{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--purple));
    display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:#fff;
    border:2px solid var(--bord2);cursor:pointer;transition:all 0.2s;}
.sb-logo{font-family:var(--fd);font-size:1.4rem;color:#fff;letter-spacing:-0.02em;padding-bottom:1rem;margin-bottom:1rem;border-bottom:1px solid var(--bord);}
.sb-logo em{color:var(--blue);font-style:normal;}
.sb-user{display:flex;align-items:center;gap:0.75rem;background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.75rem;margin-bottom:1.25rem;}
.sb-avatar{width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--purple));display:flex;align-items:center;justify-content:center;font-size:0.95rem;font-weight:700;color:#fff;flex-shrink:0;}
.sb-name{font-size:0.82rem;font-weight:600;color:var(--txt);}
.sb-role{font-size:0.68rem;color:var(--txt3);font-family:var(--fm);}
.sb-section{font-family:var(--fm);font-size:0.55rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--txt3);padding:0.3rem 0.6rem;margin-bottom:0.35rem;}
.sb-divider{border:none;border-top:1px solid var(--bord);margin:0.85rem 0;}
.sb-status{display:flex;align-items:center;gap:0.5rem;font-size:0.7rem;color:var(--txt3);padding:0.4rem 0.9rem;}
.sb-dot{width:6px;height:6px;border-radius:50%;background:var(--teal);box-shadow:0 0 6px var(--teal);flex-shrink:0;}
.team-card{display:flex;align-items:center;gap:0.9rem;background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.9rem 1.1rem;margin-bottom:0.6rem;transition:all 0.2s;}
.team-card:hover{border-color:var(--bord2);transform:translateX(4px);}
.team-av{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;}
.team-name{font-size:0.84rem;font-weight:600;color:var(--txt);}
.team-role{font-size:0.7rem;color:var(--txt3);font-family:var(--fm);}
.tech-pill{display:inline-flex;align-items:center;gap:0.3rem;background:var(--surf3);border:1px solid var(--bord);border-radius:6px;padding:4px 10px;font-size:0.73rem;color:var(--txt2);margin:0.2rem;transition:all 0.2s;}
.tech-pill:hover{border-color:var(--blue);color:var(--blue);}
.tl-item{display:flex;gap:1rem;margin-bottom:1rem;}
.tl-dot{width:9px;height:9px;border-radius:50%;background:var(--blue);margin-top:5px;flex-shrink:0;box-shadow:0 0 8px var(--blue);}
.tl-yr{font-family:var(--fm);font-size:0.62rem;color:var(--blue);letter-spacing:0.08em;margin-bottom:0.15rem;}
.tl-txt{font-size:0.8rem;color:var(--txt2);line-height:1.6;}
.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.65rem;margin-top:0.75rem;}
.metric-box{background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:1.1rem;text-align:center;transition:all 0.2s;}
.metric-box:hover{border-color:var(--bord2);transform:scale(1.03);}
.metric-v{font-family:var(--fd);font-size:1.9rem;font-weight:700;color:var(--blue);}
.metric-l{font-size:0.68rem;color:var(--txt3);}
.contact-method{display:flex;align-items:center;gap:0.9rem;background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.9rem 1.1rem;margin-bottom:0.55rem;transition:all 0.2s;}
.contact-method:hover{border-color:var(--bord2);transform:translateX(4px);}
.cm-ico{width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;}
.cm-lbl{font-size:0.68rem;color:var(--txt3);margin-bottom:0.1rem;}
.cm-val{font-size:0.83rem;font-weight:500;color:var(--txt);}
.page-hdr{display:flex;align-items:flex-start;gap:1.25rem;margin-bottom:1.5rem;}
.hdr-ico{width:50px;height:50px;flex-shrink:0;background:linear-gradient(135deg,#152040,#0d1830);border:1px solid rgba(59,158,255,0.25);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.35rem;box-shadow:0 0 20px rgba(59,158,255,0.1);}
.hdr-h{font-family:var(--fd)!important;font-size:1.75rem;color:#fff;margin:0 0 0.2rem;font-weight:700;line-height:1.2;}
.hdr-p{font-size:0.82rem;color:var(--txt2);margin:0;line-height:1.6;}
.lbl{font-family:var(--fm);font-size:0.56rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--txt3);margin-bottom:0.55rem;padding-left:1px;}
.notice{display:flex;gap:0.8rem;background:var(--surf2);border:1px solid var(--bord);border-left:3px solid var(--teal);border-radius:var(--r2);padding:0.8rem 1rem;margin-bottom:1.25rem;font-size:0.78rem;color:var(--txt2);line-height:1.65;}
.notice.err{border-left-color:var(--rose);}
.notice code{background:var(--surf3);padding:2px 6px;border-radius:4px;font-family:var(--fm);font-size:0.76em;color:var(--blue);}
.disclaimer{background:rgba(255,183,77,0.04);border:1px solid rgba(255,183,77,0.15);border-left:3px solid var(--amber);border-radius:var(--r2);padding:0.85rem 1.15rem;font-size:0.76rem;color:var(--txt2);line-height:1.7;}
.summary-g{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:0.55rem;margin-bottom:1.25rem;}
.sum-item{background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.65rem 0.9rem;}
.sum-lbl{font-size:0.65rem;color:var(--txt3);margin-bottom:0.15rem;}
.sum-val{font-family:var(--fm);font-size:0.84rem;color:var(--txt);font-weight:500;}
.dropdown{background:var(--surf2);border:1px solid var(--bord2);border-radius:var(--r);padding:0.5rem;min-width:260px;box-shadow:var(--shadow);animation:fadeUp 0.2s ease both;}
.dropdown-item{display:flex;align-items:center;gap:0.75rem;padding:0.6rem 0.75rem;border-radius:var(--r3);font-size:0.82rem;color:var(--txt2);transition:all 0.15s;}
.dropdown-item:hover{background:var(--surf3);color:var(--txt);}
.dropdown-sep{height:1px;background:var(--bord);margin:0.3rem 0;}
.dropdown-header{font-family:var(--fm);font-size:0.55rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--txt3);padding:0.4rem 0.75rem 0.2rem;}
.notif-unread{background:var(--blue-s);border-left:2px solid var(--blue);}
.notif-dot{width:7px;height:7px;border-radius:50%;background:var(--blue);flex-shrink:0;box-shadow:0 0 6px var(--blue);}
.hist-item{display:flex;align-items:center;gap:1rem;background:var(--surf);border:1px solid var(--bord);border-radius:var(--r2);padding:0.9rem 1.25rem;margin-bottom:0.5rem;transition:all 0.2s;}
.hist-item:hover{border-color:var(--bord2);transform:translateX(4px);}
.hist-risk-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
.hist-risk-dot.high{background:var(--rose);box-shadow:0 0 8px var(--rose);}
.hist-risk-dot.low{background:var(--teal);box-shadow:0 0 8px var(--teal);}
.hist-info{flex:1;}
.hist-name{font-size:0.84rem;font-weight:600;color:var(--txt);}
.hist-meta{font-size:0.7rem;color:var(--txt3);font-family:var(--fm);margin-top:0.1rem;}
.hist-badge{font-family:var(--fm);font-size:0.62rem;font-weight:600;letter-spacing:0.08em;padding:3px 10px;border-radius:20px;}
.hist-badge.high{background:rgba(255,95,126,0.12);color:var(--rose);border:1px solid rgba(255,95,126,0.25);}
.hist-badge.low{background:rgba(0,217,160,0.1);color:var(--teal);border:1px solid rgba(0,217,160,0.2);}
.hist-score{font-family:var(--fd);font-size:1.4rem;font-weight:700;}
.hist-score.high{color:var(--rose);}
.hist-score.low{color:var(--teal);}
.cal-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:3px;margin:0.75rem 0;}
.cal-dh{text-align:center;font-family:var(--fm);font-size:0.58rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--txt3);padding:5px 0;}
.cal-day{aspect-ratio:1;display:flex;align-items:center;justify-content:center;border-radius:var(--r3);font-size:0.78rem;color:var(--txt2);transition:all 0.15s;font-family:var(--fm);cursor:pointer;border:1px solid transparent;}
.cal-day:hover{background:var(--surf3);color:var(--txt);border-color:var(--bord);}
.cal-day.today{background:var(--blue-g);color:var(--blue);border-color:rgba(59,158,255,0.3);font-weight:700;}
.cal-day.has-appt{background:rgba(0,217,160,0.1);color:var(--teal);border-color:rgba(0,217,160,0.25);font-weight:600;}
.cal-day.empty{cursor:default;opacity:0;}
.appt-row{display:flex;align-items:center;gap:0.75rem;background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.7rem 1rem;margin-bottom:0.4rem;transition:all 0.2s;}
.appt-row:hover{border-color:var(--bord2);}
.about-hero{background:linear-gradient(135deg,#06101e,#0a1628);border:1px solid var(--bord2);border-radius:var(--r);padding:2.5rem;margin-bottom:1.25rem;position:relative;overflow:hidden;}
.about-hero::after{content:'';position:absolute;top:0;right:0;bottom:0;width:40%;background:radial-gradient(ellipse at right center,rgba(59,158,255,0.06),transparent 70%);}
hr{border-color:var(--bord)!important;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def goto(page):
    st.session_state.page = page
    st.session_state.show_notif = False
    st.session_state.show_profile_menu = False
    st.rerun()

def unread_count():
    return sum(1 for n in st.session_state.notifications if not n["read"])


# ═══════════════════════════════════════════════════════════════════
# FIELD RENDERERS  (number inputs — stable, no slider jitter)
# ═══════════════════════════════════════════════════════════════════
def render_number(label, key, mn, mx, default, step, unit=""):
    saved = st.session_state.input_values.get(key, default)
    if isinstance(step, float):
        saved = float(saved); mn, mx = float(mn), float(mx)
    else:
        saved = int(saved); mn, mx, step = int(mn), int(mx), int(step)
    unit_str = f" ({unit})" if unit else ""
    st.markdown(f"""<div class="field-row">
      <div class="field-label">{label}<small>Range: {mn} – {mx}{unit_str}</small></div>
    </div>""", unsafe_allow_html=True)
    val = st.number_input("", min_value=mn, max_value=mx, value=saved, step=step,
                          key=f"ni_{key}", label_visibility="collapsed")
    return val

def render_yesno(label, key, default=0):
    saved = st.session_state.input_values.get(key, default)
    saved_idx = 1 if int(saved) == 1 else 0
    st.markdown(f'<div class="yn-wrap"><span class="yn-label">{label}</span>', unsafe_allow_html=True)
    choice = st.radio("", options=["No", "Yes"], index=saved_idx, key=f"yn_{key}",
                      horizontal=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    return 1 if choice == "Yes" else 0


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
def render_sidebar():
    u = st.session_state.get("user_info", {})
    pg = st.session_state.page
    with st.sidebar:
        st.markdown('<div class="sb-logo">Cerv<em>AI</em></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="sb-user">
          <div class="sb-avatar">{u.get('initials','?')}</div>
          <div><div class="sb-name">{u.get('name','User')}</div>
          <div class="sb-role">{u.get('role','Clinician')}</div></div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
        for pid, ico, lbl in [("home","🏠","Dashboard"),("classifier","📋","Patient Classifier"),
                               ("history","🕐","History"),("calendar","📅","Calendar")]:
            cls = "nav-btn-active" if pg == pid else "nav-btn"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(f"{ico}  {lbl}", key=f"sb_{pid}"): goto(pid)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Info</div>', unsafe_allow_html=True)
        for pid, ico, lbl in [("about","ℹ️","About"),("contact","✉️","Contact Us")]:
            cls = "nav-btn-active" if pg == pid else "nav-btn"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(f"{ico}  {lbl}", key=f"sb_{pid}"): goto(pid)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Account</div>', unsafe_allow_html=True)
        for pid, ico, lbl in [("profile","👤","My Profile"),("settings","⚙️","Settings")]:
            cls = "nav-btn-active" if pg == pid else "nav-btn"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(f"{ico}  {lbl}", key=f"sb_{pid}"): goto(pid)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown(f"""<div class="sb-status"><div class="sb-dot"></div>
          {"Model Active" if model else "Demo Mode · No Model"}</div>""", unsafe_allow_html=True)
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
        if st.button("🚪  Sign Out", key="sb_logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.session_state.user_info = {}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TOPBAR
# ═══════════════════════════════════════════════════════════════════
def render_topbar():
    pg = st.session_state.page
    ico_lbl = {"home":"🏠 Dashboard","classifier":"📋 Classifier","history":"🕐 History",
               "calendar":"📅 Calendar","about":"ℹ️ About","contact":"✉️ Contact Us",
               "profile":"👤 Profile","settings":"⚙️ Settings","results":"📊 Results"}.get(pg,"🔬 Page")
    u = st.session_state.get("user_info", {})
    nc = unread_count()
    c_left, c_notif, c_search, c_prof = st.columns([6, 0.7, 0.7, 0.7])
    with c_left:
        st.markdown(f"""<div class="topbar">
          <div class="topbar-logo">Cerv<em>AI</em></div>
          <div class="topbar-sep"></div>
          <div class="topbar-page">{ico_lbl}</div>
          <div class="topbar-right"><div class="tb-pill">Research Use Only</div></div>
        </div>""", unsafe_allow_html=True)
    with c_notif:
        badge = f'<div class="tb-badge">{nc}</div>' if nc else ""
        st.markdown(f'<div class="tb-ico" style="margin-top:0.3rem;">{badge}🔔</div>', unsafe_allow_html=True)
        if st.button("🔔", key="tb_notif"):
            st.session_state.show_notif = not st.session_state.show_notif
            st.session_state.show_profile_menu = False
            st.rerun()
    with c_search:
        st.markdown('<div class="tb-ico" style="margin-top:0.3rem;">🔍</div>', unsafe_allow_html=True)
        if st.button("🔍", key="tb_search"): goto("classifier")
    with c_prof:
        st.markdown(f'<div class="tb-avatar" style="margin-top:0.3rem;">{u.get("initials","?")}</div>', unsafe_allow_html=True)
        if st.button(u.get("initials","?"), key="tb_avatar"):
            st.session_state.show_profile_menu = not st.session_state.show_profile_menu
            st.session_state.show_notif = False
            st.rerun()
    if st.session_state.show_notif:
        st.markdown('<div class="dropdown" style="max-width:320px;">', unsafe_allow_html=True)
        st.markdown('<div class="dropdown-header">Notifications</div>', unsafe_allow_html=True)
        for n in st.session_state.notifications:
            cls = "notif-unread" if not n["read"] else ""
            dot = '<div class="notif-dot"></div>' if not n["read"] else '<div style="width:7px"></div>'
            st.markdown(f"""<div class="dropdown-item {cls}">{dot}
              <div style="flex:1;"><div style="font-size:0.8rem;color:var(--txt);">{n['icon']} {n['text']}</div>
              <div style="font-size:0.65rem;color:var(--txt3);font-family:var(--fm);">{n['time']}</div></div></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("✓ Mark all read", key="mark_read"):
            for n in st.session_state.notifications: n["read"] = True
            st.session_state.show_notif = False
            st.rerun()
    if st.session_state.show_profile_menu:
        u2 = st.session_state.get("user_info", {})
        st.markdown(f"""<div class="dropdown">
          <div style="display:flex;align-items:center;gap:0.75rem;padding:0.75rem;background:var(--surf3);border-radius:var(--r3);margin-bottom:0.35rem;">
            <div class="sb-avatar" style="width:36px;height:36px;font-size:0.85rem;">{u2.get('initials','?')}</div>
            <div><div style="font-size:0.82rem;font-weight:600;color:var(--txt);">{u2.get('name','User')}</div>
            <div style="font-size:0.68rem;color:var(--txt3);">{u2.get('email','')}</div></div>
          </div><div class="dropdown-sep"></div></div>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("👤 Profile", key="pm_profile"): goto("profile")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("⚙️ Settings", key="pm_settings"): goto("settings")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="pm_logout"):
            st.session_state.logged_in = False; st.session_state.page = "login"
            st.session_state.user_info = {}; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE · LOGIN  (FIXED: inputs inside card, no overflow)
# ═══════════════════════════════════════════════════════════════════
def page_login():
    # Full-page centering via columns + vertical spacer
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, cc, _ = st.columns([1, 1.4, 1])
    with cc:
        # Logo above card
        st.markdown("""
        <div style="text-align:center;margin-bottom:1.5rem;">
          <div style="font-family:'Syne',sans-serif;font-size:2.8rem;color:#fff;
               letter-spacing:-0.04em;font-weight:800;line-height:1;">
            Cerv<span style="color:var(--blue);">AI</span>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
               letter-spacing:0.25em;text-transform:uppercase;color:#3d5270;
               margin-top:0.4rem;">Clinical Cervical Cancer Risk Platform</div>
        </div>
        """, unsafe_allow_html=True)

        # Card wrapping the form
        st.markdown("""
        <div style="background:var(--surf);border:1px solid var(--bord2);border-radius:20px;
             padding:2rem 2rem 1.5rem;animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.1s both;">
          <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
               color:var(--txt);margin-bottom:1.25rem;">🔐 Sign In to Your Account</div>
        </div>
        """, unsafe_allow_html=True)

        # Streamlit inputs (rendered natively — always inside the column width)
        email = st.text_input("Email Address", placeholder="you@gmail.com", key="li_email")
        pwd   = st.text_input("Password", type="password", placeholder="••••••••", key="li_pwd")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Sign In  →", key="li_btn"):
            user = check_login(email, pwd)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_info = {**user, "email": email.strip().lower()}
                st.session_state.page = "home"
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Check email and password.")

        st.markdown("""
        <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);
             padding:0.75rem 1rem;margin-top:1rem;font-family:'JetBrains Mono',monospace;
             font-size:0.68rem;color:#3d5270;line-height:1.9;">
          <strong style="color:#7e97be;">Team accounts:</strong><br>
          bakraouladomor@gmail.com / cerv2024<br>
          yassirjbili@gmail.com / cerv2024<br>
          ilyaselhadad@gmail.com / cerv2024<br>
          admin@cerv.ai / admin123
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE · HOME  (document-7 design, full hero + stats + features)
# ═══════════════════════════════════════════════════════════════════
def page_home():
    u = st.session_state.get("user_info", {})
    total_hist = len(st.session_state.history)
    high_risk  = sum(1 for h in st.session_state.history if h["pred"] == 1)
    low_risk   = total_hist - high_risk
    upcoming   = sum(1 for a in st.session_state.appointments
                     if datetime.datetime.strptime(a["date"],"%Y-%m-%d").date() >= datetime.date.today())

    st.markdown("""<style>
    .hero2{position:relative;border-radius:24px;overflow:hidden;border:1px solid var(--bord2);
        margin-bottom:1.5rem;min-height:360px;
        background:linear-gradient(135deg,#020a18 0%,#04111f 40%,#061428 100%);}
    .hero2-grid{position:absolute;inset:0;
        background-image:linear-gradient(rgba(59,158,255,0.06) 1px,transparent 1px),
                         linear-gradient(90deg,rgba(59,158,255,0.06) 1px,transparent 1px);
        background-size:48px 48px;}
    .hero2-glow-l{position:absolute;top:-100px;left:-100px;width:600px;height:600px;border-radius:50%;
        background:radial-gradient(circle,rgba(59,158,255,0.09) 0%,transparent 65%);}
    .hero2-glow-r{position:absolute;bottom:-80px;right:0;width:500px;height:500px;border-radius:50%;
        background:radial-gradient(circle,rgba(0,217,160,0.07) 0%,transparent 65%);}
    .hero2-ecg{position:absolute;bottom:0;left:0;right:0;height:45px;opacity:0.08;
        background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 50'%3E%3Cpolyline points='0,25 200,25 230,5 245,45 260,5 275,45 300,25 500,25 530,5 545,45 560,5 575,45 600,25 800,25 830,5 845,45 860,5 875,45 900,25 1100,25 1130,5 1145,45 1160,5 1175,45 1200,25 1440,25' fill='none' stroke='%2300d9a0' stroke-width='1.5'/%3E%3C/svg%3E") repeat-x;
        animation:ecg 5s linear infinite;}
    .hero2-inner{position:relative;z-index:2;padding:2.75rem 3rem 2.5rem;
        display:grid;grid-template-columns:1fr 300px;gap:3rem;align-items:center;}
    .hero2-badge{display:inline-flex;align-items:center;gap:0.5rem;
        font-family:var(--fm);font-size:0.58rem;letter-spacing:0.24em;text-transform:uppercase;
        color:var(--teal);background:rgba(0,217,160,0.08);border:1px solid rgba(0,217,160,0.25);
        padding:5px 16px;border-radius:20px;margin-bottom:1.25rem;}
    .hero2-badge-dot{width:6px;height:6px;border-radius:50%;background:var(--teal);
        box-shadow:0 0 8px var(--teal);animation:pulse2 2s ease infinite;}
    .hero2-title{font-family:var(--fd);font-size:3.4rem;color:#fff;line-height:1.05;
        letter-spacing:-0.04em;font-weight:800;margin-bottom:0.75rem;}
    .hero2-title .accent-teal{background:linear-gradient(135deg,var(--teal),var(--blue));
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
    .hero2-sub{font-size:0.93rem;color:var(--txt2);max-width:480px;line-height:1.85;margin-bottom:0;}
    .hero2-visual{position:relative;display:flex;align-items:center;justify-content:center;}
    .hero2-orb{position:absolute;inset:0;margin:auto;width:240px;height:240px;border-radius:50%;
        background:radial-gradient(circle,rgba(59,158,255,0.12) 0%,transparent 70%);
        border:1px solid rgba(59,158,255,0.12);}
    .hero2-orb2{position:absolute;inset:-20px;margin:auto;width:280px;height:280px;border-radius:50%;
        border:1px solid rgba(59,158,255,0.06);animation:spin2 20s linear infinite;}
    .hero2-img{width:200px;height:200px;border-radius:50%;object-fit:cover;position:relative;z-index:1;
        border:2px solid rgba(59,158,255,0.25);
        box-shadow:0 0 40px rgba(59,158,255,0.2),0 16px 48px rgba(0,0,0,0.6);}
    .hero2-img-tag{position:absolute;bottom:-8px;left:50%;transform:translateX(-50%);
        background:rgba(0,217,160,0.15);backdrop-filter:blur(12px);
        border:1px solid rgba(0,217,160,0.3);border-radius:20px;
        padding:4px 14px;font-size:0.68rem;color:var(--teal);font-family:var(--fm);
        white-space:nowrap;font-weight:600;z-index:2;}
    .hero2-float-card{position:absolute;background:rgba(11,18,32,0.9);backdrop-filter:blur(12px);
        border:1px solid var(--bord2);border-radius:12px;padding:0.6rem 0.9rem;z-index:3;}
    .hero2-float-card.tl{top:10px;left:-15px;}
    .hero2-float-card.br{bottom:10px;right:-15px;}
    .hfc-val{font-family:var(--fd);font-size:1.3rem;font-weight:700;}
    .hfc-lbl{font-size:0.62rem;color:var(--txt3);}
    .stats2{display:grid;grid-template-columns:repeat(4,1fr);gap:0.85rem;margin-bottom:1.5rem;}
    .stat2{position:relative;border-radius:16px;padding:1.4rem 1.5rem;overflow:hidden;
        border:1px solid var(--bord);transition:all 0.25s;cursor:default;}
    .stat2:hover{transform:translateY(-4px);border-color:var(--bord2);}
    .stat2-bg{position:absolute;bottom:-20px;right:-20px;width:100px;height:100px;border-radius:50%;opacity:0.06;}
    .stat2-ico-wrap{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;
        justify-content:center;font-size:1.25rem;margin-bottom:0.9rem;position:relative;z-index:1;}
    .stat2-val{font-family:var(--fd);font-size:2.4rem;font-weight:800;line-height:1;margin-bottom:0.2rem;position:relative;z-index:1;}
    .stat2-lbl{font-size:0.7rem;color:var(--txt3);position:relative;z-index:1;font-weight:500;}
    .stat2-trend{position:absolute;top:1rem;right:1rem;font-family:var(--fm);font-size:0.6rem;padding:2px 8px;border-radius:20px;z-index:1;}
    .stat2.blue{background:linear-gradient(135deg,#060e1c,#0b1628);}
    .stat2.blue .stat2-val{color:var(--blue);}
    .stat2.blue .stat2-ico-wrap{background:rgba(59,158,255,0.1);border:1px solid rgba(59,158,255,0.2);}
    .stat2.blue .stat2-bg{background:var(--blue);}
    .stat2.blue .stat2-trend{background:rgba(59,158,255,0.12);color:var(--blue);border:1px solid rgba(59,158,255,0.2);}
    .stat2.teal{background:linear-gradient(135deg,#030f12,#061519);}
    .stat2.teal .stat2-val{color:var(--teal);}
    .stat2.teal .stat2-ico-wrap{background:rgba(0,217,160,0.08);border:1px solid rgba(0,217,160,0.2);}
    .stat2.teal .stat2-bg{background:var(--teal);}
    .stat2.teal .stat2-trend{background:rgba(0,217,160,0.1);color:var(--teal);border:1px solid rgba(0,217,160,0.2);}
    .stat2.rose{background:linear-gradient(135deg,#0f0810,#170c14);}
    .stat2.rose .stat2-val{color:var(--rose);}
    .stat2.rose .stat2-ico-wrap{background:rgba(255,95,126,0.08);border:1px solid rgba(255,95,126,0.2);}
    .stat2.rose .stat2-bg{background:var(--rose);}
    .stat2.rose .stat2-trend{background:rgba(255,95,126,0.1);color:var(--rose);border:1px solid rgba(255,95,126,0.2);}
    .stat2.amber{background:linear-gradient(135deg,#0f0e07,#17140a);}
    .stat2.amber .stat2-val{color:var(--amber);}
    .stat2.amber .stat2-ico-wrap{background:rgba(255,183,77,0.08);border:1px solid rgba(255,183,77,0.2);}
    .stat2.amber .stat2-bg{background:var(--amber);}
    .stat2.amber .stat2-trend{background:rgba(255,183,77,0.1);color:var(--amber);border:1px solid rgba(255,183,77,0.2);}
    .feats2{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem;}
    .feat2{position:relative;border-radius:16px;padding:1.6rem;overflow:hidden;
        background:var(--surf);border:1px solid var(--bord);transition:all 0.25s;cursor:default;}
    .feat2:hover{border-color:var(--bord2);transform:translateY(-5px);box-shadow:0 12px 40px rgba(0,0,0,0.4);}
    .feat2::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
        transform:scaleX(0);transform-origin:left;transition:transform 0.35s;}
    .feat2:hover::after{transform:scaleX(1);}
    .feat2.f-blue::after{background:linear-gradient(90deg,var(--blue),var(--purple));}
    .feat2.f-teal::after{background:linear-gradient(90deg,var(--teal),var(--blue));}
    .feat2.f-purple::after{background:linear-gradient(90deg,var(--purple),var(--rose));}
    .feat2.f-amber::after{background:linear-gradient(90deg,var(--amber),var(--rose));}
    .feat2.f-rose::after{background:linear-gradient(90deg,var(--rose),var(--purple));}
    .feat2.f-green::after{background:linear-gradient(90deg,#22c55e,var(--teal));}
    .feat2-num{font-family:var(--fm);font-size:0.55rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--txt3);margin-bottom:0.9rem;}
    .feat2-ico{font-size:2rem;margin-bottom:0.75rem;display:block;}
    .feat2-title{font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:0.5rem;line-height:1.3;}
    .feat2-desc{font-size:0.76rem;color:var(--txt3);line-height:1.7;}
    .feat2-link{position:absolute;top:1.25rem;right:1.25rem;width:28px;height:28px;
        border-radius:8px;display:flex;align-items:center;justify-content:center;
        font-size:0.8rem;background:var(--surf2);border:1px solid var(--bord);color:var(--txt3);transition:all 0.2s;}
    .feat2:hover .feat2-link{background:var(--blue-g);border-color:rgba(59,158,255,0.3);color:var(--blue);}
    /* Home CTA buttons — same size, Run Assessment glows */
    .home-cta-row{display:flex;gap:0.75rem;margin-bottom:1.5rem;}
    .btn-primary .stButton>button{
        background:linear-gradient(135deg,#0a2a22,#0f3a2e)!important;
        color:var(--teal)!important;border:1.5px solid var(--teal)!important;
        box-shadow:0 0 18px rgba(0,217,160,0.3)!important;
        padding:0.78rem 1.5rem!important;font-size:0.88rem!important;
        width:100%!important;animation:home-glow 2.5s ease infinite!important;}
    .btn-primary .stButton>button:hover{
        background:linear-gradient(135deg,#0d3a2e,#12503f)!important;
        box-shadow:0 0 32px rgba(0,217,160,0.55)!important;transform:translateY(-2px)!important;}
    .btn-ghost2 .stButton>button{
        background:rgba(255,255,255,0.05)!important;color:var(--txt2)!important;
        border:1px solid rgba(255,255,255,0.1)!important;box-shadow:none!important;
        padding:0.78rem 1.5rem!important;font-size:0.88rem!important;width:100%!important;}
    .btn-ghost2 .stButton>button:hover{
        background:rgba(255,255,255,0.09)!important;color:var(--txt)!important;transform:none!important;box-shadow:none!important;}
    @keyframes home-glow{
        0%,100%{box-shadow:0 0 18px rgba(0,217,160,0.30);}
        50%{box-shadow:0 0 28px rgba(0,217,160,0.55);}}
    </style>""", unsafe_allow_html=True)

    # Hero
    st.markdown(f"""<div class="hero2 au">
      <div class="hero2-grid"></div>
      <div class="hero2-glow-l"></div><div class="hero2-glow-r"></div>
      <div class="hero2-ecg"></div>
      <div class="hero2-inner">
        <div>
          <div class="hero2-badge"><span class="hero2-badge-dot"></span>AI-Powered · Clinical Grade · Research Platform</div>
          <div class="hero2-title">Cervical Cancer<br>Risk <span class="accent-teal">Classifier</span></div>
          <div class="hero2-sub">Welcome back, <strong style="color:var(--txt);font-weight:700;">{u.get('name','Doctor')}</strong>.<br>
            Evidence-informed biopsy risk prediction trained on <strong style="color:var(--txt);">858 clinical records</strong> across <strong style="color:var(--txt);">36 risk factors</strong>.</div>
        </div>
        <div class="hero2-visual">
          <div class="hero2-orb"></div>
          <div class="hero2-orb2"></div>
          <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Papilloma_Virus_%28HPV%29_EM.jpg/480px-Papilloma_Virus_%28HPV%29_EM.jpg"
               class="hero2-img" alt="HPV"/>
          <div class="hero2-img-tag">🔬 HPV · Primary Causal Agent</div>
          <div class="hero2-float-card tl">
            <div class="hfc-val" style="color:var(--blue);">{total_hist}</div>
            <div class="hfc-lbl">Assessments</div>
          </div>
          <div class="hero2-float-card br">
            <div class="hfc-val" style="color:var(--teal);">99%</div>
            <div class="hfc-lbl">HPV-linked</div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # CTA buttons — equal width, Run Assessment distinguished
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("🔬  Run Assessment", key="h_assess"): goto("classifier")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-ghost2">', unsafe_allow_html=True)
        if st.button("🕐  View History", key="h_hist"): goto("history")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="btn-ghost2">', unsafe_allow_html=True)
        if st.button("📅  Calendar", key="h_cal"): goto("calendar")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats
    s1, s2, s3, s4 = st.columns(4, gap="small")
    with s1:
        st.markdown(f"""<div class="stat2 blue"><div class="stat2-bg"></div>
          <div class="stat2-trend">All time</div><div class="stat2-ico-wrap">🔬</div>
          <div class="stat2-val">{total_hist}</div><div class="stat2-lbl">Assessments Run</div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="stat2 teal"><div class="stat2-bg"></div>
          <div class="stat2-trend">✓ Safe</div><div class="stat2-ico-wrap">✅</div>
          <div class="stat2-val">{low_risk}</div><div class="stat2-lbl">Low Risk Results</div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="stat2 rose"><div class="stat2-bg"></div>
          <div class="stat2-trend">⚠ Review</div><div class="stat2-ico-wrap">⚠️</div>
          <div class="stat2-val">{high_risk}</div><div class="stat2-lbl">High Risk Flagged</div></div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""<div class="stat2 amber"><div class="stat2-bg"></div>
          <div class="stat2-trend">Upcoming</div><div class="stat2-ico-wrap">📅</div>
          <div class="stat2-val">{upcoming}</div><div class="stat2-lbl">Appointments</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("""<div class="feats2 au3">
      <div class="feat2 f-blue"><div class="feat2-num">01 · Profiling</div><div class="feat2-link">↗</div>
        <span class="feat2-ico">📋</span><div class="feat2-title">Comprehensive Patient Profiling</div>
        <div class="feat2-desc">5 clinical modules · 36 risk factors covering demographics, STDs, contraceptives and prior diagnoses.</div></div>
      <div class="feat2 f-teal"><div class="feat2-num">02 · Engine</div><div class="feat2-link">↗</div>
        <span class="feat2-ico">⚡</span><div class="feat2-title">Instant AI Prediction</div>
        <div class="feat2-desc">XGBoost + SMOTE oversampling. Sub-second inference with probability scoring.</div></div>
      <div class="feat2 f-purple"><div class="feat2-num">03 · Insight</div><div class="feat2-link">↗</div>
        <span class="feat2-ico">📊</span><div class="feat2-title">Explainable AI Results</div>
        <div class="feat2-desc">Feature importance charts reveal which clinical factors drove each prediction.</div></div>
      <div class="feat2 f-amber"><div class="feat2-num">04 · Audit</div><div class="feat2-link">↗</div>
        <span class="feat2-ico">🕐</span><div class="feat2-title">Full Assessment History</div>
        <div class="feat2-desc">Every prediction logged with patient data, risk scores and timestamps.</div></div>
      <div class="feat2 f-rose"><div class="feat2-num">05 · Schedule</div><div class="feat2-link">↗</div>
        <span class="feat2-ico">📅</span><div class="feat2-title">Appointment Calendar</div>
        <div class="feat2-desc">Schedule follow-ups, biopsies and consultations with interactive calendar.</div></div>
      <div class="feat2 f-green"><div class="feat2-num">06 · Privacy</div><div class="feat2-link">↗</div>
        <span class="feat2-ico">🔒</span><div class="feat2-title">Secure & Private</div>
        <div class="feat2-desc">Zero data persistence. All inputs processed in-memory — cleared on session end.</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="disclaimer au4"><strong>⚕ Research &amp; Educational Use Only.</strong>
    CervAI is a machine learning prototype. It is <em>not a certified medical device</em>.</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE · CLASSIFIER  (our version — number inputs, non-blocking overlay)
# ═══════════════════════════════════════════════════════════════════
def page_classifier():
    # ── If overlay showing, render result + buttons ONLY ─────────────
    if st.session_state.get("show_result_overlay") and st.session_state.prediction:
        p      = st.session_state.prediction
        pred   = p["pred"]; pct = p["pct"]
        rc     = "high" if pred == 1 else "low"
        accent = "var(--rose)" if pred == 1 else "var(--teal)"
        verdict    = "Biopsy Indicated"     if pred == 1 else "Biopsy Not Indicated"
        badge_txt  = "⚠ HIGH RISK"          if pred == 1 else "✓ LOW RISK"
        detail     = ("The model predicts a <strong>positive biopsy result</strong>. Warrants immediate specialist referral."
                      if pred == 1 else
                      "The model predicts a <strong>negative biopsy result</strong>. Continue routine screening per guidelines.")
        badge_bg   = "rgba(255,95,126,0.12)" if pred == 1 else "rgba(0,217,160,0.1)"
        badge_brd  = "rgba(255,95,126,0.3)"  if pred == 1 else "rgba(0,217,160,0.25)"
        bar_grad   = "linear-gradient(90deg,#ff8fa0,#cc1133)" if pred == 1 else "linear-gradient(90deg,#7ee8be,#22c55e)"
        ring_cls   = "ring-fill-high" if pred == 1 else "ring-fill-low"
        offset     = round(339 * (1 - pct / 100), 1)
        top_col    = "rgba(255,95,126,0.08)" if pred == 1 else "rgba(0,217,160,0.08)"
        bord_c     = "rgba(255,95,126,0.3)"  if pred == 1 else "rgba(0,217,160,0.3)"
        glow_c     = "rgba(255,95,126,0.12)" if pred == 1 else "rgba(0,217,160,0.1)"

        st.markdown(f"""<div style="background:linear-gradient(135deg,{top_col},rgba(4,8,15,0.5));
            border:1.5px solid {bord_c};border-radius:20px;padding:0;max-width:720px;
            margin:0 auto 1.5rem;animation:fadeUp 0.4s ease both;overflow:hidden;
            box-shadow:0 24px 80px rgba(0,0,0,0.6),0 0 60px {glow_c};">
          <div style="height:3px;background:{'linear-gradient(90deg,var(--rose),#cc1133)' if pred==1 else 'linear-gradient(90deg,var(--teal),#22c55e)'};"></div>
          <div style="padding:2rem 2rem 0;">
            <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.25rem;flex-wrap:wrap;">
              <span style="font-family:var(--fm);font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;
                    color:{accent};padding:4px 14px;border-radius:20px;
                    background:{badge_bg};border:1px solid {badge_brd};">{badge_txt}</span>
              <span style="font-family:var(--fm);font-size:0.65rem;color:var(--txt3);">
                {p.get('patient_id','—')} &nbsp;·&nbsp; {p.get('timestamp','—')}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 140px;gap:2rem;align-items:center;">
              <div>
                <div style="font-family:var(--fd);font-size:2.2rem;font-weight:800;color:{accent};line-height:1.1;margin-bottom:0.5rem;">{verdict}</div>
                <div style="font-size:0.84rem;color:var(--txt2);line-height:1.75;">{detail}</div>
              </div>
              <div style="position:relative;width:140px;height:140px;flex-shrink:0;">
                <svg class="ring-svg" width="140" height="140" viewBox="0 0 120 120">
                  <circle class="ring-track" cx="60" cy="60" r="54"/>
                  <circle class="{ring_cls}" cx="60" cy="60" r="54" style="stroke-dashoffset:{offset};"/>
                </svg>
                <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">
                  <div style="font-family:var(--fd);font-size:2.6rem;font-weight:800;line-height:1;color:{accent};">{pct}%</div>
                  <div style="font-family:var(--fm);font-size:0.52rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--txt3);">Risk Score</div>
                </div>
              </div>
            </div>
          </div>
          <div style="padding:1.25rem 2rem 2rem;">
            <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem;">
              <div style="font-family:var(--fm);font-size:0.58rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--txt3);margin-bottom:0.5rem;">Probability</div>
              <div style="background:var(--surf4);border-radius:100px;height:10px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;border-radius:100px;background:{bar_grad};"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-family:var(--fm);font-size:0.63rem;color:var(--txt3);margin-top:0.3rem;">
                <span>0%</span><span style="color:{accent};font-weight:600;">{pct}% predicted</span><span>100%</span>
              </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;margin-bottom:1rem;">
              <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:0.8rem;text-align:center;">
                <div style="font-family:var(--fm);font-size:0.55rem;color:var(--txt3);margin-bottom:0.25rem;">MODEL</div>
                <div style="font-size:0.82rem;font-weight:600;color:var(--txt);">XGBoost</div>
              </div>
              <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:0.8rem;text-align:center;">
                <div style="font-family:var(--fm);font-size:0.55rem;color:var(--txt3);margin-bottom:0.25rem;">AGE</div>
                <div style="font-size:0.82rem;font-weight:600;color:var(--txt);">{p.get('age','?')} yrs</div>
              </div>
              <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:0.8rem;text-align:center;">
                <div style="font-family:var(--fm);font-size:0.55rem;color:var(--txt3);margin-bottom:0.25rem;">VERDICT</div>
                <div style="font-size:0.82rem;font-weight:600;color:{accent};">{"POSITIVE" if pred==1 else "NEGATIVE"}</div>
              </div>
            </div>
            <div style="background:rgba(255,183,77,0.04);border:1px solid rgba(255,183,77,0.15);
                 border-left:3px solid var(--amber);border-radius:10px;padding:0.7rem 1rem;
                 font-size:0.73rem;color:var(--txt2);line-height:1.6;">
              <strong style="color:var(--amber);">⚕ Clinical Disclaimer:</strong> AI prediction only. Consult a qualified provider.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<p style="text-align:center;font-family:var(--fm);font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--txt3);margin-bottom:0.75rem;">— Choose an action —</p>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3, gap="medium")
        with b1:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("← Return to Form", key="ov_close"):
                st.session_state.show_result_overlay = False; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("📊  Full Report", key="ov_full"):
                st.session_state.show_result_overlay = False; goto("results")
            st.markdown('</div>', unsafe_allow_html=True)
        with b3:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("🔄  New Assessment", key="ov_new"):
                st.session_state.show_result_overlay = False
                st.session_state.prediction = None
                st.session_state.input_values = {}
                for k in list(st.session_state.keys()):
                    if k.startswith("ni_") or k.startswith("yn_"):
                        del st.session_state[k]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        return  # don't render form while overlay is showing

    # ── NORMAL FORM ────────────────────────────────────────────────
    st.markdown("""<div class="page-hdr au">
      <div class="hdr-ico">📋</div>
      <div><div class="hdr-h">Patient Risk Profile</div>
      <p class="hdr-p">Fill in the patient's clinical data below, then run the prediction.</p></div>
    </div>""", unsafe_allow_html=True)

    c_pid, c_status = st.columns([2, 3])
    with c_pid:
        pat_id = st.text_input("Patient ID / Reference", placeholder="e.g. PAT-2024-0042", key="pat_id_field")
    with c_status:
        if model is None:
            st.markdown('<div class="notice err" style="margin-top:1.7rem;"><span>⚠️</span><div><strong>Model not loaded.</strong> Place <code>xgboost_model.joblib</code> in <code>models/</code>.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="notice" style="margin-top:1.7rem;"><span>✅</span><div>Model ready · <code>{model_path}</code></div></div>', unsafe_allow_html=True)

    total_f = 36
    filled  = sum(1 for v in st.session_state.input_values.values() if v not in (0, 0.0))
    pct_p   = int(filled / total_f * 100)
    st.markdown(f"""<div class="prog-wrap">
      <div class="prog-lbl">Profile Completion</div>
      <div class="prog-bar"><div class="prog-fill" style="width:{pct_p}%;"></div></div>
      <div class="prog-pct">{pct_p}%</div>
    </div>""", unsafe_allow_html=True)

    input_values = {}
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""<div class="section-card"><div class="sc-top-blue"></div>
          <div class="section-eyebrow">👤 · Module 01</div>
          <div class="section-title">Demographics <span class="section-subtitle">Patient background</span></div>
          <div class="section-desc">Age, sexual history and reproductive background.</div></div>""", unsafe_allow_html=True)
        input_values["Age"]                     = render_number("Age", "Age", 13, 84, 25, 1, "yrs")
        input_values["Num_sexual_partners"]      = render_number("Lifetime sexual partners", "Num_sexual_partners", 0, 28, 2, 1)
        input_values["First_sexual_intercourse"] = render_number("Age at first intercourse", "First_sexual_intercourse", 10, 32, 17, 1, "yrs")
        input_values["Num_of_pregnancies"]       = render_number("Number of pregnancies", "Num_of_pregnancies", 0, 11, 1, 1)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""<div class="section-card"><div class="sc-top-smoke"></div>
          <div class="section-eyebrow">🚬 · Module 02</div>
          <div class="section-title">Smoking History <span class="section-subtitle">Tobacco exposure</span></div>
          <div class="section-desc">Smoking is a significant co-factor in HPV persistence.</div></div>""", unsafe_allow_html=True)
        input_values["Smokes"]            = render_yesno("Smoker (current or former)", "Smokes")
        input_values["Smokes_years"]      = render_number("Smoking duration (years)", "Smokes_years", 0.0, 37.0, 0.0, 0.5)
        input_values["Smokes_packs_year"] = render_number("Intensity (packs/year)", "Smokes_packs_year", 0.0, 40.0, 0.0, 0.5)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""<div class="section-card"><div class="sc-top-dx"></div>
          <div class="section-eyebrow">🔬 · Module 05</div>
          <div class="section-title">Prior Diagnoses & Tests</div>
          <div class="section-desc">Previous diagnoses and gynecological screening results.</div></div>""", unsafe_allow_html=True)
        for lbl, key in [
            ("Prior cancer diagnosis","Dx_Cancer"),("Prior CIN diagnosis","Dx_CIN"),
            ("Prior HPV diagnosis","Dx_HPV"),("General diagnosis flag","Dx"),
            ("Hinselmann test positive","Hinselmann"),("Schiller test positive","Schiller"),
            ("Cytology positive","Citology"),
        ]:
            input_values[key] = render_yesno(lbl, key)

    with col2:
        st.markdown("""<div class="section-card"><div class="sc-top-contra"></div>
          <div class="section-eyebrow">💊 · Module 03</div>
          <div class="section-title">Contraceptive Use <span class="section-subtitle">Hormonal & IUD</span></div>
          <div class="section-desc">Long-term hormonal use is associated with increased cervical cancer risk.</div></div>""", unsafe_allow_html=True)
        input_values["Hormonal_Contraceptives"]       = render_yesno("Using hormonal contraceptives", "Hormonal_Contraceptives")
        input_values["Hormonal_Contraceptives_years"] = render_number("Duration of use (years)", "Hormonal_Contraceptives_years", 0.0, 30.0, 0.0, 0.5)
        input_values["IUD"]                           = render_yesno("IUD in use or history", "IUD")
        input_values["IUD_years"]                     = render_number("IUD duration (years)", "IUD_years", 0.0, 19.0, 0.0, 0.5)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""<div class="section-card"><div class="sc-top-std"></div>
          <div class="section-eyebrow">🦠 · Module 04</div>
          <div class="section-title">STD History <span class="section-subtitle">Sexually transmitted infections</span></div>
          <div class="section-desc">STD co-infections modulate HPV persistence and dysplasia progression.</div></div>""", unsafe_allow_html=True)
        input_values["STDs"]        = render_yesno("History of STD(s)", "STDs")
        input_values["STDs_number"] = render_number("Number of distinct STDs", "STDs_number", 0, 4, 0, 1)
        with st.expander("🔽  Specific STD types", expanded=False):
            for lbl, key in [
                ("Condylomatosis","STDs_condylomatosis"),("Cervical condylomatosis","STDs_cervical_condylomatosis"),
                ("Vaginal condylomatosis","STDs_vaginal_condylomatosis"),("Vulvo-perineal condylomatosis","STDs_vulvo_perineal_condylomatosis"),
                ("Syphilis","STDs_syphilis"),("Pelvic inflammatory disease","STDs_pelvic_inflammatory_disease"),
                ("Genital herpes","STDs_genital_herpes"),("Molluscum contagiosum","STDs_molluscum_contagiosum"),
                ("AIDS","STDs_AIDS"),("HIV","STDs_HIV"),("Hepatitis B","STDs_Hepatitis_B"),("HPV","STDs_HPV"),
            ]:
                input_values[key] = render_yesno(lbl, key)
        input_values["STDs_Number_of_diagnosis"]        = render_number("Total STD diagnoses", "STDs_Number_of_diagnosis", 0, 3, 0, 1)
        input_values["STDs_Time_since_first_diagnosis"] = render_number("Years since first STD dx", "STDs_Time_since_first_diagnosis", 0.0, 29.0, 0.0, 0.5)
        input_values["STDs_Time_since_last_diagnosis"]  = render_number("Years since last STD dx",  "STDs_Time_since_last_diagnosis",  0.0, 29.0, 0.0, 0.5)

    for key in ["STDs_condylomatosis","STDs_cervical_condylomatosis","STDs_vaginal_condylomatosis",
                "STDs_vulvo_perineal_condylomatosis","STDs_syphilis","STDs_pelvic_inflammatory_disease",
                "STDs_genital_herpes","STDs_molluscum_contagiosum","STDs_AIDS","STDs_HIV","STDs_Hepatitis_B","STDs_HPV"]:
        if key not in input_values:
            input_values[key] = st.session_state.input_values.get(key, 0)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="background:linear-gradient(135deg,rgba(59,158,255,0.06),rgba(123,111,255,0.04));
        border:1px solid rgba(59,158,255,0.14);border-radius:16px;padding:1.5rem;text-align:center;">
      <div style="font-family:var(--fd);font-size:1.1rem;font-weight:700;color:var(--txt);margin-bottom:0.3rem;">Ready to Run Assessment?</div>
      <div style="font-size:0.8rem;color:var(--txt2);margin-bottom:1.1rem;">Review the profile above, then click below.</div>
    </div>""", unsafe_allow_html=True)
    _, bcol, _ = st.columns([1, 2, 1])
    with bcol:
        st.markdown('<div class="run-btn-wrap">', unsafe_allow_html=True)
        run_clicked = st.button("🔬  Run Biopsy Risk Prediction", disabled=(model is None), key="predict_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer" style="margin-top:1rem;"><strong>⚕ Research Use Only.</strong> Does not replace clinical judgment.</div>', unsafe_allow_html=True)

    if run_clicked:
        st.session_state.input_values = dict(input_values)
        row = {KEY_TO_COL[k]: v for k, v in input_values.items() if k in KEY_TO_COL}
        df  = pd.DataFrame([row])

        # ── Debug: show what was actually collected ──────────────────
        with st.expander("🔍 Debug — Values sent to model (click to verify)", expanded=False):
            non_zero = {k: v for k, v in row.items() if v not in (0, 0.0)}
            st.write("**Non-zero inputs:**", non_zero)
            st.write("**All inputs:**", row)

        try:
            tc  = assets["columns"]; imp = assets["imputer"]; sc = assets["scaler"]
            # Align columns exactly as training, then impute then scale
            df_aligned = df.reindex(columns=tc, fill_value=0)
            df_imp     = pd.DataFrame(imp.transform(df_aligned), columns=tc)
            df_scaled  = pd.DataFrame(sc.transform(df_imp), columns=tc)
            df = df_scaled
        except Exception as e:
            st.warning(f"Preprocessing fallback: {e}")
            df = df.reindex(columns=COLUMN_ORDER, fill_value=0)

        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])
        pct  = round(prob * 100, 1)

        fi = None
        try:
            if hasattr(model, "feature_importances_"):
                raw_imps = model.feature_importances_
            elif hasattr(model, "get_feature_importance"):
                raw_imps = model.get_feature_importance()
            else:
                raw_imps = None
            if raw_imps is not None:
                fn      = (assets.get("columns", COLUMN_ORDER) if assets else COLUMN_ORDER)[:len(raw_imps)]
                top_n   = 12
                top_idx = np.argsort(raw_imps)[::-1][:top_n]
                fi = {"names": [fn[i] for i in top_idx][::-1],
                      "vals":  [float(raw_imps[i]) for i in top_idx][::-1],
                      "mean":  float(np.mean(raw_imps))}
        except Exception:
            fi = None

        new_id = st.session_state.last_prediction_id + 1
        entry  = {
            "id": new_id, "pred": pred, "prob": prob, "pct": pct,
            "feature_importance": fi,
            "input_summary": {KEY_TO_COL[k]: v for k, v in input_values.items()
                              if v not in (0, 0.0) and k in KEY_TO_COL},
            "patient_id": (pat_id.strip() or
                           f"PAT-{datetime.date.today().strftime('%Y%m%d')}-{len(st.session_state.history)+1:03d}"),
            "timestamp": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
            "age": input_values.get("Age", "?"),
        }
        st.session_state.prediction         = entry
        st.session_state.last_prediction_id = new_id
        st.session_state.history.append(entry)
        st.session_state.show_result_overlay = True
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# PAGE · RESULTS  (our version — gradient SHAP waterfall chart)
# ═══════════════════════════════════════════════════════════════════
def page_results():
    p = st.session_state.prediction
    if not p:
        st.warning("No prediction found. Run an assessment first.")
        if st.button("← Go to Classifier"): goto("classifier")
        return

    pred = p["pred"]; pct = p["pct"]
    verdict = "Biopsy Indicated"  if pred == 1 else "Biopsy Unlikely"
    badge_t = "⚠ High Risk"       if pred == 1 else "✓ Low Risk"
    detail  = ("The model predicts a <strong>positive biopsy result</strong>. Warrants further evaluation."
               if pred == 1 else
               "The model predicts a <strong>negative biopsy result</strong>. Continue routine screening.")
    accent  = "var(--rose)"       if pred == 1 else "var(--teal)"
    offset  = round(339 * (1 - pct/100), 1)
    ring_cls = "ring-fill-high"   if pred == 1 else "ring-fill-low"
    bg_grad  = "rgba(255,95,126,0.08)" if pred==1 else "rgba(0,217,160,0.08)"
    bord_c   = "rgba(255,95,126,0.3)"  if pred==1 else "rgba(0,217,160,0.3)"
    top_bg   = "linear-gradient(90deg,var(--rose),#cc1133)" if pred==1 else "linear-gradient(90deg,var(--teal),#22c55e)"
    bar_grad = "linear-gradient(90deg,#ff8fa0,#cc1133)" if pred==1 else "linear-gradient(90deg,#7ee8be,#22c55e)"
    badge_bg = "rgba(255,95,126,0.12)" if pred==1 else "rgba(0,217,160,0.1)"
    badge_bd = "rgba(255,95,126,0.25)" if pred==1 else "rgba(0,217,160,0.2)"

    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(f'<div class="hdr-h au">Prediction Results</div>'
                    f'<p style="font-size:0.78rem;color:var(--txt3);margin-bottom:1.5rem;">'
                    f'{p.get("patient_id","—")} &nbsp;·&nbsp; {p.get("timestamp","—")}</p>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("← Edit", key="res_back"): goto("classifier")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""<div style="background:linear-gradient(135deg,{bg_grad},rgba(4,8,15,0.6));
        border:1.5px solid {bord_c};border-radius:18px;padding:2.25rem;margin-bottom:1.5rem;
        position:relative;overflow:hidden;animation:fadeUp 0.5s ease both;">
      <div style="position:absolute;top:0;left:0;right:0;height:3px;border-radius:18px 18px 0 0;background:{top_bg};"></div>
      <div style="display:grid;grid-template-columns:1fr auto;gap:2rem;align-items:center;">
        <div>
          <div style="display:inline-flex;align-items:center;font-size:0.68rem;font-weight:600;
               letter-spacing:0.1em;text-transform:uppercase;padding:4px 12px;border-radius:20px;
               margin-bottom:0.8rem;font-family:var(--fm);color:{accent};
               background:{badge_bg};border:1px solid {badge_bd};">{badge_t}</div>
          <div style="font-family:var(--fd);font-size:2rem;font-weight:700;line-height:1.15;
               margin-bottom:0.55rem;color:{accent};">{verdict}</div>
          <div style="font-size:0.84rem;color:var(--txt2);line-height:1.75;margin-bottom:1.4rem;">{detail}</div>
          <div style="background:var(--surf3);border-radius:100px;height:9px;overflow:hidden;border:1px solid var(--bord);">
            <div style="width:{pct}%;height:100%;border-radius:100px;background:{bar_grad};"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-family:var(--fm);font-size:0.65rem;color:var(--txt3);margin-top:0.3rem;">
            <span>Predicted biopsy probability</span><span style="font-weight:600;">{pct}%</span>
          </div>
        </div>
        <div style="position:relative;width:130px;height:130px;flex-shrink:0;">
          <svg class="ring-svg" width="130" height="130" viewBox="0 0 120 120">
            <circle class="ring-track" cx="60" cy="60" r="54"/>
            <circle class="{ring_cls}" cx="60" cy="60" r="54" style="stroke-dashoffset:{offset};"/>
          </svg>
          <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">
            <div style="font-family:var(--fd);font-size:2.4rem;font-weight:800;line-height:1;color:{accent};">{pct}%</div>
            <div style="font-family:var(--fm);font-size:0.52rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--txt3);">Risk Score</div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    summary = {k: v for k, v in p.get("input_summary", {}).items() if v not in (0, 0.0)}
    if summary:
        st.markdown('<p class="lbl">Notable Input Values</p>', unsafe_allow_html=True)
        html = '<div class="summary-g">'
        for k, v in list(summary.items())[:12]:
            html += f'<div class="sum-item"><div class="sum-lbl">{k}</div><div class="sum-val">{v}</div></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # ── SHAP gradient waterfall chart ─────────────────────────────────
    fi = p.get("feature_importance")
    if fi and fi.get("vals"):
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            names   = fi["names"]
            vals    = fi["vals"]
            n       = len(names)
            max_val = max(vals) if vals else 1
            norm    = [v / max_val for v in vals]

            LOW_C  = np.array([0.06, 0.35, 0.65])
            HIGH_C = np.array([1.00, 0.37, 0.49])

            fig, ax = plt.subplots(figsize=(11, max(5, n * 0.52 + 1.2)))
            fig.patch.set_facecolor("#0b1220")
            ax.set_facecolor("#0b1220")

            for i, (val, nv) in enumerate(zip(vals, norm)):
                color = tuple(LOW_C + nv * (HIGH_C - LOW_C))
                ax.barh(i, max_val, height=0.58, color="#111d2e", edgecolor="none", zorder=1)
                ax.barh(i, val, height=0.58, color=color, edgecolor="none", zorder=3, alpha=0.92)
                ax.text(val + max_val * 0.018, i, f"{val:.2f}",
                        va="center", ha="left", fontsize=8.5, color="#aec4dc", fontfamily="monospace")
                if val > max_val * 0.18:
                    ax.text(val * 0.5, i, f"{nv*100:.0f}%",
                            va="center", ha="center", fontsize=7.5,
                            color="white", fontfamily="monospace", alpha=0.55)

            ax.set_yticks(range(n))
            ax.set_yticklabels(names, fontsize=9.5, color="#7e97be")
            ax.tick_params(axis="y", length=0, pad=6)
            ax.tick_params(axis="x", colors="#1c2d46", labelsize=8, labelcolor="#3d5270")
            ax.set_xlim(0, max_val * 1.28)
            ax.set_ylim(-0.7, n - 0.3)
            for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#1c2d46")
            ax.xaxis.grid(True, color="#151f30", linewidth=0.7, linestyle="--", zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel("Feature Importance Score", fontsize=9, color="#3d5270",
                          labelpad=10, fontfamily="monospace")
            ax.set_title("Top Contributing Features — Ranked by Predictive Strength",
                         fontsize=11.5, color="#d8e4f5", pad=14, loc="left", fontweight="bold")
            legend_handles = [
                mpatches.Patch(color=tuple(LOW_C),  alpha=0.9, label="Lower importance"),
                mpatches.Patch(color=tuple(HIGH_C), alpha=0.9, label="Highest importance"),
            ]
            ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
                      frameon=True, framealpha=0.85, edgecolor="#1c2d46",
                      facecolor="#0b1220", labelcolor="#7e97be")
            plt.tight_layout(pad=1.6)
            st.markdown('<p class="lbl" style="margin-top:0.5rem;">Feature importance analysis</p>', unsafe_allow_html=True)
            st.caption("Bars show each feature's contribution to the prediction. "
                       "Longer bar = stronger influence. Blue → rose gradient shows relative rank.")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.info(f"Chart unavailable: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("← Edit Profile", key="r_edit"): goto("classifier")
        st.markdown('</div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("🕐 View History", key="r_hist"): goto("history")
        st.markdown('</div>', unsafe_allow_html=True)
    with a3:
        if st.button("🔄 New Assessment", key="r_new"):
            st.session_state.input_values = {}; st.session_state.prediction = None
            st.session_state.show_result_overlay = False; goto("classifier")
    st.markdown("""<div class="disclaimer"><strong>⚕ Clinical Disclaimer:</strong>
    AI prediction only. Not a substitute for professional medical judgment.</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# REMAINING PAGES  (from document-7, team names updated)
# ═══════════════════════════════════════════════════════════════════
def page_history():
    st.markdown("""<div class="page-hdr au"><div class="hdr-ico">🕐</div><div>
      <div class="hdr-h">Assessment History</div>
      <p class="hdr-p">Complete log of all biopsy risk predictions.</p></div></div>""", unsafe_allow_html=True)
    hist = st.session_state.history
    if not hist:
        st.markdown("""<div class="card" style="text-align:center;padding:3rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">📭</div>
          <div style="font-family:var(--fd);font-size:1.1rem;color:var(--txt);margin-bottom:0.5rem;">No assessments yet</div>
          <div style="font-size:0.82rem;color:var(--txt3);">Run your first prediction to see results here.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Run First Assessment", key="hist_cta"): goto("classifier")
        return
    total = len(hist); highs = sum(1 for h in hist if h["pred"]==1); avg = np.mean([h["pct"] for h in hist])
    st.markdown(f"""<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.9rem;margin-bottom:1.5rem;">
      <div class="card-sm"><div class="lbl">Total</div><div style="font-family:var(--fd);font-size:1.8rem;font-weight:700;color:var(--blue);">{total}</div></div>
      <div class="card-sm"><div class="lbl">High Risk</div><div style="font-family:var(--fd);font-size:1.8rem;font-weight:700;color:var(--rose);">{highs}</div></div>
      <div class="card-sm"><div class="lbl">Avg Score</div><div style="font-family:var(--fd);font-size:1.8rem;font-weight:700;color:var(--amber);">{avg:.1f}%</div></div>
    </div>""", unsafe_allow_html=True)
    f1, f2, _ = st.columns([1,1,3])
    with f1: filt = st.selectbox("Filter", ["All","High Risk","Low Risk"], key="hist_filter")
    with f2:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("🗑 Clear", key="hist_clear"): st.session_state.history = []; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    filtered = [h for h in reversed(hist) if filt=="All" or (filt=="High Risk" and h["pred"]==1) or (filt=="Low Risk" and h["pred"]==0)]
    for i, h in enumerate(filtered):
        rc2 = "high" if h["pred"]==1 else "low"
        c1, c2 = st.columns([9,1])
        with c1:
            st.markdown(f"""<div class="hist-item">
              <div class="hist-risk-dot {rc2}"></div>
              <div class="hist-info"><div class="hist-name">{h.get('patient_id','—')}</div>
              <div class="hist-meta">{h.get('timestamp','—')} · Age: {h.get('age','?')}</div></div>
              <div class="hist-badge {rc2}">{"⚠ High" if rc2=='high' else "✓ Low"}</div>
              <div class="hist-score {rc2}">{h['pct']}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            if st.button("View", key=f"hv_{i}"):
                st.session_state.prediction = h; goto("results")


def page_calendar():
    st.markdown("""<div class="page-hdr au"><div class="hdr-ico">📅</div><div>
      <div class="hdr-h">Appointment Calendar</div>
      <p class="hdr-p">Schedule and manage patient appointments.</p></div></div>""", unsafe_allow_html=True)
    today = datetime.date.today()
    if "cal_y" not in st.session_state: st.session_state.cal_y = today.year
    if "cal_m" not in st.session_state: st.session_state.cal_m = today.month
    y, m = st.session_state.cal_y, st.session_state.cal_m
    import calendar
    month_name = datetime.date(y, m, 1).strftime("%B %Y")
    appt_days  = {datetime.datetime.strptime(a["date"],"%Y-%m-%d").date().day
                  for a in st.session_state.appointments
                  if datetime.datetime.strptime(a["date"],"%Y-%m-%d").date().year==y
                  and datetime.datetime.strptime(a["date"],"%Y-%m-%d").date().month==m}
    col_cal, col_form = st.columns([1.3,1])
    with col_cal:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        n1,n2,n3 = st.columns([1,4,1])
        with n1:
            if st.button("‹",key="cp"):
                if m==1: st.session_state.cal_m=12; st.session_state.cal_y-=1
                else: st.session_state.cal_m-=1
                st.rerun()
        with n2: st.markdown(f'<div style="text-align:center;font-family:var(--fd);font-size:1rem;font-weight:700;padding:0.35rem 0;">{month_name}</div>', unsafe_allow_html=True)
        with n3:
            if st.button("›",key="cn"):
                if m==12: st.session_state.cal_m=1; st.session_state.cal_y+=1
                else: st.session_state.cal_m+=1
                st.rerun()
        cal_html='<div class="cal-grid">'
        for dh in ["M","T","W","T","F","S","S"]: cal_html+=f'<div class="cal-dh">{dh}</div>'
        for week in calendar.monthcalendar(y,m):
            for day in week:
                if day==0: cal_html+='<div class="cal-day empty"></div>'
                else:
                    cls="cal-day"
                    if day==today.day and m==today.month and y==today.year: cls+=" today"
                    if day in appt_days: cls+=" has-appt"
                    cal_html+=f'<div class="{cls}">{day}</div>'
        cal_html+='</div>'
        st.markdown(cal_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">➕ New Appointment</div>', unsafe_allow_html=True)
        pn = st.text_input("Patient Name", placeholder="Full name", key="ap_n")
        at = st.selectbox("Type", ["Initial Consultation","Follow-up","Biopsy Review","Screening","Post-Treatment","Emergency"], key="ap_t")
        ad = st.date_input("Date", value=today, key="ap_d")
        ah = st.selectbox("Time", ["08:00","08:30","09:00","09:30","10:00","10:30","11:00","11:30","14:00","14:30","15:00","15:30","16:00","17:00"], key="ap_h")
        an = st.text_input("Notes", placeholder="Optional", key="ap_note")
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("📅 Schedule Appointment", key="ap_add"):
            if pn.strip():
                st.session_state.appointments.append({"patient":pn.strip(),"type":at,"date":ad.strftime("%Y-%m-%d"),"time":ah,"note":an})
                st.success(f"✅ Scheduled for {pn}"); st.rerun()
            else: st.error("Patient name required.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="lbl">Upcoming Appointments</p>', unsafe_allow_html=True)
    upcoming = sorted([a for a in st.session_state.appointments
                       if datetime.datetime.strptime(a["date"],"%Y-%m-%d").date()>=today],
                      key=lambda x:x["date"]+x["time"])
    if not upcoming:
        st.markdown('<div class="card" style="text-align:center;padding:1.5rem;color:var(--txt3);font-size:0.82rem;">No upcoming appointments.</div>', unsafe_allow_html=True)
    else:
        for i, a in enumerate(upcoming):
            d=datetime.datetime.strptime(a["date"],"%Y-%m-%d").date(); dstr=d.strftime("%a, %d %b %Y")
            c1,c2=st.columns([9,1])
            with c1:
                st.markdown(f"""<div class="appt-row"><div class="appt-dot"></div>
                  <div style="flex:1;"><div style="font-size:0.83rem;font-weight:500;color:var(--txt);">{a['patient']}</div>
                  <div style="font-size:0.7rem;color:var(--txt3);font-family:var(--fm);">{a['type']} · {dstr} at {a['time']}</div></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="ghost">', unsafe_allow_html=True)
                if st.button("🗑",key=f"da_{i}"): st.session_state.appointments.remove(a); st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


def page_about():
    st.markdown("""<div class="page-hdr au"><div class="hdr-ico">ℹ️</div><div>
      <div class="hdr-h">About CervAI</div>
      <p class="hdr-p">Mission, methodology, and the team behind the platform.</p></div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="about-hero au">
      <div style="font-family:var(--fd);font-size:2rem;font-weight:800;color:#fff;margin-bottom:0.75rem;position:relative;z-index:1;">
        Advancing Cervical Cancer<br><em style="color:var(--blue);">Early Detection</em> with AI</div>
      <p style="font-size:0.86rem;color:var(--txt2);max-width:580px;line-height:1.85;position:relative;z-index:1;">
        CervAI bridges advanced machine learning and gynecological oncology, built during Coding Week 2026 at Centrale Casablanca.</p></div>""", unsafe_allow_html=True)
    col1,col2=st.columns([1.1,1])
    with col1:
        st.markdown("""<div class="card au2" style="margin-bottom:1rem;">
          <div style="font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🧬 The Dataset</div>
          <p style="font-size:0.82rem;color:var(--txt2);line-height:1.8;">UCI Cervical Cancer (Risk Factors) Dataset — 858 patients, 36 clinical attributes.</p>
          <div class="metric-grid">
            <div class="metric-box"><div class="metric-v">858</div><div class="metric-l">Patients</div></div>
            <div class="metric-box"><div class="metric-v">36</div><div class="metric-l">Features</div></div>
            <div class="metric-box"><div class="metric-v">4</div><div class="metric-l">Target Tests</div></div>
          </div></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="card au3">
          <div style="font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">⚙️ Technology Stack</div>
          <div><span class="tech-pill">🐍 Python 3.11</span><span class="tech-pill">🎈 Streamlit</span>
               <span class="tech-pill">🚀 XGBoost</span><span class="tech-pill">⚖️ SMOTE</span>
               <span class="tech-pill">🔢 NumPy</span><span class="tech-pill">🐼 Pandas</span>
               <span class="tech-pill">🎯 Scikit-learn</span><span class="tech-pill">📊 Matplotlib</span></div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="card au2">
          <div style="font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">👥 Research Team — Coding Week 2026</div>
          <div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#1a3a5c,#0e2240);">👨‍⚕️</div><div><div class="team-name">Bakr Aoulad Omar</div><div class="team-role">Lead Researcher · Clinical Advisor</div></div></div>
          <div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#1a2a50,#0a1830);">👨‍💻</div><div><div class="team-name">Yassir Jbili</div><div class="team-role">ML Engineer · Model Architecture</div></div></div>
          <div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#1a1a40,#0a0a28);">🔬</div><div><div class="team-name">Ilyas El Hadad</div><div class="team-role">Data Scientist · Feature Engineering</div></div></div>
          <div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#102a30,#081820);">📊</div><div><div class="team-name">Mohamed El Haddad</div><div class="team-role">Biostatistics · Validation</div></div></div>
          <div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#2a1a40,#180a28);">🏥</div><div><div class="team-name">Yahya El Omari</div><div class="team-role">Clinical Review · QA</div></div></div>
        </div>""", unsafe_allow_html=True)


def page_contact():
    st.markdown("""<div class="page-hdr au"><div class="hdr-ico">✉️</div><div>
      <div class="hdr-h">Contact Us</div>
      <p class="hdr-p">Reach our team for support or research inquiries.</p></div></div>""", unsafe_allow_html=True)
    col1,col2=st.columns([1.2,1])
    with col1:
        st.markdown('<div class="card au">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1.25rem;">📬 Send Us a Message</div>', unsafe_allow_html=True)
        cn=st.text_input("Full Name",placeholder="Your name",key="ct_n")
        ce=st.text_input("Your Email",placeholder="you@example.com",key="ct_e")
        cto=st.selectbox("Contact",["Bakr Aoulad Omar","Yassir Jbili","Ilyas El Hadad","Mohamed El Haddad","Yahya El Omari"],key="ct_to")
        cs=st.selectbox("Subject",["Technical Support","Clinical Query","Research Collaboration","Bug Report","General Inquiry"],key="ct_s")
        cm=st.text_area("Message",placeholder="Describe your inquiry...",height=130,key="ct_m")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📨  Send Message",key="ct_send"):
            if cn.strip() and ce.strip() and cm.strip(): st.success("✅ Message sent! We'll respond within 48 hours.")
            else: st.error("Please fill in name, email, and message.")
    with col2:
        st.markdown("""<div class="card au2">
          <div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1.1rem;">👥 Team Contacts</div>
          <div class="contact-method"><div class="cm-ico" style="background:rgba(59,158,255,0.1);border:1px solid rgba(59,158,255,0.2);">👨‍⚕️</div><div><div class="cm-lbl">Lead Researcher</div><div class="cm-val">Bakr Aoulad Omar</div><div style="font-size:0.7rem;color:var(--blue);font-family:var(--fm);">bakraouladomor@gmail.com</div></div></div>
          <div class="contact-method"><div class="cm-ico" style="background:rgba(0,217,160,0.1);border:1px solid rgba(0,217,160,0.2);">👨‍💻</div><div><div class="cm-lbl">ML Engineer</div><div class="cm-val">Yassir Jbili</div><div style="font-size:0.7rem;color:var(--blue);font-family:var(--fm);">yassirjbili@gmail.com</div></div></div>
          <div class="contact-method"><div class="cm-ico" style="background:rgba(123,111,255,0.1);border:1px solid rgba(123,111,255,0.2);">🔬</div><div><div class="cm-lbl">Data Scientist</div><div class="cm-val">Ilyas El Hadad</div><div style="font-size:0.7rem;color:var(--blue);font-family:var(--fm);">ilyaselhadad@gmail.com</div></div></div>
          <div class="contact-method"><div class="cm-ico" style="background:rgba(255,183,77,0.1);border:1px solid rgba(255,183,77,0.2);">📊</div><div><div class="cm-lbl">Biostatistics</div><div class="cm-val">Mohamed El Haddad</div><div style="font-size:0.7rem;color:var(--blue);font-family:var(--fm);">mohammedelhadad@gmail.com</div></div></div>
          <div class="contact-method"><div class="cm-ico" style="background:rgba(255,95,126,0.1);border:1px solid rgba(255,95,126,0.2);">🏥</div><div><div class="cm-lbl">Clinical Review</div><div class="cm-val">Yahya El Omari</div><div style="font-size:0.7rem;color:var(--blue);font-family:var(--fm);">yahyaelomari@gmail.com</div></div></div>
        </div>""", unsafe_allow_html=True)


def page_profile():
    u=st.session_state.get("user_info",{})
    st.markdown("""<div class="page-hdr au"><div class="hdr-ico">👤</div><div>
      <div class="hdr-h">My Profile</div>
      <p class="hdr-p">Manage your account information.</p></div></div>""", unsafe_allow_html=True)
    c1,c2=st.columns([1,1.5])
    with c1:
        st.markdown(f"""<div class="card au" style="text-align:center;padding:2rem;">
          <div style="width:80px;height:80px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--purple));
               display:flex;align-items:center;justify-content:center;font-size:1.8rem;font-weight:700;
               color:#fff;margin:0 auto 1rem;border:3px solid var(--bord2);">{u.get('initials','?')}</div>
          <div style="font-family:var(--fd);font-size:1.2rem;font-weight:700;color:var(--txt);">{u.get('name','User')}</div>
          <div style="font-size:0.76rem;color:var(--txt3);font-family:var(--fm);margin-top:0.25rem;">{u.get('role','—')} · {u.get('dept','—')}</div>
          <div style="font-size:0.76rem;color:var(--txt3);margin-top:0.5rem;">{u.get('email','—')}</div>
          <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--bord);display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
            <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.75rem;">
              <div style="font-family:var(--fd);font-size:1.4rem;color:var(--blue);font-weight:700;">{len(st.session_state.history)}</div>
              <div style="font-size:0.65rem;color:var(--txt3);">Assessments</div></div>
            <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.75rem;">
              <div style="font-family:var(--fd);font-size:1.4rem;color:var(--teal);font-weight:700;">{len(st.session_state.appointments)}</div>
              <div style="font-size:0.65rem;color:var(--txt3);">Appointments</div></div>
          </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card au2">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1.1rem;">✏️ Edit Profile</div>', unsafe_allow_html=True)
        new_name=st.text_input("Full Name",value=u.get("name",""),key="prof_name")
        st.text_input("Email",value=u.get("email",""),disabled=True,key="prof_email")
        st.text_input("Department",value=u.get("dept",""),key="prof_dept")
        st.selectbox("Role",["Oncologist","Gynecologist","Radiologist","Researcher","Administrator"],key="prof_role")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Changes",key="prof_save"):
            st.session_state.user_info["name"]=new_name
            st.session_state.user_info["initials"]="".join(w[0].upper() for w in new_name.split()[:2]) if new_name else u.get("initials","?")
            st.success("✅ Profile updated!")
        st.markdown('</div>', unsafe_allow_html=True)


def page_settings():
    st.markdown("""<div class="page-hdr au"><div class="hdr-ico">⚙️</div><div>
      <div class="hdr-h">Settings</div>
      <p class="hdr-p">Configure your platform preferences.</p></div></div>""", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="card au">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🎨 Display</div>', unsafe_allow_html=True)
        st.selectbox("Theme",["Dark (Default)","Dark Blue","Dark Purple"],key="s_theme")
        st.selectbox("Language",["English","Français","العربية","Español"],key="s_lang")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<br><div class="card au3">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🔬 Classifier</div>', unsafe_allow_html=True)
        st.checkbox("Auto-save inputs",value=True,key="s_autosave")
        st.checkbox("Show feature importance",value=True,key="s_fi")
        st.selectbox("Risk threshold",["50% (Standard)","40% (Sensitive)","60% (Specific)"],key="s_thresh")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card au2">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🔔 Notifications</div>', unsafe_allow_html=True)
        st.checkbox("Email alerts for high-risk",value=True,key="s_email")
        st.checkbox("Appointment reminders",value=True,key="s_appt_rem")
        st.checkbox("Weekly summary",value=False,key="s_weekly")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<br><div class="card au4">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🗂 Data Management</div>', unsafe_allow_html=True)
        cc1,cc2=st.columns(2)
        with cc1:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("🗑 Clear History",key="s_clr_h"): st.session_state.history=[]; st.success("Cleared.")
            st.markdown('</div>', unsafe_allow_html=True)
        with cc2:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("🗑 Clear Appointments",key="s_clr_a"): st.session_state.appointments=[]; st.success("Cleared.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 Save All Settings",key="s_save"): st.success("✅ Settings saved!")


# ═══════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    page_login()
else:
    render_sidebar()
    render_topbar()
    pg = st.session_state.page
    if   pg == "home":       page_home()
    elif pg == "classifier": page_classifier()
    elif pg == "results":    page_results()
    elif pg == "history":    page_history()
    elif pg == "calendar":   page_calendar()
    elif pg == "about":      page_about()
    elif pg == "contact":    page_contact()
    elif pg == "profile":    page_profile()
    elif pg == "settings":   page_settings()
    else:                    page_home()
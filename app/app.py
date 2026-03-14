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

# ── Session state ─────────────────────────────────────────────────────────────
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
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Users ─────────────────────────────────────────────────────────────────────
USERS = {
    "dr.martin@cerv.ai":  {"password": "cerv2024", "name": "Dr. Sophie Martin", "role": "Oncologist",    "initials": "SM", "dept": "Oncology"},
    "dr.hassan@cerv.ai":  {"password": "cerv2024", "name": "Dr. Karim Hassan",  "role": "Gynecologist",  "initials": "KH", "dept": "Gynecology"},
    "admin@cerv.ai":      {"password": "admin123", "name": "Admin User",        "role": "Administrator", "initials": "AU", "dept": "IT & Systems"},
}

def check_login(email, password):
    for k, v in USERS.items():
        if k.lower() == email.strip().lower() and v["password"] == password.strip():
            return v
    return None

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    mp_list = ["models/xgboost_model.joblib","xgboost_model.joblib","../models/xgboost_model.joblib"]
    ap_list = ["models/xgboost_assets.joblib","xgboost_assets.joblib","../models/xgboost_assets.joblib"]
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
    "STDs_condylomatosis":"STDs:condylomatosis","STDs_cervical_condylomatosis":"STDs:cervical condylomatosis",
    "STDs_vaginal_condylomatosis":"STDs:vaginal condylomatosis",
    "STDs_vulvo_perineal_condylomatosis":"STDs:vulvo-perineal condylomatosis",
    "STDs_syphilis":"STDs:syphilis","STDs_pelvic_inflammatory_disease":"STDs:pelvic inflammatory disease",
    "STDs_genital_herpes":"STDs:genital herpes","STDs_molluscum_contagiosum":"STDs:molluscum contagiosum",
    "STDs_AIDS":"STDs:AIDS","STDs_HIV":"STDs:HIV","STDs_Hepatitis_B":"STDs:Hepatitis B",
    "STDs_HPV":"STDs:HPV","STDs_Number_of_diagnosis":"STDs: Number of diagnosis",
    "STDs_Time_since_first_diagnosis":"STDs: Time since first diagnosis",
    "STDs_Time_since_last_diagnosis":"STDs: Time since last diagnosis",
    "Dx_Cancer":"Dx:Cancer","Dx_CIN":"Dx:CIN","Dx_HPV":"Dx:HPV","Dx":"Dx",
    "Hinselmann":"Hinselmann","Schiller":"Schiller","Citology":"Citology",
}
FEATURES = {
    "Demographics":{"icon":"👤","desc":"Age, sexual history, pregnancies","fields":[
        ("Age","Age",13,84,25,1,"Patient age in years"),
        ("Sexual partners (lifetime)","Num_sexual_partners",0,28,2,1,"Lifetime number"),
        ("Age at first intercourse","First_sexual_intercourse",10,32,17,1,""),
        ("Number of pregnancies","Num_of_pregnancies",0,11,1,1,""),
    ]},
    "Smoking":{"icon":"🚬","desc":"Tobacco use history","fields":[
        ("Smokes (0=No 1=Yes)","Smokes",0,1,0,1,""),
        ("Smoking duration (years)","Smokes_years",0.0,37.0,0.0,0.5,""),
        ("Smoking intensity (packs/year)","Smokes_packs_year",0.0,40.0,0.0,0.5,""),
    ]},
    "Contraceptives":{"icon":"💊","desc":"Hormonal & IUD use","fields":[
        ("Hormonal contraceptives (0/1)","Hormonal_Contraceptives",0,1,0,1,""),
        ("Hormonal contraceptive use (yrs)","Hormonal_Contraceptives_years",0.0,30.0,0.0,0.5,""),
        ("IUD (0=No 1=Yes)","IUD",0,1,0,1,""),
        ("IUD use (years)","IUD_years",0.0,19.0,0.0,0.5,""),
    ]},
    "STDs":{"icon":"🦠","desc":"STD history","fields":[
        ("Has STD(s) (0/1)","STDs",0,1,0,1,""),("Number of STDs","STDs_number",0,4,0,1,""),
        ("Condylomatosis","STDs_condylomatosis",0,1,0,1,"(0/1)"),
        ("Cervical condylomatosis","STDs_cervical_condylomatosis",0,1,0,1,"(0/1)"),
        ("Vaginal condylomatosis","STDs_vaginal_condylomatosis",0,1,0,1,"(0/1)"),
        ("Vulvo-perineal condylomatosis","STDs_vulvo_perineal_condylomatosis",0,1,0,1,"(0/1)"),
        ("Syphilis","STDs_syphilis",0,1,0,1,"(0/1)"),
        ("Pelvic inflammatory disease","STDs_pelvic_inflammatory_disease",0,1,0,1,"(0/1)"),
        ("Genital herpes","STDs_genital_herpes",0,1,0,1,"(0/1)"),
        ("Molluscum contagiosum","STDs_molluscum_contagiosum",0,1,0,1,"(0/1)"),
        ("AIDS","STDs_AIDS",0,1,0,1,"(0/1)"),("HIV","STDs_HIV",0,1,0,1,"(0/1)"),
        ("Hepatitis B","STDs_Hepatitis_B",0,1,0,1,"(0/1)"),("HPV","STDs_HPV",0,1,0,1,"(0/1)"),
        ("STD diagnoses (count)","STDs_Number_of_diagnosis",0,3,0,1,""),
        ("Years since first STD dx","STDs_Time_since_first_diagnosis",0.0,29.0,0.0,0.5,""),
        ("Years since last STD dx","STDs_Time_since_last_diagnosis",0.0,29.0,0.0,0.5,""),
    ]},
    "Prior Diagnoses & Tests":{"icon":"🔬","desc":"Cancer, CIN, HPV & tests","fields":[
        ("Previous cancer dx (0/1)","Dx_Cancer",0,1,0,1,""),
        ("Previous CIN dx (0/1)","Dx_CIN",0,1,0,1,""),
        ("Previous HPV dx (0/1)","Dx_HPV",0,1,0,1,""),
        ("General diagnosis flag (0/1)","Dx",0,1,0,1,""),
        ("Hinselmann test (0=Neg 1=Pos)","Hinselmann",0,1,0,1,""),
        ("Schiller test (0=Neg 1=Pos)","Schiller",0,1,0,1,""),
        ("Cytology (0=Neg 1=Pos)","Citology",0,1,0,1,""),
    ]},
}

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #04080f;
    --bg2:       #070d1a;
    --surf:      #0b1220;
    --surf2:     #0f1828;
    --surf3:     #141f32;
    --surf4:     #1a2740;
    --bord:      #1c2d46;
    --bord2:     #253a58;
    --bord3:     #2f4870;
    --blue:      #3b9eff;
    --blue2:     #1a7de8;
    --blue-g:    rgba(59,158,255,0.18);
    --blue-s:    rgba(59,158,255,0.08);
    --teal:      #00d9a0;
    --purple:    #7b6fff;
    --rose:      #ff5f7e;
    --amber:     #ffb74d;
    --txt:       #d8e4f5;
    --txt2:      #7e97be;
    --txt3:      #3d5270;
    --r:         14px;
    --r2:        10px;
    --r3:        8px;
    --fd:        'Syne', sans-serif;
    --fb:        'Outfit', sans-serif;
    --fm:        'JetBrains Mono', monospace;
    --shadow:    0 8px 32px rgba(0,0,0,0.45);
    --shadow2:   0 2px 12px rgba(0,0,0,0.3);
}

html,body,[class*="css"]{font-family:var(--fb)!important;background:var(--bg)!important;color:var(--txt)!important;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:var(--bg);}
.main .block-container{padding:1.5rem 2rem 4rem;max-width:1380px;}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--bord2);border-radius:3px;}

/* ── GLOBAL BG MESH ── */
.stApp::before{
    content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
    background:
        radial-gradient(ellipse 1000px 800px at 10% 15%,rgba(59,158,255,0.04) 0%,transparent 60%),
        radial-gradient(ellipse 700px 600px at 90% 85%,rgba(123,111,255,0.04) 0%,transparent 60%),
        radial-gradient(ellipse 600px 500px at 55% 50%,rgba(0,217,160,0.025) 0%,transparent 60%);
}
.stApp::after{
    content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
    background-image:radial-gradient(circle 1px at 1px 1px,rgba(59,158,255,0.055) 1px,transparent 0);
    background-size:52px 52px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{background:var(--surf)!important;border-right:1px solid var(--bord)!important;min-width:260px!important;max-width:260px!important;}
[data-testid="stSidebar"]>div:first-child{padding:1.25rem 1rem 2rem;}

/* ── ALL BUTTONS ── */
.stButton>button{
    font-family:var(--fb)!important;font-weight:600!important;border-radius:var(--r2)!important;
    font-size:0.875rem!important;transition:all 0.2s!important;border:none!important;width:100%;
    background:linear-gradient(135deg,var(--blue),var(--blue2))!important;
    color:#fff!important;padding:0.7rem 1.5rem!important;
    box-shadow:0 4px 18px rgba(59,158,255,0.28)!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(59,158,255,0.45)!important;}
.stButton>button:active{transform:translateY(0)!important;}

.ghost .stButton>button{
    background:var(--surf2)!important;color:var(--txt2)!important;
    border:1px solid var(--bord)!important;box-shadow:none!important;
    padding:0.5rem 1rem!important;font-size:0.78rem!important;width:auto!important;
}
.ghost .stButton>button:hover{background:var(--surf3)!important;color:var(--txt)!important;transform:none!important;box-shadow:none!important;}

.nav-btn .stButton>button{
    background:transparent!important;color:var(--txt2)!important;
    border:none!important;box-shadow:none!important;text-align:left!important;
    padding:0.6rem 0.9rem!important;font-size:0.84rem!important;
    border-radius:var(--r2)!important;width:100%!important;
    display:flex!important;align-items:center!important;
}
.nav-btn .stButton>button:hover{background:var(--surf2)!important;color:var(--txt)!important;transform:none!important;box-shadow:none!important;}
.nav-btn-active .stButton>button{background:var(--blue-g)!important;color:var(--blue)!important;box-shadow:none!important;}
.nav-btn-active .stButton>button:hover{background:var(--blue-g)!important;transform:none!important;}

/* ── INPUTS ── */
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{
    background:var(--surf2)!important;border:1px solid var(--bord)!important;
    color:var(--txt)!important;border-radius:var(--r2)!important;
    font-family:var(--fb)!important;transition:all 0.2s!important;
}
[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{
    border-color:var(--blue)!important;box-shadow:0 0 0 3px var(--blue-g)!important;
    background:var(--surf3)!important;outline:none!important;
}
[data-testid="stNumberInput"] input{
    background:var(--surf2)!important;border:1px solid var(--bord)!important;
    color:var(--txt)!important;border-radius:var(--r2)!important;
    font-family:var(--fm)!important;font-size:0.88rem!important;transition:all 0.2s!important;
}
[data-testid="stNumberInput"] input:focus{border-color:var(--blue)!important;box-shadow:0 0 0 3px var(--blue-g)!important;background:var(--surf3)!important;}
[data-testid="stNumberInput"] button{background:var(--surf3)!important;border:1px solid var(--bord)!important;color:var(--txt2)!important;}
[data-testid="stSelectbox"] *{background:var(--surf2)!important;color:var(--txt)!important;border-color:var(--bord)!important;}
label[data-testid="stWidgetLabel"] p{font-size:0.75rem!important;color:var(--txt2)!important;font-weight:500!important;}

/* ── EXPANDER ── */
[data-testid="stExpander"]{background:var(--surf)!important;border:1px solid var(--bord)!important;border-radius:var(--r)!important;margin-bottom:0.65rem!important;}
[data-testid="stExpander"] summary{background:var(--surf2)!important;border-radius:var(--r)!important;color:var(--txt)!important;font-size:0.84rem!important;font-weight:500!important;padding:0.85rem 1.2rem!important;}
details[open] summary{border-radius:var(--r) var(--r) 0 0!important;}
[data-testid="stExpander"]>div:last-child{background:var(--surf)!important;border-top:1px solid var(--bord)!important;padding:1.2rem!important;}

/* ── ALERTS ── */
.stSuccess>div{background:rgba(0,217,160,0.08)!important;border:1px solid rgba(0,217,160,0.25)!important;border-radius:var(--r2)!important;color:var(--teal)!important;}
.stError>div{background:rgba(255,95,126,0.08)!important;border:1px solid rgba(255,95,126,0.25)!important;border-radius:var(--r2)!important;}
.stInfo>div{background:var(--blue-s)!important;border:1px solid rgba(59,158,255,0.2)!important;border-radius:var(--r2)!important;}
.stWarning>div{background:rgba(255,183,77,0.08)!important;border:1px solid rgba(255,183,77,0.2)!important;border-radius:var(--r2)!important;}

/* ── ANIMATIONS ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
@keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(59,158,255,0.4);}70%{box-shadow:0 0 0 8px transparent;}}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes glow{0%,100%{box-shadow:0 0 20px rgba(59,158,255,0.2);}50%{box-shadow:0 0 40px rgba(59,158,255,0.4);}}
@keyframes ecg{0%{background-position:0 0;}100%{background-position:-1440px 0;}}
@keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-8px);}}

.au{animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) both;}
.au2{animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.08s both;}
.au3{animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.16s both;}
.au4{animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.24s both;}

/* ── CARD ── */
.card{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);padding:1.5rem;}
.card-sm{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);padding:1rem 1.25rem;}
.card-hover{transition:border-color 0.2s,transform 0.2s,box-shadow 0.2s;}
.card-hover:hover{border-color:var(--bord2);transform:translateY(-3px);box-shadow:var(--shadow);}

/* ── TOPBAR ── */
.topbar{
    display:flex;align-items:center;gap:1rem;
    padding:0.8rem 1.4rem;
    background:rgba(11,18,32,0.9);
    backdrop-filter:blur(20px);
    border:1px solid var(--bord);
    border-radius:var(--r);
    margin-bottom:1.75rem;
    position:sticky;top:0.5rem;z-index:99;
    box-shadow:var(--shadow2);
    animation:fadeIn 0.4s ease both;
}
.topbar-logo{font-family:var(--fd);font-size:1.1rem;color:#fff;letter-spacing:-0.02em;white-space:nowrap;}
.topbar-logo em{color:var(--blue);font-style:normal;}
.topbar-sep{width:1px;height:16px;background:var(--bord2);}
.topbar-page{font-family:var(--fm);font-size:0.6rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--txt2);}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:0.6rem;}
.tb-pill{font-family:var(--fm);font-size:0.55rem;letter-spacing:0.12em;text-transform:uppercase;padding:3px 11px;border-radius:20px;background:rgba(59,158,255,0.1);color:var(--blue);border:1px solid rgba(59,158,255,0.2);}
.tb-ico{
    width:32px;height:32px;border-radius:8px;cursor:pointer;
    display:flex;align-items:center;justify-content:center;font-size:0.95rem;
    background:var(--surf2);border:1px solid var(--bord);transition:all 0.2s;
    position:relative;
}
.tb-ico:hover{background:var(--surf3);border-color:var(--bord2);}
.tb-badge{
    position:absolute;top:-4px;right:-4px;width:16px;height:16px;border-radius:50%;
    background:var(--rose);color:#fff;font-size:0.5rem;font-weight:700;
    display:flex;align-items:center;justify-content:center;border:2px solid var(--bg);
}
.tb-avatar{
    width:32px;height:32px;border-radius:50%;
    background:linear-gradient(135deg,var(--blue),var(--purple));
    display:flex;align-items:center;justify-content:center;
    font-size:0.75rem;font-weight:700;color:#fff;
    border:2px solid var(--bord2);cursor:pointer;transition:all 0.2s;
}
.tb-avatar:hover{border-color:var(--blue);}

/* Dropdown menus */
.dropdown{
    background:var(--surf2);border:1px solid var(--bord2);border-radius:var(--r);
    padding:0.5rem;min-width:260px;box-shadow:var(--shadow);
    animation:fadeUp 0.2s ease both;
}
.dropdown-item{
    display:flex;align-items:center;gap:0.75rem;
    padding:0.6rem 0.75rem;border-radius:var(--r3);cursor:pointer;
    font-size:0.82rem;color:var(--txt2);transition:all 0.15s;
}
.dropdown-item:hover{background:var(--surf3);color:var(--txt);}
.dropdown-sep{height:1px;background:var(--bord);margin:0.3rem 0;}
.dropdown-header{font-family:var(--fm);font-size:0.55rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--txt3);padding:0.4rem 0.75rem 0.2rem;}
.notif-unread{background:var(--blue-s);border-left:2px solid var(--blue);}
.notif-dot{width:7px;height:7px;border-radius:50%;background:var(--blue);flex-shrink:0;box-shadow:0 0 6px var(--blue);}

/* ── SIDEBAR COMPONENTS ── */
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

/* ── LOGIN ── */
.login-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:85vh;}
.login-logo{font-family:var(--fd);font-size:2.8rem;color:#fff;text-align:center;letter-spacing:-0.04em;font-weight:800;margin-bottom:0.2rem;}
.login-logo em{color:var(--blue);font-style:normal;}
.login-sub{text-align:center;font-family:var(--fm);font-size:0.6rem;letter-spacing:0.25em;text-transform:uppercase;color:var(--txt3);margin-bottom:2.5rem;}
.login-card{background:var(--surf);border:1px solid var(--bord2);border-radius:20px;padding:2.25rem 2rem;width:100%;max-width:400px;animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.1s both;}
.login-card-title{font-family:var(--fd);font-size:1.1rem;font-weight:700;color:var(--txt);margin-bottom:1.4rem;}
.login-demo{background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.75rem 1rem;margin-top:1.25rem;font-family:var(--fm);font-size:0.7rem;color:var(--txt3);line-height:1.8;}
.login-demo strong{color:var(--txt2);}

/* ── HOME HERO ── */
.hero{
    position:relative;border-radius:22px;overflow:hidden;
    border:1px solid var(--bord2);margin-bottom:1.75rem;
    background:linear-gradient(150deg,#030710 0%,#060f1e 50%,#03090f 100%);
    animation:fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) both;
}
.hero-bg-dots{position:absolute;inset:0;background-image:radial-gradient(circle 1px at 1px 1px,rgba(59,158,255,0.07) 1px,transparent 0);background-size:38px 38px;}
.hero-glow1{position:absolute;top:-150px;left:50%;transform:translateX(-50%);width:800px;height:800px;background:radial-gradient(circle,rgba(59,158,255,0.07) 0%,transparent 65%);border-radius:50%;}
.hero-glow2{position:absolute;bottom:-100px;right:5%;width:400px;height:400px;background:radial-gradient(circle,rgba(0,217,160,0.05) 0%,transparent 65%);border-radius:50%;}
.hero-ecg{position:absolute;bottom:0;left:0;right:0;height:50px;opacity:0.07;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 50'%3E%3Cpolyline points='0,25 200,25 230,5 245,45 260,5 275,45 300,25 500,25 530,5 545,45 560,5 575,45 600,25 800,25 830,5 845,45 860,5 875,45 900,25 1100,25 1130,5 1145,45 1160,5 1175,45 1200,25 1440,25' fill='none' stroke='%2300d9a0' stroke-width='1.5'/%3E%3C/svg%3E") repeat-x;animation:ecg 5s linear infinite;}
.hero-inner{position:relative;z-index:1;display:grid;grid-template-columns:1fr auto;gap:2rem;align-items:center;padding:3.5rem 3.5rem 3rem;}
.hero-eyebrow{font-family:var(--fm);font-size:0.6rem;letter-spacing:0.25em;text-transform:uppercase;color:var(--blue);display:inline-flex;align-items:center;gap:0.5rem;background:rgba(59,158,255,0.08);border:1px solid rgba(59,158,255,0.2);padding:4px 14px;border-radius:20px;margin-bottom:1.2rem;}
.hero-title{font-family:var(--fd);font-size:3.6rem;color:#fff;line-height:1.08;letter-spacing:-0.04em;font-weight:800;margin-bottom:0.6rem;}
.hero-title em{color:var(--blue);font-style:normal;}
.hero-sub{font-size:0.95rem;color:var(--txt2);max-width:520px;line-height:1.85;margin-bottom:2rem;}
.hero-actions{display:flex;gap:0.75rem;flex-wrap:wrap;}
.hero-cta{display:inline-flex;align-items:center;gap:0.5rem;background:linear-gradient(135deg,var(--blue),var(--blue2));color:#fff;padding:0.75rem 1.75rem;border-radius:var(--r2);font-weight:600;font-size:0.88rem;cursor:pointer;border:none;transition:all 0.2s;box-shadow:0 4px 20px rgba(59,158,255,0.35);}
.hero-cta:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(59,158,255,0.5);}
.hero-cta2{display:inline-flex;align-items:center;gap:0.5rem;background:var(--surf2);color:var(--txt2);padding:0.75rem 1.75rem;border-radius:var(--r2);font-weight:500;font-size:0.88rem;cursor:pointer;border:1px solid var(--bord2);transition:all 0.2s;}
.hero-cta2:hover{background:var(--surf3);color:var(--txt);transform:translateY(-2px);}
.hero-img-wrap{position:relative;width:260px;flex-shrink:0;}
.hero-img-wrap img{width:100%;border-radius:16px;opacity:0.9;box-shadow:0 12px 48px rgba(0,0,0,0.6);}
.hero-img-badge{position:absolute;bottom:-12px;left:50%;transform:translateX(-50%);background:var(--surf2);border:1px solid var(--bord2);border-radius:20px;padding:5px 14px;font-size:0.72rem;color:var(--teal);font-family:var(--fm);white-space:nowrap;font-weight:500;}

/* ── STATS ROW ── */
.stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:0.9rem;margin-bottom:1.75rem;}
.stat{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);padding:1.25rem 1.4rem;transition:all 0.2s;position:relative;overflow:hidden;}
.stat::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.stat.blue::before{background:linear-gradient(90deg,var(--blue),transparent);}
.stat.teal::before{background:linear-gradient(90deg,var(--teal),transparent);}
.stat.purple::before{background:linear-gradient(90deg,var(--purple),transparent);}
.stat.amber::before{background:linear-gradient(90deg,var(--amber),transparent);}
.stat:hover{border-color:var(--bord2);transform:translateY(-3px);}
.stat-ico{font-size:1.4rem;margin-bottom:0.6rem;}
.stat-val{font-family:var(--fd);font-size:2rem;font-weight:700;line-height:1;margin-bottom:0.25rem;}
.stat.blue .stat-val{color:var(--blue);}
.stat.teal .stat-val{color:var(--teal);}
.stat.purple .stat-val{color:var(--purple);}
.stat.amber .stat-val{color:var(--amber);}
.stat-lbl{font-size:0.72rem;color:var(--txt3);}

/* ── FEATURES GRID ── */
.feats{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.75rem;}
.feat{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r);padding:1.5rem;cursor:pointer;transition:all 0.2s;position:relative;overflow:hidden;}
.feat::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--blue),var(--purple));transform:scaleX(0);transform-origin:left;transition:transform 0.3s;}
.feat:hover::after{transform:scaleX(1);}
.feat:hover{border-color:var(--bord2);transform:translateY(-4px);box-shadow:var(--shadow);}
.feat-ico{font-size:1.75rem;margin-bottom:0.9rem;}
.feat-title{font-family:var(--fd);font-size:0.9rem;font-weight:700;color:var(--txt);margin-bottom:0.4rem;}
.feat-desc{font-size:0.76rem;color:var(--txt3);line-height:1.7;}
.feat-arrow{position:absolute;top:1.25rem;right:1.25rem;font-size:0.8rem;color:var(--txt3);transition:all 0.2s;}
.feat:hover .feat-arrow{color:var(--blue);transform:translate(2px,-2px);}

/* ── HISTORY ── */
.hist-item{display:flex;align-items:center;gap:1rem;background:var(--surf);border:1px solid var(--bord);border-radius:var(--r2);padding:0.9rem 1.25rem;margin-bottom:0.5rem;transition:all 0.2s;cursor:pointer;}
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

/* ── CALENDAR ── */
.cal-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:3px;margin:0.75rem 0;}
.cal-dh{text-align:center;font-family:var(--fm);font-size:0.58rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--txt3);padding:5px 0;}
.cal-day{aspect-ratio:1;display:flex;align-items:center;justify-content:center;border-radius:var(--r3);font-size:0.78rem;color:var(--txt2);transition:all 0.15s;font-family:var(--fm);cursor:pointer;border:1px solid transparent;}
.cal-day:hover{background:var(--surf3);color:var(--txt);border-color:var(--bord);}
.cal-day.today{background:var(--blue-g);color:var(--blue);border-color:rgba(59,158,255,0.3);font-weight:700;}
.cal-day.has-appt{background:rgba(0,217,160,0.1);color:var(--teal);border-color:rgba(0,217,160,0.25);font-weight:600;}
.cal-day.empty{cursor:default;opacity:0;}
.appt-row{display:flex;align-items:center;gap:0.75rem;background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.7rem 1rem;margin-bottom:0.4rem;transition:all 0.2s;}
.appt-row:hover{border-color:var(--bord2);}
.appt-dot{width:8px;height:8px;border-radius:50%;background:var(--teal);box-shadow:0 0 8px var(--teal);flex-shrink:0;}
.appt-n{font-size:0.83rem;font-weight:500;color:var(--txt);}
.appt-m{font-size:0.7rem;color:var(--txt3);font-family:var(--fm);}

/* ── RESULTS ── */
.res-card{border-radius:18px;padding:2.25rem;border:1.5px solid;position:relative;overflow:hidden;margin-bottom:1.5rem;animation:fadeUp 0.5s ease both;}
.res-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:18px 18px 0 0;}
.res-card.high{background:linear-gradient(135deg,rgba(255,95,126,0.08),rgba(4,8,15,0.6));border-color:rgba(255,95,126,0.3);}
.res-card.high::before{background:linear-gradient(90deg,var(--rose),#cc1133);}
.res-card.low{background:linear-gradient(135deg,rgba(0,217,160,0.08),rgba(4,8,15,0.6));border-color:rgba(0,217,160,0.3);}
.res-card.low::before{background:linear-gradient(90deg,var(--teal),#22c55e);}
.res-inner{display:flex;align-items:flex-start;gap:2rem;flex-wrap:wrap;}
.res-left{flex:1;min-width:240px;}
.res-right{text-align:right;}
.res-badge{display:inline-flex;align-items:center;gap:0.4rem;font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;padding:4px 12px;border-radius:20px;margin-bottom:0.8rem;font-family:var(--fm);}
.res-card.high .res-badge{background:rgba(255,95,126,0.12);color:var(--rose);border:1px solid rgba(255,95,126,0.25);}
.res-card.low  .res-badge{background:rgba(0,217,160,0.1);color:var(--teal);border:1px solid rgba(0,217,160,0.2);}
.res-verdict{font-family:var(--fd);font-size:2rem;font-weight:700;line-height:1.15;margin-bottom:0.55rem;}
.res-card.high .res-verdict{color:var(--rose);}
.res-card.low  .res-verdict{color:var(--teal);}
.res-detail{font-size:0.84rem;color:var(--txt2);line-height:1.75;margin-bottom:1.4rem;}
.prob-big{font-family:var(--fd);font-size:4.5rem;line-height:1;font-weight:800;}
.res-card.high .prob-big{color:rgba(255,95,126,0.75);}
.res-card.low  .prob-big{color:rgba(0,217,160,0.75);}
.prob-lbl{font-family:var(--fm);font-size:0.58rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--txt3);margin-top:0.25rem;}
.prob-track{background:var(--surf3);border-radius:100px;height:9px;overflow:hidden;margin:0.5rem 0 0.3rem;border:1px solid var(--bord);}
.prob-fill{height:100%;border-radius:100px;}
.prob-fill.high{background:linear-gradient(90deg,#ff8fa0,#cc1133);box-shadow:0 0 10px rgba(255,95,126,0.5);}
.prob-fill.low{background:linear-gradient(90deg,#7ee8be,#22c55e);box-shadow:0 0 10px rgba(34,197,94,0.5);}
.prob-meta{display:flex;justify-content:space-between;font-family:var(--fm);font-size:0.65rem;color:var(--txt3);}

/* ── ABOUT ── */
.about-hero{background:linear-gradient(135deg,#06101e,#0a1628);border:1px solid var(--bord2);border-radius:var(--r);padding:2.5rem;margin-bottom:1.25rem;position:relative;overflow:hidden;}
.about-hero::after{content:'';position:absolute;top:0;right:0;bottom:0;width:40%;background:radial-gradient(ellipse at right center,rgba(59,158,255,0.06),transparent 70%);}
.team-card{display:flex;align-items:center;gap:0.9rem;background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.9rem 1.1rem;margin-bottom:0.6rem;transition:all 0.2s;}
.team-card:hover{border-color:var(--bord2);transform:translateX(4px);}
.team-av{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;}
.team-name{font-size:0.84rem;font-weight:600;color:var(--txt);}
.team-role{font-size:0.7rem;color:var(--txt3);font-family:var(--fm);}
.tech-pill{display:inline-flex;align-items:center;gap:0.3rem;background:var(--surf3);border:1px solid var(--bord);border-radius:6px;padding:4px 10px;font-size:0.73rem;color:var(--txt2);margin:0.2rem;transition:all 0.2s;cursor:default;}
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

/* ── CONTACT ── */
.contact-method{display:flex;align-items:center;gap:0.9rem;background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.9rem 1.1rem;margin-bottom:0.55rem;transition:all 0.2s;cursor:pointer;}
.contact-method:hover{border-color:var(--bord2);transform:translateX(4px);}
.cm-ico{width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;}
.cm-lbl{font-size:0.68rem;color:var(--txt3);margin-bottom:0.1rem;}
.cm-val{font-size:0.83rem;font-weight:500;color:var(--txt);}
.faq-item{background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.9rem 1.1rem;margin-bottom:0.4rem;cursor:pointer;transition:all 0.2s;}
.faq-item:hover{border-color:var(--bord2);}
.faq-item.open{border-color:var(--blue);background:var(--blue-s);}
.faq-q{font-size:0.84rem;font-weight:600;color:var(--txt);display:flex;justify-content:space-between;align-items:center;}
.faq-a{font-size:0.78rem;color:var(--txt2);line-height:1.65;margin-top:0.6rem;padding-top:0.6rem;border-top:1px solid var(--bord);}

/* ── PROGRESS BAR ── */
.prog-wrap{background:var(--surf);border:1px solid var(--bord);border-radius:var(--r2);padding:0.75rem 1.25rem;margin-bottom:1.25rem;display:flex;align-items:center;gap:1rem;}
.prog-lbl{font-family:var(--fm);font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--txt3);white-space:nowrap;}
.prog-bar{flex:1;background:var(--surf3);border-radius:100px;height:6px;}
.prog-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--blue),var(--purple));transition:width 0.4s ease;}
.prog-pct{font-family:var(--fm);font-size:0.7rem;color:var(--blue);white-space:nowrap;font-weight:500;}

/* ── MISC ── */
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
.sum-item{background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.65rem 0.9rem;transition:border-color 0.2s;}
.sum-item:hover{border-color:var(--bord2);}
.sum-lbl{font-size:0.65rem;color:var(--txt3);margin-bottom:0.15rem;}
.sum-val{font-family:var(--fm);font-size:0.84rem;color:var(--txt);font-weight:500;}
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
# SIDEBAR  (fully interactive)
# ═══════════════════════════════════════════════════════════════════
def render_sidebar():
    u = st.session_state.get("user_info", {})
    pg = st.session_state.page
    with st.sidebar:
        st.markdown(f'<div class="sb-logo">Cerv<em>AI</em></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sb-user">
          <div class="sb-avatar">{u.get('initials','?')}</div>
          <div>
            <div class="sb-name">{u.get('name','User')}</div>
            <div class="sb-role">{u.get('role','Clinician')}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
        NAV = [
            ("home",       "🏠", "Dashboard"),
            ("classifier", "📋", "Patient Classifier"),
            ("history",    "🕐", "History"),
            ("calendar",   "📅", "Calendar"),
        ]
        for pid, ico, lbl in NAV:
            cls = "nav-btn-active" if pg == pid else "nav-btn"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(f"{ico}  {lbl}", key=f"sb_{pid}"):
                goto(pid)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Info</div>', unsafe_allow_html=True)
        for pid, ico, lbl in [("about","ℹ️","About"),("contact","✉️","Contact Us")]:
            cls = "nav-btn-active" if pg == pid else "nav-btn"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(f"{ico}  {lbl}", key=f"sb_{pid}"):
                goto(pid)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Account</div>', unsafe_allow_html=True)
        cls2 = "nav-btn-active" if pg == "profile" else "nav-btn"
        st.markdown(f'<div class="{cls2}">', unsafe_allow_html=True)
        if st.button("👤  My Profile", key="sb_profile"):
            goto("profile")
        st.markdown('</div>', unsafe_allow_html=True)

        cls3 = "nav-btn-active" if pg == "settings" else "nav-btn"
        st.markdown(f'<div class="{cls3}">', unsafe_allow_html=True)
        if st.button("⚙️  Settings", key="sb_settings"):
            goto("settings")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sb-status">
          <div class="sb-dot"></div>
          {"Model Active" if model else "Demo Mode · No Model"}
        </div>""", unsafe_allow_html=True)
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

        st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
        if st.button("🚪  Sign Out", key="sb_logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.session_state.user_info = {}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TOPBAR  (interactive icons)
# ═══════════════════════════════════════════════════════════════════
def render_topbar():
    pg = st.session_state.page
    _, lbl = {"home":"🏠 Dashboard","classifier":"📋 Classifier","history":"🕐 History",
               "calendar":"📅 Calendar","about":"ℹ️ About","contact":"✉️ Contact Us",
               "profile":"👤 Profile","settings":"⚙️ Settings","results":"📊 Results"}.get(pg,("","")).split(" ",1) if " " in {"home":"🏠 Dashboard","classifier":"📋 Classifier","history":"🕐 History","calendar":"📅 Calendar","about":"ℹ️ About","contact":"✉️ Contact Us","profile":"👤 Profile","settings":"⚙️ Settings","results":"📊 Results"}.get(pg,"") else ("","Page")
    ico_lbl = {"home":"🏠 Dashboard","classifier":"📋 Classifier","history":"🕐 History",
               "calendar":"📅 Calendar","about":"ℹ️ About","contact":"✉️ Contact Us",
               "profile":"👤 Profile","settings":"⚙️ Settings","results":"📊 Results"}.get(pg,"🔬 Page")

    u = st.session_state.get("user_info", {})
    nc = unread_count()

    c_left, c_notif, c_search, c_prof = st.columns([6, 0.7, 0.7, 0.7])

    with c_left:
        st.markdown(f"""
        <div class="topbar">
          <div class="topbar-logo">Cerv<em>AI</em></div>
          <div class="topbar-sep"></div>
          <div class="topbar-page">{ico_lbl}</div>
          <div class="topbar-right">
            <div class="tb-pill">Research Use Only</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c_notif:
        badge = f'<div class="tb-badge">{nc}</div>' if nc else ""
        st.markdown(f'<div class="tb-ico" style="margin-top:0.3rem;">{badge}🔔</div>', unsafe_allow_html=True)
        if st.button("🔔", key="tb_notif", help="Notifications"):
            st.session_state.show_notif = not st.session_state.show_notif
            st.session_state.show_profile_menu = False
            st.rerun()

    with c_search:
        st.markdown('<div class="tb-ico" style="margin-top:0.3rem;">🔍</div>', unsafe_allow_html=True)
        if st.button("🔍", key="tb_search", help="Quick Search"):
            goto("classifier")

    with c_prof:
        st.markdown(f'<div class="tb-avatar" style="margin-top:0.3rem;">{u.get("initials","?")}</div>', unsafe_allow_html=True)
        if st.button(u.get("initials","?"), key="tb_avatar", help="Profile & Settings"):
            st.session_state.show_profile_menu = not st.session_state.show_profile_menu
            st.session_state.show_notif = False
            st.rerun()

    # ── Notification dropdown ──
    if st.session_state.show_notif:
        st.markdown('<div class="dropdown" style="max-width:320px;">', unsafe_allow_html=True)
        st.markdown('<div class="dropdown-header">Notifications</div>', unsafe_allow_html=True)
        for n in st.session_state.notifications:
            cls = "notif-unread" if not n["read"] else ""
            dot = '<div class="notif-dot"></div>' if not n["read"] else '<div style="width:7px"></div>'
            st.markdown(f"""
            <div class="dropdown-item {cls}">
              {dot}
              <div style="flex:1;">
                <div style="font-size:0.8rem;color:var(--txt);">{n['icon']} {n['text']}</div>
                <div style="font-size:0.65rem;color:var(--txt3);font-family:var(--fm);">{n['time']}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("✓ Mark all read", key="mark_read"):
            for n in st.session_state.notifications:
                n["read"] = True
            st.session_state.show_notif = False
            st.rerun()

    # ── Profile dropdown ──
    if st.session_state.show_profile_menu:
        u = st.session_state.get("user_info", {})
        st.markdown(f"""
        <div class="dropdown">
          <div style="display:flex;align-items:center;gap:0.75rem;padding:0.75rem;
                      background:var(--surf3);border-radius:var(--r3);margin-bottom:0.35rem;">
            <div class="sb-avatar" style="width:36px;height:36px;font-size:0.85rem;">{u.get('initials','?')}</div>
            <div>
              <div style="font-size:0.82rem;font-weight:600;color:var(--txt);">{u.get('name','User')}</div>
              <div style="font-size:0.68rem;color:var(--txt3);">{u.get('email','')}</div>
            </div>
          </div>
          <div class="dropdown-sep"></div>
        </div>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("👤 Profile", key="pm_profile"):
                goto("profile")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("⚙️ Settings", key="pm_settings"):
                goto("settings")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="pm_logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.session_state.user_info = {}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE · LOGIN
# ═══════════════════════════════════════════════════════════════════
def page_login():
    _, cc, _ = st.columns([1, 1.5, 1])
    with cc:
        st.markdown("""
        <div class="login-wrap">
          <div class="login-logo">Cerv<em>AI</em></div>
          <div class="login-sub">Clinical Cervical Cancer Risk Platform</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-card-title">🔐 Sign In to Your Account</div>', unsafe_allow_html=True)
        email = st.text_input("Email Address", placeholder="doctor@cerv.ai", key="li_email")
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
                st.error("❌ Invalid credentials.")
        st.markdown("""
        <div class="login-demo">
          <strong>Demo accounts:</strong><br>
          admin@cerv.ai / admin123<br>
          dr.martin@cerv.ai / cerv2024<br>
          dr.hassan@cerv.ai / cerv2024
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE · HOME / DASHBOARD
# ═══════════════════════════════════════════════════════════════════
def page_home():
    u = st.session_state.get("user_info", {})

    # ── Hero ──
    st.markdown(f"""
    <div class="hero au">
      <div class="hero-bg-dots"></div>
      <div class="hero-glow1"></div>
      <div class="hero-glow2"></div>
      <div class="hero-ecg"></div>
      <div class="hero-inner">
        <div>
          <div class="hero-eyebrow">⬤ AI-Powered · Clinical Grade</div>
          <div class="hero-title">Cervical Cancer<br>Risk <em>Classifier</em></div>
          <div class="hero-sub">
            Welcome back, <strong style="color:var(--txt);">{u.get('name','Doctor')}</strong>.<br>
            Evidence-informed biopsy risk prediction trained on 858 clinical records across 36 risk factors.
          </div>
        </div>
        <div class="hero-img-wrap">
          <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Papilloma_Virus_%28HPV%29_EM.jpg/480px-Papilloma_Virus_%28HPV%29_EM.jpg"
               alt="HPV Virus — Cervical Cancer" title="Human Papillomavirus (HPV) — primary cause of cervical cancer"/>
          <div class="hero-img-badge">🔬 HPV — Primary Causal Agent</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # CTA buttons
    c1, c2, c3, _ = st.columns([1.2, 1.2, 1, 3])
    with c1:
        if st.button("🔬 Run Assessment", key="h_assess"):
            goto("classifier")
    with c2:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("🕐 View History", key="h_hist"):
            goto("history")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("📅 Calendar", key="h_cal"):
            goto("calendar")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stats ──
    total_hist = len(st.session_state.history)
    high_risk  = sum(1 for h in st.session_state.history if h["pred"] == 1)
    upcoming   = sum(1 for a in st.session_state.appointments
                     if datetime.datetime.strptime(a["date"],"%Y-%m-%d").date() >= datetime.date.today())
    st.markdown(f"""
    <div class="stats-row au2">
      <div class="stat blue"><div class="stat-ico">🔬</div><div class="stat-val">{total_hist}</div><div class="stat-lbl">Assessments Run</div></div>
      <div class="stat teal"><div class="stat-ico">✅</div><div class="stat-val">{total_hist - high_risk}</div><div class="stat-lbl">Low Risk Results</div></div>
      <div class="stat purple"><div class="stat-ico">⚠️</div><div class="stat-val">{high_risk}</div><div class="stat-lbl">High Risk Flagged</div></div>
      <div class="stat amber"><div class="stat-ico">📅</div><div class="stat-val">{upcoming}</div><div class="stat-lbl">Upcoming Appointments</div></div>
    </div>""", unsafe_allow_html=True)

    # ── Feature cards ──
    st.markdown("""
    <div class="feats au3">
      <div class="feat"><div class="feat-arrow">↗</div><div class="feat-ico">📋</div><div class="feat-title">Comprehensive Profiling</div><div class="feat-desc">5 clinical categories, 36 risk factors covering demographics, STDs, contraceptives and prior diagnoses.</div></div>
      <div class="feat"><div class="feat-arrow">↗</div><div class="feat-ico">⚡</div><div class="feat-title">Instant Prediction</div><div class="feat-desc">XGBoost + SMOTE oversampling, 80/20 stratified split. Sub-second inference for any patient profile.</div></div>
      <div class="feat"><div class="feat-arrow">↗</div><div class="feat-ico">📊</div><div class="feat-title">Explainable AI</div><div class="feat-desc">Top feature importance chart reveals which clinical factors are driving each individual prediction.</div></div>
      <div class="feat"><div class="feat-arrow">↗</div><div class="feat-ico">🕐</div><div class="feat-title">Assessment History</div><div class="feat-desc">Every prediction is logged with full patient data, timestamps, and risk scores for audit trails.</div></div>
      <div class="feat"><div class="feat-arrow">↗</div><div class="feat-ico">📅</div><div class="feat-title">Appointment Calendar</div><div class="feat-desc">Schedule follow-ups, biopsies, and consultations directly within the platform.</div></div>
      <div class="feat"><div class="feat-arrow">↗</div><div class="feat-ico">🔒</div><div class="feat-title">Secure & Private</div><div class="feat-desc">No patient data stored beyond the session. All processing in-memory, no external transmissions.</div></div>
    </div>""", unsafe_allow_html=True)

    # ── Cervical cancer info ──
    st.markdown('<div class="lbl au4">Cervical Cancer · Key Facts</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("""
        <div class="card au4" style="margin-bottom:0;">
          <div style="font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🧬 About Cervical Cancer</div>
          <p style="font-size:0.82rem;color:var(--txt2);line-height:1.85;margin-bottom:0.75rem;">
            Cervical cancer is the <strong style="color:var(--txt);">4th most common cancer</strong> in women worldwide,
            with approximately 600,000 new cases and 342,000 deaths annually (WHO, 2020).
            Over <strong style="color:var(--txt);">99% of cases</strong> are linked to persistent infection with
            <em>high-risk Human Papillomavirus (HPV)</em>.
          </p>
          <p style="font-size:0.82rem;color:var(--txt2);line-height:1.85;margin-bottom:0.75rem;">
            Early detection through regular Pap smears, HPV testing, and colposcopy can reduce mortality
            by up to <strong style="color:var(--teal);">80%</strong>. Risk factors include early sexual activity,
            multiple partners, smoking, long-term oral contraceptive use, and immunosuppression.
          </p>
          <div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin-top:1rem;">
            <div style="background:rgba(255,95,126,0.1);border:1px solid rgba(255,95,126,0.2);border-radius:8px;padding:0.7rem 1rem;flex:1;min-width:100px;text-align:center;">
              <div style="font-family:var(--fd);font-size:1.6rem;color:var(--rose);font-weight:700;">604k</div>
              <div style="font-size:0.66rem;color:var(--txt3);">New cases / year</div>
            </div>
            <div style="background:rgba(255,183,77,0.08);border:1px solid rgba(255,183,77,0.2);border-radius:8px;padding:0.7rem 1rem;flex:1;min-width:100px;text-align:center;">
              <div style="font-family:var(--fd);font-size:1.6rem;color:var(--amber);font-weight:700;">342k</div>
              <div style="font-size:0.66rem;color:var(--txt3);">Deaths per year</div>
            </div>
            <div style="background:rgba(0,217,160,0.08);border:1px solid rgba(0,217,160,0.2);border-radius:8px;padding:0.7rem 1rem;flex:1;min-width:100px;text-align:center;">
              <div style="font-family:var(--fd);font-size:1.6rem;color:var(--teal);font-weight:700;">80%</div>
              <div style="font-size:0.66rem;color:var(--txt3);">Preventable with screening</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="margin-bottom:0;">
          <div style="font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🩺 Clinical Screening Guidelines</div>
          <div style="display:flex;flex-direction:column;gap:0.6rem;">
            <div style="display:flex;gap:0.8rem;align-items:flex-start;">
              <div style="width:28px;height:28px;border-radius:8px;background:rgba(59,158,255,0.1);border:1px solid rgba(59,158,255,0.2);display:flex;align-items:center;justify-content:center;font-size:0.85rem;flex-shrink:0;">1</div>
              <div><div style="font-size:0.82rem;font-weight:600;color:var(--txt);">Ages 21–29</div><div style="font-size:0.76rem;color:var(--txt2);">Pap smear every 3 years. HPV testing not recommended alone.</div></div>
            </div>
            <div style="display:flex;gap:0.8rem;align-items:flex-start;">
              <div style="width:28px;height:28px;border-radius:8px;background:rgba(59,158,255,0.1);border:1px solid rgba(59,158,255,0.2);display:flex;align-items:center;justify-content:center;font-size:0.85rem;flex-shrink:0;">2</div>
              <div><div style="font-size:0.82rem;font-weight:600;color:var(--txt);">Ages 30–65</div><div style="font-size:0.76rem;color:var(--txt2);">Pap smear + HPV co-test every 5 years, or Pap alone every 3 years.</div></div>
            </div>
            <div style="display:flex;gap:0.8rem;align-items:flex-start;">
              <div style="width:28px;height:28px;border-radius:8px;background:rgba(0,217,160,0.1);border:1px solid rgba(0,217,160,0.2);display:flex;align-items:center;justify-content:center;font-size:0.85rem;flex-shrink:0;">✓</div>
              <div><div style="font-size:0.82rem;font-weight:600;color:var(--txt);">HPV Vaccination</div><div style="font-size:0.76rem;color:var(--txt2);">Recommended for ages 9–26. Protects against HPV strains 16 & 18 (70% of cervical cancers).</div></div>
            </div>
            <div style="display:flex;gap:0.8rem;align-items:flex-start;">
              <div style="width:28px;height:28px;border-radius:8px;background:rgba(255,95,126,0.1);border:1px solid rgba(255,95,126,0.2);display:flex;align-items:center;justify-content:center;font-size:0.85rem;flex-shrink:0;">⚠</div>
              <div><div style="font-size:0.82rem;font-weight:600;color:var(--txt);">High-Risk Patients</div><div style="font-size:0.76rem;color:var(--txt2);">HIV+, immunosuppressed, or DES exposure: annual screening regardless of age.</div></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
    <strong>⚕ Research & Educational Use Only.</strong> CervAI is a machine learning prototype trained on the UCI Cervical Cancer dataset.
    It is <em>not a certified medical device</em> and must not substitute for clinical judgment.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE · HISTORY
# ═══════════════════════════════════════════════════════════════════
def page_history():
    st.markdown("""
    <div class="page-hdr au">
      <div class="hdr-ico">🕐</div>
      <div><div class="hdr-h">Assessment History</div>
      <p class="hdr-p">Complete log of all biopsy risk predictions with patient data, scores, and timestamps.</p></div>
    </div>""", unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">📭</div>
          <div style="font-family:var(--fd);font-size:1.1rem;color:var(--txt);margin-bottom:0.5rem;">No assessments yet</div>
          <div style="font-size:0.82rem;color:var(--txt3);">Run your first biopsy risk prediction to see results here.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Run First Assessment", key="hist_cta"):
            goto("classifier")
        return

    # Summary stats
    total = len(hist)
    highs = sum(1 for h in hist if h["pred"] == 1)
    avg   = np.mean([h["pct"] for h in hist])
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.9rem;margin-bottom:1.5rem;">
      <div class="card-sm"><div class="lbl">Total Assessments</div><div style="font-family:var(--fd);font-size:1.8rem;font-weight:700;color:var(--blue);">{total}</div></div>
      <div class="card-sm"><div class="lbl">High Risk</div><div style="font-family:var(--fd);font-size:1.8rem;font-weight:700;color:var(--rose);">{highs}</div></div>
      <div class="card-sm"><div class="lbl">Avg Risk Score</div><div style="font-family:var(--fd);font-size:1.8rem;font-weight:700;color:var(--amber);">{avg:.1f}%</div></div>
    </div>""", unsafe_allow_html=True)

    # Filter
    f1, f2, _ = st.columns([1, 1, 3])
    with f1:
        filt = st.selectbox("Filter", ["All", "High Risk", "Low Risk"], key="hist_filter")
    with f2:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("🗑 Clear History", key="hist_clear"):
            st.session_state.history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    filtered = [h for h in reversed(hist) if filt == "All" or (filt == "High Risk" and h["pred"] == 1) or (filt == "Low Risk" and h["pred"] == 0)]

    for i, h in enumerate(filtered):
        rc = "high" if h["pred"] == 1 else "low"
        c1, c2 = st.columns([9, 1])
        with c1:
            st.markdown(f"""
            <div class="hist-item">
              <div class="hist-risk-dot {rc}"></div>
              <div class="hist-info">
                <div class="hist-name">{h.get('patient_id','Patient #'+str(len(hist)-filtered.index(h)))}</div>
                <div class="hist-meta">{h.get('timestamp','—')} &nbsp;·&nbsp; Age: {h.get('age','?')} &nbsp;·&nbsp; Model: XGBoost</div>
              </div>
              <div class="hist-badge {rc}">{"⚠ High Risk" if rc=='high' else "✓ Low Risk"}</div>
              <div class="hist-score {rc}">{h['pct']}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            if st.button("View", key=f"hist_view_{i}"):
                st.session_state.prediction = h
                goto("results")


# ═══════════════════════════════════════════════════════════════════
# PAGE · CLASSIFIER
# ═══════════════════════════════════════════════════════════════════
def page_classifier():
    st.markdown("""
    <div class="page-hdr au">
      <div class="hdr-ico">📋</div>
      <div><div class="hdr-h">Patient Risk Profile</div>
      <p class="hdr-p">Complete all relevant sections, then click <strong style="color:var(--txt);">Run Biopsy Risk Prediction</strong>.</p></div>
    </div>""", unsafe_allow_html=True)

    # Patient ID
    pat_id = st.text_input("Patient ID / Reference", placeholder="e.g. PAT-2024-0042", key="pat_id_field")

    # Progress
    total_f = sum(len(d["fields"]) for d in FEATURES.values())
    filled  = sum(1 for k, v in st.session_state.input_values.items() if v not in (0, 0.0))
    pct_p   = int(filled / max(total_f, 1) * 100)
    st.markdown(f"""
    <div class="prog-wrap">
      <div class="prog-lbl">Profile Completion</div>
      <div class="prog-bar"><div class="prog-fill" style="width:{pct_p}%;"></div></div>
      <div class="prog-pct">{pct_p}%</div>
    </div>""", unsafe_allow_html=True)

    if model is None:
        st.markdown('<div class="notice err"><span>⚠️</span><div><strong>Model not loaded.</strong> Place <code>xgboost_model.joblib</code> + <code>xgboost_assets.joblib</code> in <code>models/</code>.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="notice"><span>✅</span><div>Model ready &nbsp;·&nbsp; <code>{model_path}</code></div></div>', unsafe_allow_html=True)

    input_values = {}
    col1, col2 = st.columns(2, gap="medium")

    def render_section(name, data, container):
        with container:
            with st.expander(f"{data['icon']}  {name}  ·  {data['desc']}", expanded=(name in ["Demographics","Smoking"])):
                fields = data["fields"]
                n = 2 if len(fields) > 2 else len(fields)
                cols = st.columns(n)
                for i, (label, key, mn, mx, default, step, help_text) in enumerate(fields):
                    saved = st.session_state.input_values.get(key, default)
                    with cols[i % n]:
                        if isinstance(step, int):
                            input_values[key] = st.number_input(label, min_value=int(mn), max_value=int(mx), value=int(saved), step=step, help=help_text or None, key=f"f_{key}")
                        else:
                            input_values[key] = st.number_input(label, min_value=float(mn), max_value=float(mx), value=float(saved), step=step, help=help_text or None, key=f"f_{key}")

    with col1:
        st.markdown('<p class="lbl">Patient Characteristics</p>', unsafe_allow_html=True)
        for n in ["Demographics","Contraceptives","Prior Diagnoses & Tests"]:
            render_section(n, FEATURES[n], col1)
    with col2:
        st.markdown('<p class="lbl">Risk Factors & History</p>', unsafe_allow_html=True)
        for n in ["Smoking","STDs"]:
            render_section(n, FEATURES[n], col2)

    st.markdown("<br>", unsafe_allow_html=True)
    _, b2, _ = st.columns([1, 2, 1])
    with b2:
        if st.button("🔬  Run Biopsy Risk Prediction", disabled=(model is None), key="predict_btn"):
            st.session_state.input_values = dict(input_values)
            row = {KEY_TO_COL[k]: v for k, v in input_values.items()}
            df  = pd.DataFrame([row])
            try:
                tc = assets["columns"]; imp = assets["imputer"]; sc = assets["scaler"]
                df = pd.DataFrame(imp.transform(df.reindex(columns=tc, fill_value=0)), columns=tc)
                df = pd.DataFrame(sc.transform(df), columns=tc)
            except:
                df = df.reindex(columns=COLUMN_ORDER, fill_value=0)

            pred = int(model.predict(df)[0])
            prob = float(model.predict_proba(df)[0][1])
            pct  = round(prob * 100, 1)

            fi = None
            if hasattr(model, "feature_importances_"):
                try:
                    imps = model.feature_importances_
                    fn   = (assets.get("columns", COLUMN_ORDER) if assets else COLUMN_ORDER)[:len(imps)]
                    ti   = np.argsort(imps)[::-1][:12]
                    fi   = {"names": [fn[i] for i in ti][::-1], "vals": [float(imps[i]) for i in ti][::-1]}
                except: pass

            entry = {
                "pred": pred, "prob": prob, "pct": pct,
                "feature_importance": fi,
                "input_summary": {KEY_TO_COL[k]: v for k, v in input_values.items() if v not in (0, 0.0) and k in KEY_TO_COL},
                "patient_id": pat_id.strip() or f"PAT-{datetime.date.today().strftime('%Y%m%d')}-{len(st.session_state.history)+1:03d}",
                "timestamp": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
                "age": input_values.get("Age", "?"),
            }
            st.session_state.prediction = entry
            st.session_state.history.append(entry)
            goto("results")


# ═══════════════════════════════════════════════════════════════════
# PAGE · RESULTS
# ═══════════════════════════════════════════════════════════════════
def page_results():
    p = st.session_state.prediction
    if not p:
        st.warning("No prediction. Please complete the patient profile first.")
        if st.button("← Go to Classifier"): goto("classifier")
        return

    pred = p["pred"]; pct = p["pct"]
    rc   = "high" if pred == 1 else "low"
    verdict   = "Biopsy Indicated" if pred == 1 else "Biopsy Unlikely"
    badge_txt = "⚠ High Risk"       if pred == 1 else "✓ Low Risk"
    detail    = ("The model predicts a <strong>positive biopsy result</strong>. This patient's risk profile warrants further clinical evaluation and immediate specialist referral." if pred == 1
                 else "The model predicts a <strong>negative biopsy result</strong>. Continue routine cervical screening as per current clinical guidelines.")

    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(f'<div class="hdr-h au">Prediction Results</div><p style="font-size:0.78rem;color:var(--txt3);margin-bottom:1.5rem;">{p.get("patient_id","—")} &nbsp;·&nbsp; {p.get("timestamp","—")} &nbsp;·&nbsp; XGBoost</p>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("← Edit", key="res_back"): goto("classifier")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="res-card {rc}">
      <div class="res-inner">
        <div class="res-left">
          <div class="res-badge">{badge_txt}</div>
          <div class="res-verdict">{verdict}</div>
          <div class="res-detail">{detail}</div>
          <div>
            <div class="prob-track"><div class="prob-fill {rc}" style="width:{pct}%;"></div></div>
            <div class="prob-meta"><span>Predicted biopsy probability</span><span style="font-weight:600;">{pct}%</span></div>
          </div>
        </div>
        <div class="res-right">
          <div class="prob-big">{pct}%</div>
          <div class="prob-lbl">Risk Score</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    summary = {k: v for k, v in p.get("input_summary",{}).items() if v not in (0, 0.0)}
    if summary:
        st.markdown('<p class="lbl" style="margin-top:0.25rem;">Notable Input Values</p>', unsafe_allow_html=True)
        html = '<div class="summary-g">'
        for k, v in list(summary.items())[:12]:
            html += f'<div class="sum-item"><div class="sum-lbl">{k}</div><div class="sum-val">{v}</div></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    fi = p.get("feature_importance")
    if fi:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            names = fi["names"]; vals = fi["vals"]; n = len(names)
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#0b1220"); ax.set_facecolor("#0b1220")
            colors = [(0.23+i/n*0.4, 0.62-i/n*0.2, 1.0-i/n*0.35) for i in range(n)]
            bars = ax.barh(range(n), vals, color=colors, height=0.55, edgecolor="none", zorder=3)
            xmax = max(vals)
            for b, v in zip(bars, vals):
                ax.text(b.get_width()+xmax*0.015, b.get_y()+b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=9, color="#3d5270", fontfamily="monospace")
            ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=10, color="#7e97be")
            ax.set_xlabel("Importance Score", fontsize=9, color="#3d5270", labelpad=8, fontfamily="monospace")
            ax.tick_params(axis="x", colors="#1c2d46", labelsize=9, labelcolor="#3d5270")
            ax.tick_params(axis="y", length=0)
            for s in ["top","right","left"]: ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#1c2d46")
            ax.xaxis.grid(True, color="#0f1828", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
            ax.set_xlim(0, xmax*1.2)
            ax.set_title("Top Contributing Features", fontsize=12, color="#d8e4f5", pad=12, loc="left", fontweight="bold")
            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
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
            st.session_state.input_values = {}; st.session_state.prediction = None; goto("classifier")

    st.markdown("""
    <div class="disclaimer">
    <strong>⚕ Clinical Disclaimer:</strong> This prediction is generated by a machine learning model and is
    <em>not a substitute</em> for professional medical judgment. All clinical decisions must be made by a
    qualified healthcare provider following established guidelines.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE · CALENDAR
# ═══════════════════════════════════════════════════════════════════
def page_calendar():
    st.markdown("""
    <div class="page-hdr au">
      <div class="hdr-ico">📅</div>
      <div><div class="hdr-h">Appointment Calendar</div>
      <p class="hdr-p">Schedule and manage patient appointments and follow-up consultations.</p></div>
    </div>""", unsafe_allow_html=True)

    today = datetime.date.today()
    if "cal_y" not in st.session_state: st.session_state.cal_y = today.year
    if "cal_m" not in st.session_state: st.session_state.cal_m = today.month

    y, m = st.session_state.cal_y, st.session_state.cal_m
    import calendar
    month_name = datetime.date(y, m, 1).strftime("%B %Y")
    appt_days  = {datetime.datetime.strptime(a["date"],"%Y-%m-%d").date().day
                  for a in st.session_state.appointments
                  if datetime.datetime.strptime(a["date"],"%Y-%m-%d").date().year == y
                  and datetime.datetime.strptime(a["date"],"%Y-%m-%d").date().month == m}

    col_cal, col_form = st.columns([1.3, 1])

    with col_cal:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        n1, n2, n3 = st.columns([1, 4, 1])
        with n1:
            if st.button("‹", key="cal_prev"):
                if m == 1: st.session_state.cal_m = 12; st.session_state.cal_y -= 1
                else: st.session_state.cal_m -= 1
                st.rerun()
        with n2:
            st.markdown(f'<div style="text-align:center;font-family:var(--fd);font-size:1rem;font-weight:700;padding:0.35rem 0;">{month_name}</div>', unsafe_allow_html=True)
        with n3:
            if st.button("›", key="cal_next"):
                if m == 12: st.session_state.cal_m = 1; st.session_state.cal_y += 1
                else: st.session_state.cal_m += 1
                st.rerun()

        cal_html = '<div class="cal-grid">'
        for dh in ["M","T","W","T","F","S","S"]:
            cal_html += f'<div class="cal-dh">{dh}</div>'
        for week in calendar.monthcalendar(y, m):
            for day in week:
                if day == 0: cal_html += '<div class="cal-day empty"></div>'
                else:
                    cls = "cal-day"
                    if day == today.day and m == today.month and y == today.year: cls += " today"
                    if day in appt_days: cls += " has-appt"
                    cal_html += f'<div class="{cls}">{day}</div>'
        cal_html += '</div>'
        st.markdown(cal_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">➕ New Appointment</div>', unsafe_allow_html=True)
        pn = st.text_input("Patient Name", placeholder="Full name", key="ap_n")
        at = st.selectbox("Type", ["Initial Consultation","Follow-up","Biopsy Review","Screening","Post-Treatment","Emergency"], key="ap_t")
        ad = st.date_input("Date", value=today, key="ap_d")
        ah = st.selectbox("Time", ["08:00","08:30","09:00","09:30","10:00","10:30","11:00","11:30","14:00","14:30","15:00","15:30","16:00","17:00"], key="ap_h")
        an = st.text_input("Notes", placeholder="Optional notes", key="ap_note")
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("📅 Schedule Appointment", key="ap_add"):
            if pn.strip():
                st.session_state.appointments.append({"patient":pn.strip(),"type":at,"date":ad.strftime("%Y-%m-%d"),"time":ah,"note":an})
                st.success(f"✅ Appointment scheduled for {pn}")
                st.rerun()
            else: st.error("Patient name required.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="lbl">Upcoming Appointments</p>', unsafe_allow_html=True)
    upcoming = sorted([a for a in st.session_state.appointments
                       if datetime.datetime.strptime(a["date"],"%Y-%m-%d").date() >= today],
                      key=lambda x: x["date"]+x["time"])

    if not upcoming:
        st.markdown('<div class="card" style="text-align:center;padding:1.5rem;color:var(--txt3);font-size:0.82rem;">No upcoming appointments scheduled.</div>', unsafe_allow_html=True)
    else:
        for i, a in enumerate(upcoming):
            d    = datetime.datetime.strptime(a["date"],"%Y-%m-%d").date()
            dstr = d.strftime("%a, %d %b %Y")
            c1, c2 = st.columns([9, 1])
            with c1:
                st.markdown(f"""
                <div class="appt-row">
                  <div class="appt-dot"></div>
                  <div style="flex:1;">
                    <div class="appt-n">{a['patient']}</div>
                    <div class="appt-m">{a['type']} · {dstr} at {a['time']}{' · '+a['note'] if a['note'] else ''}</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="ghost">', unsafe_allow_html=True)
                if st.button("🗑", key=f"del_ap_{i}"):
                    st.session_state.appointments.remove(a); st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE · ABOUT
# ═══════════════════════════════════════════════════════════════════
def page_about():
    st.markdown("""
    <div class="page-hdr au">
      <div class="hdr-ico">ℹ️</div>
      <div><div class="hdr-h">About CervAI</div>
      <p class="hdr-p">Mission, methodology, technology stack, and the team behind the platform.</p></div>
    </div>""", unsafe_allow_html=True)

    # Hero banner
    st.markdown("""
    <div class="about-hero au">
      <div style="font-family:var(--fd);font-size:2rem;font-weight:800;color:#fff;margin-bottom:0.75rem;position:relative;z-index:1;">
        Advancing Cervical Cancer<br><em style="color:var(--blue);">Early Detection</em> with AI
      </div>
      <p style="font-size:0.86rem;color:var(--txt2);max-width:580px;line-height:1.85;position:relative;z-index:1;">
        CervAI bridges the gap between advanced machine learning and gynecological oncology.
        Cervical cancer is one of the most preventable cancers when detected early — yet expert risk
        stratification is unevenly accessible worldwide. We're changing that.
      </p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("""
        <div class="card au2" style="margin-bottom:1rem;">
          <div style="font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🧬 The Dataset</div>
          <p style="font-size:0.82rem;color:var(--txt2);line-height:1.8;">
            Trained on the <strong style="color:var(--txt);">UCI Cervical Cancer (Risk Factors) Dataset</strong> — 858 patients from the
            Hospital Universitario de Caracas, Venezuela. 36 clinical attributes span demographics,
            sexual history, contraceptive use, STD history, and prior diagnostic tests.
          </p>
          <div class="metric-grid">
            <div class="metric-box"><div class="metric-v">858</div><div class="metric-l">Patients</div></div>
            <div class="metric-box"><div class="metric-v">36</div><div class="metric-l">Features</div></div>
            <div class="metric-box"><div class="metric-v">4</div><div class="metric-l">Target Tests</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="card au3" style="margin-bottom:1rem;">
          <div style="font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">⚙️ Technology Stack</div>
          <div>
            <span class="tech-pill">🐍 Python 3.11</span><span class="tech-pill">🎈 Streamlit</span>
            <span class="tech-pill">🚀 XGBoost</span><span class="tech-pill">⚖️ SMOTE</span>
            <span class="tech-pill">🔢 NumPy</span><span class="tech-pill">🐼 Pandas</span>
            <span class="tech-pill">🎯 Scikit-learn</span><span class="tech-pill">📊 Matplotlib</span>
            <span class="tech-pill">💾 Joblib</span><span class="tech-pill">🧪 Imbalanced-learn</span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="card au4">
          <div style="font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">⚕️ Ethical Commitment</div>
          <p style="font-size:0.82rem;color:var(--txt2);line-height:1.8;">
            CervAI follows <strong style="color:var(--txt);">FAIR data principles</strong>, publishes methodology openly,
            and subjects every model update to bias audits across demographic subgroups.
            This platform is strictly for research and educational use — never a substitute for clinical judgment.
          </p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card au2" style="margin-bottom:1rem;">
          <div style="font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">👥 Research Team</div>
          <div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#1a3a5c,#0e2240);">👩‍⚕️</div><div><div class="team-name">Dr. Sophie Martin</div><div class="team-role">Lead Oncologist · Clinical Advisor</div></div></div>
          <div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#1a2a50,#0a1830);">👨‍💻</div><div><div class="team-name">Dr. Karim Hassan</div><div class="team-role">ML Engineer · Model Architecture</div></div></div>
          <div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#1a1a40,#0a0a28);">🔬</div><div><div class="team-name">Dr. Amina Osei</div><div class="team-role">Data Scientist · Feature Engineering</div></div></div>
          <div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#102a30,#081820);">📊</div><div><div class="team-name">Prof. Liu Wei</div><div class="team-role">Biostatistics · Validation Lead</div></div></div>
          <div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#2a1a40,#180a28);">🏥</div><div><div class="team-name">Dr. Elena Vasquez</div><div class="team-role">Gynecologist · Clinical Review</div></div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="card au3">
          <div style="font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">📅 Development Timeline</div>
          <div class="tl-item"><div class="tl-dot"></div><div><div class="tl-yr">Q1 2023</div><div class="tl-txt">Dataset curation and exploratory analysis on UCI repository data.</div></div></div>
          <div class="tl-item"><div class="tl-dot"></div><div><div class="tl-yr">Q3 2023</div><div class="tl-txt">XGBoost model development with SMOTE oversampling for class imbalance.</div></div></div>
          <div class="tl-item"><div class="tl-dot"></div><div><div class="tl-yr">Q1 2024</div><div class="tl-txt">Platform v1.0 launch with clinical validation and IRB approval.</div></div></div>
          <div class="tl-item" style="margin-bottom:0;"><div class="tl-dot" style="background:var(--teal);box-shadow:0 0 8px var(--teal);"></div><div><div class="tl-yr" style="color:var(--teal);">2025 — Now</div><div class="tl-txt">Federated learning integration and multi-hospital validation underway.</div></div></div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE · CONTACT US  (5 email contacts)
# ═══════════════════════════════════════════════════════════════════
def page_contact():
    st.markdown("""
    <div class="page-hdr au">
      <div class="hdr-ico">✉️</div>
      <div><div class="hdr-h">Contact Us</div>
      <p class="hdr-p">Reach our team for support, research collaborations, or clinical inquiries.</p></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="card au">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1.25rem;">📬 Send Us a Message</div>', unsafe_allow_html=True)
        cn  = st.text_input("Full Name", placeholder="Dr. Jane Smith", key="ct_n")
        ce  = st.text_input("Your Email", placeholder="you@hospital.org", key="ct_e")
        cdp = st.selectbox("Department", ["Oncology","Gynecology","Radiology","Research & Academia","Hospital Administration","Medical AI / Data Science","Other"], key="ct_dp")
        cto = st.selectbox("Send To", [
            "📋 General Inquiries — info@cerv.ai",
            "🔬 Research — research@cerv.ai",
            "🛠 Technical Support — support@cerv.ai",
            "🤝 Partnerships — partners@cerv.ai",
            "📊 Clinical Affairs — clinical@cerv.ai",
        ], key="ct_to")
        cs  = st.selectbox("Subject", ["Technical Support","Clinical Query","Research Collaboration","Model Access / API","Data Partnership","Bug Report","General Inquiry"], key="ct_s")
        cm  = st.text_area("Message", placeholder="Describe your inquiry in detail...", height=130, key="ct_m")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📨  Send Message", key="ct_send"):
            if cn.strip() and ce.strip() and cm.strip():
                st.session_state.contact_sent = True
                st.success(f"✅ Message sent to {cto.split('—')[-1].strip()}! We'll respond within 48 hours.")
            else:
                st.error("Please fill in your name, email, and message.")

        st.markdown("""
        <div style="background:var(--blue-s);border:1px solid rgba(59,158,255,0.2);border-left:3px solid var(--blue);
                    border-radius:var(--r2);padding:0.8rem 1rem;margin-top:1rem;font-size:0.76rem;color:var(--txt2);line-height:1.7;">
          <strong style="color:var(--txt);">⏱ Response Times:</strong><br>
          Technical support: 24–48 h &nbsp;·&nbsp; Clinical queries: 48–72 h &nbsp;·&nbsp; Research: 5–7 business days
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card au2">
          <div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1.1rem;">📧 Our Email Contacts</div>

          <div class="contact-method">
            <div class="cm-ico" style="background:rgba(59,158,255,0.1);border:1px solid rgba(59,158,255,0.2);">📋</div>
            <div><div class="cm-lbl">General Inquiries</div><div class="cm-val">info@cerv.ai</div></div>
          </div>
          <div class="contact-method">
            <div class="cm-ico" style="background:rgba(0,217,160,0.1);border:1px solid rgba(0,217,160,0.2);">🔬</div>
            <div><div class="cm-lbl">Research & Collaboration</div><div class="cm-val">research@cerv.ai</div></div>
          </div>
          <div class="contact-method">
            <div class="cm-ico" style="background:rgba(123,111,255,0.1);border:1px solid rgba(123,111,255,0.2);">🛠</div>
            <div><div class="cm-lbl">Technical Support</div><div class="cm-val">support@cerv.ai</div></div>
          </div>
          <div class="contact-method">
            <div class="cm-ico" style="background:rgba(255,183,77,0.1);border:1px solid rgba(255,183,77,0.2);">🤝</div>
            <div><div class="cm-lbl">Partnerships & Business</div><div class="cm-val">partners@cerv.ai</div></div>
          </div>
          <div class="contact-method">
            <div class="cm-ico" style="background:rgba(255,95,126,0.1);border:1px solid rgba(255,95,126,0.2);">📊</div>
            <div><div class="cm-lbl">Clinical Affairs</div><div class="cm-val">clinical@cerv.ai</div></div>
          </div>

          <div style="margin-top:1.2rem;background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.9rem;">
            <div style="font-size:0.72rem;color:var(--txt3);margin-bottom:0.3rem;">📍 Address</div>
            <div style="font-size:0.82rem;color:var(--txt);font-weight:500;">Institute of Medical AI</div>
            <div style="font-size:0.76rem;color:var(--txt3);">12 Innovation Drive, Suite 400<br>San Francisco, CA 94105, USA</div>
          </div>
          <div style="margin-top:0.65rem;background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.9rem;">
            <div style="font-size:0.72rem;color:var(--txt3);margin-bottom:0.3rem;">📞 Clinical Hotline (Research)</div>
            <div style="font-size:0.82rem;color:var(--txt);font-weight:500;">+1 (555) 0192-CERV</div>
            <div style="font-size:0.72rem;color:var(--txt3);">Mon–Fri · 09:00–17:00 PST</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Interactive FAQ ──
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<p class="lbl">Frequently Asked Questions</p>', unsafe_allow_html=True)
        FAQS = [
            ("Is CervAI approved for clinical use?", "No. CervAI is a research prototype and is not a certified medical device. All predictions must be reviewed by a qualified clinician before any clinical action is taken."),
            ("Can I access the source dataset?", "The UCI Cervical Cancer dataset is publicly available at the UCI Machine Learning Repository under a CC BY 4.0 license. Contact research@cerv.ai for processed versions."),
            ("How do I integrate CervAI via API?", "Contact partners@cerv.ai for API access. Full integration documentation and sandbox access is available for approved research institutions."),
            ("Is patient data stored or transmitted?", "No. CervAI does not store any patient data entered during sessions. All inputs are processed in-memory only and cleared on session end."),
            ("How can I report a bug or model issue?", "Email support@cerv.ai with a description of the issue, your browser/OS, and steps to reproduce. We aim to triage all reports within 24 hours."),
        ]
        for i, (q, a) in enumerate(FAQS):
            is_open = st.session_state.active_faq == i
            cls = "faq-item open" if is_open else "faq-item"
            arrow = "▲" if is_open else "▼"
            st.markdown(f'<div class="{cls}"><div class="faq-q">{q} <span style="color:var(--txt3);font-size:0.8rem;">{arrow}</span></div>', unsafe_allow_html=True)
            if is_open:
                st.markdown(f'<div class="faq-a">{a}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button(q[:30]+"...", key=f"faq_{i}"):
                st.session_state.active_faq = None if is_open else i
                st.rerun()


# ═══════════════════════════════════════════════════════════════════
# PAGE · PROFILE
# ═══════════════════════════════════════════════════════════════════
def page_profile():
    u = st.session_state.get("user_info", {})
    st.markdown(f"""
    <div class="page-hdr au">
      <div class="hdr-ico">👤</div>
      <div><div class="hdr-h">My Profile</div>
      <p class="hdr-p">Manage your account information and preferences.</p></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown(f"""
        <div class="card au" style="text-align:center;padding:2rem;">
          <div style="width:80px;height:80px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--purple));
                      display:flex;align-items:center;justify-content:center;font-size:1.8rem;font-weight:700;
                      color:#fff;margin:0 auto 1rem;border:3px solid var(--bord2);">{u.get('initials','?')}</div>
          <div style="font-family:var(--fd);font-size:1.2rem;font-weight:700;color:var(--txt);">{u.get('name','User')}</div>
          <div style="font-size:0.76rem;color:var(--txt3);font-family:var(--fm);margin-top:0.25rem;">{u.get('role','—')} · {u.get('dept','—')}</div>
          <div style="font-size:0.76rem;color:var(--txt3);margin-top:0.5rem;">{u.get('email','—')}</div>
          <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--bord);display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
            <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.75rem;"><div style="font-family:var(--fd);font-size:1.4rem;color:var(--blue);font-weight:700;">{len(st.session_state.history)}</div><div style="font-size:0.65rem;color:var(--txt3);">Assessments</div></div>
            <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:0.75rem;"><div style="font-family:var(--fd);font-size:1.4rem;color:var(--teal);font-weight:700;">{len(st.session_state.appointments)}</div><div style="font-size:0.65rem;color:var(--txt3);">Appointments</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card au2">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1.1rem;">✏️ Edit Profile</div>', unsafe_allow_html=True)
        new_name = st.text_input("Full Name", value=u.get("name",""), key="prof_name")
        st.text_input("Email", value=u.get("email",""), disabled=True, key="prof_email")
        st.text_input("Department", value=u.get("dept",""), key="prof_dept")
        st.selectbox("Role", ["Oncologist","Gynecologist","Radiologist","Researcher","Administrator","Nurse Practitioner"], key="prof_role")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Changes", key="prof_save"):
            st.session_state.user_info["name"] = new_name
            new_init = "".join(w[0].upper() for w in new_name.split()[:2]) if new_name else u.get("initials","?")
            st.session_state.user_info["initials"] = new_init
            st.success("✅ Profile updated successfully!")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br><div class="card au3">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🔒 Change Password</div>', unsafe_allow_html=True)
        st.text_input("Current Password", type="password", key="pw_cur")
        st.text_input("New Password", type="password", key="pw_new")
        st.text_input("Confirm New Password", type="password", key="pw_conf")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔒 Update Password", key="pw_save"):
            st.info("Password change requires backend integration (demo mode).")
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE · SETTINGS
# ═══════════════════════════════════════════════════════════════════
def page_settings():
    st.markdown("""
    <div class="page-hdr au">
      <div class="hdr-ico">⚙️</div>
      <div><div class="hdr-h">Settings</div>
      <p class="hdr-p">Configure your platform preferences and notifications.</p></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card au">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🎨 Display</div>', unsafe_allow_html=True)
        st.selectbox("Theme", ["Dark (Default)","Dark Blue","Dark Purple"], key="s_theme")
        st.selectbox("Language", ["English","Français","العربية","Español"], key="s_lang")
        st.selectbox("Date Format", ["DD/MM/YYYY","MM/DD/YYYY","YYYY-MM-DD"], key="s_date")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br><div class="card au3">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🔬 Classifier</div>', unsafe_allow_html=True)
        st.checkbox("Auto-save patient inputs", value=True, key="s_autosave")
        st.checkbox("Show feature importance chart by default", value=True, key="s_fi")
        st.selectbox("Default risk threshold", ["50% (Standard)","40% (Sensitive)","60% (Specific)"], key="s_thresh")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card au2">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🔔 Notifications</div>', unsafe_allow_html=True)
        st.checkbox("Email alerts for high-risk results", value=True, key="s_email")
        st.checkbox("Appointment reminders (24h before)", value=True, key="s_appt_rem")
        st.checkbox("Model update notifications", value=False, key="s_model_notif")
        st.checkbox("Weekly summary report", value=False, key="s_weekly")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br><div class="card au4">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--fd);font-size:0.95rem;font-weight:700;color:var(--txt);margin-bottom:1rem;">🗂 Data Management</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.8rem;color:var(--txt2);margin-bottom:0.75rem;line-height:1.6;">Manage your session data. These actions cannot be undone.</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("🗑 Clear History", key="s_clr_h"):
                st.session_state.history = []; st.success("History cleared.")
            st.markdown('</div>', unsafe_allow_html=True)
        with cc2:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("🗑 Clear Appointments", key="s_clr_a"):
                st.session_state.appointments = []; st.success("Appointments cleared.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 Save All Settings", key="s_save"):
        st.success("✅ Settings saved successfully!")


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

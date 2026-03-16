import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import datetime
import calendar

st.set_page_config(
    page_title="CervAI · Clinical Risk Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ──────────────────────────────────
_defaults = {
    "page": "login",
    "logged_in": False,
    "prediction": None,
    "input_values": {},
    "appointments": [],
    "history": [],
    "notifications": [
        {"id": 1, "text": "New screening guidelines updated", "time": "2h ago",  "read": False, "icon": "📋"},
        {"id": 2, "text": "Model v2.1 deployed successfully",  "time": "5h ago",  "read": False, "icon": "🚀"},
        {"id": 3, "text": "3 appointments scheduled today",    "time": "1d ago",  "read": True,  "icon": "📅"},
    ],
    "show_notif": False,
    "show_profile_menu": False,
    "contact_sent": False,
    "settings": {"theme": "dark", "alerts": True, "autosave": True},
    "active_faq": None,
    "show_result_overlay": False,
    "last_prediction_id": 0,
    "cal_y": datetime.date.today().year,
    "cal_m": datetime.date.today().month,
    "patient_gender": "Female",
    "show_guide": False,
    "guide_topic": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Users ──────────────────────────────────────────────────────────
USERS = {
    "bakraouladomor@gmail.com":  {"password": "cerv2024", "name": "Bakr Aoulad Omar",  "role": "Lead Researcher", "initials": "BO", "dept": "Clinical Research"},
    "yassirjbili@gmail.com":     {"password": "cerv2024", "name": "Yassir Jbili",       "role": "ML Engineer",     "initials": "YJ", "dept": "AI & Engineering"},
    "ilyaselhadad@gmail.com":    {"password": "cerv2024", "name": "Ilyas El Hadad",     "role": "Data Scientist",  "initials": "IE", "dept": "Data Science"},
    "mohammedelhadad@gmail.com": {"password": "cerv2024", "name": "Mohamed El Hadad",   "role": "Biostatistics",   "initials": "ME", "dept": "Statistics"},
    "yahyaelomari@gmail.com":    {"password": "cerv2024", "name": "Yahya El Omari",     "role": "Clinical Review", "initials": "YO", "dept": "Clinical QA"},
    "admin@cerv.ai":             {"password": "admin123", "name": "Admin User",         "role": "Administrator",   "initials": "AU", "dept": "IT & Systems"},
}

def check_login(email, password):
    for k, v in USERS.items():
        if k.lower() == email.strip().lower() and v["password"] == password.strip():
            return v
    return None

# ── Model loading ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    paths = [
        ("models/xgboost_model.joblib",  "models/xgboost_assets.joblib"),
        ("xgboost_model.joblib",          "xgboost_assets.joblib"),
        ("../models/xgboost_model.joblib","../models/xgboost_assets.joblib"),
        ("models/catboost_model.joblib",  "models/catboost_assets.joblib"),
        ("catboost_model.joblib",         "catboost_assets.joblib"),
        ("../models/catboost_model.joblib","../models/catboost_assets.joblib"),
    ]
    for mp, ap in paths:
        if Path(mp).exists() and Path(ap).exists():
            return joblib.load(mp), joblib.load(ap), mp
    return None, None, None

model, assets, model_path = load_model()

# ── Column / key maps ──────────────────────────────────────────────
COLUMN_ORDER = [
    "Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies",
    "Smokes", "Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)", "IUD", "IUD (years)", "STDs", "STDs (number)",
    "STDs:condylomatosis", "STDs:cervical condylomatosis", "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease",
    "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV",
    "STDs:Hepatitis B", "STDs:HPV", "STDs: Number of diagnosis",
    "STDs: Time since first diagnosis", "STDs: Time since last diagnosis",
    "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx", "Hinselmann", "Schiller", "Citology",
]
KEY_TO_COL = {
    "Age": "Age", "Num_sexual_partners": "Number of sexual partners",
    "First_sexual_intercourse": "First sexual intercourse", "Num_of_pregnancies": "Num of pregnancies",
    "Smokes": "Smokes", "Smokes_years": "Smokes (years)", "Smokes_packs_year": "Smokes (packs/year)",
    "Hormonal_Contraceptives": "Hormonal Contraceptives",
    "Hormonal_Contraceptives_years": "Hormonal Contraceptives (years)",
    "IUD": "IUD", "IUD_years": "IUD (years)", "STDs": "STDs", "STDs_number": "STDs (number)",
    "STDs_condylomatosis": "STDs:condylomatosis",
    "STDs_cervical_condylomatosis": "STDs:cervical condylomatosis",
    "STDs_vaginal_condylomatosis": "STDs:vaginal condylomatosis",
    "STDs_vulvo_perineal_condylomatosis": "STDs:vulvo-perineal condylomatosis",
    "STDs_syphilis": "STDs:syphilis",
    "STDs_pelvic_inflammatory_disease": "STDs:pelvic inflammatory disease",
    "STDs_genital_herpes": "STDs:genital herpes",
    "STDs_molluscum_contagiosum": "STDs:molluscum contagiosum",
    "STDs_AIDS": "STDs:AIDS", "STDs_HIV": "STDs:HIV", "STDs_Hepatitis_B": "STDs:Hepatitis B",
    "STDs_HPV": "STDs:HPV", "STDs_Number_of_diagnosis": "STDs: Number of diagnosis",
    "STDs_Time_since_first_diagnosis": "STDs: Time since first diagnosis",
    "STDs_Time_since_last_diagnosis": "STDs: Time since last diagnosis",
    "Dx_Cancer": "Dx:Cancer", "Dx_CIN": "Dx:CIN", "Dx_HPV": "Dx:HPV", "Dx": "Dx",
    "Hinselmann": "Hinselmann", "Schiller": "Schiller", "Citology": "Citology",
}

STD_KEYS = [
    "STDs_condylomatosis", "STDs_cervical_condylomatosis", "STDs_vaginal_condylomatosis",
    "STDs_vulvo_perineal_condylomatosis", "STDs_syphilis", "STDs_pelvic_inflammatory_disease",
    "STDs_genital_herpes", "STDs_molluscum_contagiosum", "STDs_AIDS", "STDs_HIV",
    "STDs_Hepatitis_B", "STDs_HPV",
]



# ══════════════════════════════════════════════════════════════════
# MEDICAL GUIDE DATA
# ══════════════════════════════════════════════════════════════════
MEDICAL_GUIDE = {
    "HPV (Human Papillomavirus)": {
        "icon": "🦠", "category": "Virus",
        "color": "#0c9b58",
        "short": "Primary cause of cervical cancer. A sexually transmitted virus with 200+ strains.",
        "description": "HPV is a group of more than 200 related viruses spread through sexual contact. High-risk strains (HPV 16 and 18) cause about 70% of cervical cancers. Most HPV infections clear on their own, but persistent infections can lead to cell changes and, eventually, cancer.",
        "symptoms": "Usually asymptomatic. Some strains cause genital warts. Persistent infection may show as abnormal Pap smear results.",
        "prevention": "HPV vaccine (Gardasil), condom use, regular Pap smears and HPV tests.",
        "relevance": "Directly linked to cervical cancer. Presence of HPV infection is a major risk factor in this assessment.",
    },
    "Condylomatosis": {
        "icon": "🔬", "category": "STD", "color": "#0891b2",
        "short": "Genital warts caused by low-risk HPV strains (6 and 11). Highly contagious.",
        "description": "Condylomatosis refers to genital warts (condylomata acuminata) caused primarily by HPV types 6 and 11. They appear on the genitals, anus, and surrounding areas. While these strains rarely cause cancer, their presence indicates HPV infection and sexual transmission of the virus.",
        "symptoms": "Soft, flesh-colored growths on genitals or anus. Usually painless but can cause itching.",
        "prevention": "HPV vaccination, condom use, regular STI screening.",
        "relevance": "Indicates active HPV infection; co-infection with high-risk HPV strains increases cervical cancer risk.",
    },
    "Cervical Condylomatosis": {
        "icon": "🔬", "category": "STD / Gynecology", "color": "#0891b2",
        "short": "Genital warts specifically located on the cervix.",
        "description": "Cervical condylomatosis is the presence of HPV-induced warts directly on the cervix. These lesions are often invisible to the naked eye and are detected during colposcopy or Pap smear. They indicate direct HPV involvement at the cervix.",
        "symptoms": "Usually asymptomatic. May cause abnormal vaginal discharge or post-coital bleeding.",
        "prevention": "Regular Pap smears, colposcopy follow-up, HPV vaccination.",
        "relevance": "HPV at the cervix is the primary mechanism of cervical dysplasia and cancer.",
    },
    "Syphilis": {
        "icon": "⚕️", "category": "Bacterial STD", "color": "#d97706",
        "short": "A bacterial STI caused by Treponema pallidum. Progresses in stages if untreated.",
        "description": "Syphilis progresses through primary (chancre sore), secondary (rash), latent, and tertiary stages. Untreated, it can damage the heart, brain, and other organs. It is transmitted through direct contact with a sore during sexual activity.",
        "symptoms": "Stage 1: painless sore. Stage 2: rash on palms/soles, fever. Stage 3: organ damage.",
        "prevention": "Condom use, regular STI testing, penicillin treatment for infected individuals.",
        "relevance": "Co-infection may impair immune response, potentially facilitating HPV persistence.",
    },
    "Pelvic Inflammatory Disease (PID)": {
        "icon": "🏥", "category": "Gynecological Condition", "color": "#e53e51",
        "short": "Infection of the female reproductive organs, often caused by STIs.",
        "description": "PID is an infection of the uterus, fallopian tubes, and/or ovaries. It commonly results from untreated STIs such as chlamydia and gonorrhea. PID can cause chronic pelvic pain, ectopic pregnancy, and infertility if untreated.",
        "symptoms": "Lower abdominal pain, abnormal discharge, fever, pain during intercourse, irregular menstrual bleeding.",
        "prevention": "STI screening and treatment, condom use, early antibiotic treatment.",
        "relevance": "Indicates history of STI exposure and potential immune compromise affecting HPV clearance.",
    },
    "Genital Herpes (HSV-2)": {
        "icon": "⚠️", "category": "Viral STD", "color": "#7c3aed",
        "short": "Caused by Herpes Simplex Virus type 2. Causes recurrent sores. Lifelong.",
        "description": "Genital herpes is caused by HSV-2. It causes recurrent outbreaks of painful blisters around the genitals. The virus remains dormant in nerve cells. There is no cure, but antiviral medications reduce symptoms and transmission risk.",
        "symptoms": "Painful blisters/sores on genitals, burning urination, flu-like symptoms during first outbreak.",
        "prevention": "Condom use, antiviral suppressive therapy, avoiding contact during outbreaks.",
        "relevance": "HSV-2 co-infection may increase susceptibility to HIV and facilitate HPV-related dysplasia.",
    },
    "Molluscum Contagiosum": {
        "icon": "🔵", "category": "Viral Skin Infection", "color": "#0891b2",
        "short": "A benign viral skin infection causing small dome-shaped bumps. Sexually transmitted in adults.",
        "description": "Caused by a poxvirus, molluscum contagiosum produces small pearly-white dome-shaped bumps. In adults it is often sexually transmitted. It is generally self-limiting but can persist in immunocompromised individuals.",
        "symptoms": "Small firm bumps (2-5mm) with a central dimple, usually on genitals or inner thighs.",
        "prevention": "Avoid skin-to-skin contact during outbreaks, maintain hygiene.",
        "relevance": "Indicates STI exposure history; immune suppression allowing molluscum to persist may also impair HPV clearance.",
    },
    "AIDS / HIV": {
        "icon": "🔴", "category": "Viral Immunodeficiency", "color": "#e53e51",
        "short": "HIV destroys immune cells. AIDS is the advanced stage. Severely increases cancer risk.",
        "description": "HIV attacks CD4+ T-cells, weakening the immune system. AIDS is the advanced stage. People with HIV/AIDS are significantly more vulnerable to opportunistic infections and cancers including cervical cancer. Cervical cancer is an AIDS-defining illness.",
        "symptoms": "HIV: flu-like illness initially, then often asymptomatic for years. AIDS: severe infections, weight loss.",
        "prevention": "Condom use, PrEP medication, antiretroviral therapy (ART).",
        "relevance": "HIV/AIDS is a major risk multiplier. Immunosuppression prevents HPV clearance and accelerates cancer progression.",
    },
    "Hepatitis B": {
        "icon": "🟡", "category": "Viral Hepatitis", "color": "#d97706",
        "short": "A liver infection caused by HBV. Transmitted sexually and through blood.",
        "description": "Hepatitis B is caused by HBV and can be acute or chronic. Chronic HBV can cause cirrhosis and liver cancer. Transmitted through blood, sexual contact, and mother-to-child at birth.",
        "symptoms": "Jaundice, fatigue, abdominal pain, dark urine. Many chronic cases are asymptomatic.",
        "prevention": "Hepatitis B vaccine (highly effective), condom use, antiviral treatment.",
        "relevance": "Shares transmission routes with HPV; co-infection indicates high-risk sexual behaviour.",
    },
    "Hormonal Contraceptives": {
        "icon": "💊", "category": "Medication / Risk Factor", "color": "#7c3aed",
        "short": "Pills, patches, injections containing hormones. Long-term use slightly increases cervical cancer risk.",
        "description": "Hormonal contraceptives use synthetic estrogen and/or progestin to prevent pregnancy. Long-term use (5+ years) slightly increases cervical cancer risk, possibly by affecting the cervical microenvironment or influencing HPV gene expression.",
        "symptoms": "Not a disease — possible side effects include nausea, mood changes, spotting.",
        "prevention": "Regular Pap smears for long-term users; discuss risks with healthcare provider.",
        "relevance": "Women using hormonal contraceptives for >5 years have a slightly elevated cervical cancer risk.",
    },
    "IUD (Intrauterine Device)": {
        "icon": "🩺", "category": "Contraceptive Device", "color": "#10b981",
        "short": "A small T-shaped device inserted into the uterus. Hormonal or copper-based.",
        "description": "An IUD is a contraceptive device inserted into the uterus. Hormonal IUDs release progestin. Copper IUDs are non-hormonal. Some studies suggest copper IUDs may reduce cervical cancer risk by inducing a local immune response.",
        "symptoms": "Not a disease. Heavier periods (copper) or lighter periods (hormonal) are common.",
        "prevention": "Regular gynecological check-ups.",
        "relevance": "IUD type and duration are tracked as contraceptive variables with potential influence on cancer risk.",
    },
    "Hinselmann Test (Colposcopy)": {
        "icon": "🔎", "category": "Diagnostic Test", "color": "#0891b2",
        "short": "Visual cervical examination using a colposcope. Detects abnormal tissue after acetic acid application.",
        "description": "Colposcopy (Hinselmann test) uses magnification to examine the cervix after applying acetic acid. Abnormal tissue turns white (acetowhite lesion) and is biopsied. Used when a Pap smear shows abnormal results.",
        "symptoms": "Not applicable — this is a diagnostic procedure.",
        "prevention": "Not applicable.",
        "relevance": "A positive result indicates visually abnormal cervical tissue — a strong predictor of dysplasia or cancer.",
    },
    "Schiller Test": {
        "icon": "🧪", "category": "Diagnostic Test", "color": "#d97706",
        "short": "Iodine-based test. Healthy tissue stains brown; abnormal tissue stays pale/unstained.",
        "description": "The Schiller test applies Lugol's iodine to the cervix. Normal cells (containing glycogen) stain dark brown. Abnormal or cancerous cells do not stain and appear pale yellow or white. These areas are then biopsied.",
        "symptoms": "Not applicable — this is a diagnostic procedure.",
        "prevention": "Not applicable.",
        "relevance": "Positive (iodine-negative) result indicates potentially abnormal cervical cells requiring biopsy.",
    },
    "Cytology (Pap Smear)": {
        "icon": "🧬", "category": "Diagnostic Test", "color": "#0c9b58",
        "short": "Screening test collecting cervical cells to detect abnormalities under a microscope.",
        "description": "The Pap smear collects cervical cells for microscopic examination. Results range from normal to ASCUS, LSIL, HSIL, or cancer. Regular Pap smears have dramatically reduced cervical cancer mortality worldwide.",
        "symptoms": "Not applicable — this is a screening test.",
        "prevention": "Not applicable.",
        "relevance": "A positive cytology result indicates cellular abnormalities and is a primary indicator for further workup.",
    },
    "CIN (Cervical Intraepithelial Neoplasia)": {
        "icon": "⚕️", "category": "Pre-cancerous Condition", "color": "#e53e51",
        "short": "Abnormal cell growth on the cervix. Graded CIN 1, 2, or 3. May progress to cancer.",
        "description": "CIN is a precancerous cervical condition caused by HPV. CIN 1 = mild dysplasia (often self-resolving). CIN 2 = moderate dysplasia. CIN 3 = severe dysplasia/carcinoma in situ with significant cancer progression risk if untreated. Treatment: LEEP, cryotherapy, or cone biopsy.",
        "symptoms": "Usually asymptomatic. Detected on Pap smear or colposcopy.",
        "prevention": "HPV vaccination, regular Pap smears, treatment of high-grade CIN lesions.",
        "relevance": "Prior CIN diagnosis is a major risk factor. CIN 2/3 are direct precursors to invasive cervical cancer.",
    },
}

# ══════════════════════════════════════════════════════════════════
# MEDICAL GUIDE MODAL RENDERER
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# MEDICAL GUIDE MODAL RENDERER
# ══════════════════════════════════════════════════════════════════
def render_guide_modal():
    topic = st.session_state.get("guide_topic")
    if not topic or topic not in MEDICAL_GUIDE:
        return
    info  = MEDICAL_GUIDE[topic]
    c_val = info["color"]
    st.markdown(
        f'<div style="background:#fff;border:2px solid {c_val};border-radius:16px;padding:1.5rem 2rem;'
        f'margin-bottom:1.25rem;animation:fadeUp .3s ease both;box-shadow:var(--sh3);">'
        f'<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1rem;">'
        f'<span style="font-size:1.8rem;">{info["icon"]}</span>'
        f'<div><div style="font-family:var(--ff-display);font-size:1.2rem;color:var(--txt);">{topic}</div>'
        f'<div style="font-family:var(--ff-mono);font-size:.57rem;letter-spacing:.18em;text-transform:uppercase;color:{c_val};">{info["category"]}</div>'
        f'</div></div>'
        f'<div style="font-size:.82rem;color:var(--txt2);line-height:1.8;margin-bottom:1rem;">{info["description"]}</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-bottom:.75rem;">'
        f'<div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:.75rem 1rem;">'
        f'<div style="font-family:var(--ff-mono);font-size:.54rem;letter-spacing:.15em;text-transform:uppercase;color:var(--txt3);margin-bottom:.35rem;">Symptoms</div>'
        f'<div style="font-size:.78rem;color:var(--txt2);line-height:1.65;">{info["symptoms"]}</div></div>'
        f'<div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:.75rem 1rem;">'
        f'<div style="font-family:var(--ff-mono);font-size:.54rem;letter-spacing:.15em;text-transform:uppercase;color:var(--txt3);margin-bottom:.35rem;">Prevention</div>'
        f'<div style="font-size:.78rem;color:var(--txt2);line-height:1.65;">{info["prevention"]}</div></div></div>'
        f'<div style="background:rgba(12,155,88,.06);border:1px solid rgba(12,155,88,.2);border-left:3px solid {c_val};border-radius:10px;padding:.75rem 1rem;">'
        f'<div style="font-family:var(--ff-mono);font-size:.54rem;letter-spacing:.15em;text-transform:uppercase;color:var(--txt3);margin-bottom:.35rem;">Relevance to cervical cancer risk</div>'
        f'<div style="font-size:.78rem;color:var(--txt2);line-height:1.65;">{info["relevance"]}</div></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("Close Guide  ✕", key="close_guide"):
        st.session_state.show_guide  = False
        st.session_state.guide_topic = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

:root {
  /* Palette */
  --bg:       #f6faf7;
  --bg2:      #edf5f0;
  --surf:     #ffffff;
  --surf2:    #f3faf6;
  --surf3:    #e8f5ed;
  --surf4:    #d9eee3;
  --bord:     #cce4d6;
  --bord2:    #b0d4bf;
  --bord3:    #88bfa4;

  /* Brand */
  --green:    #0c9b58;
  --green2:   #0a8a4f;
  --green3:   #067840;
  --emerald:  #10b981;
  --mint:     #e6f7ef;
  --green-g:  rgba(12,155,88,.11);
  --green-s:  rgba(12,155,88,.055);

  /* Accents */
  --teal:     #0891b2;
  --rose:     #e53e51;
  --amber:    #d97706;
  --purple:   #7c3aed;
  --sage:     #5aaf82;

  /* Text */
  --txt:      #0e2e1e;
  --txt2:     #2d6649;
  --txt3:     #6b9e80;

  /* Shape */
  --r:  14px;
  --r2: 10px;
  --r3:  7px;

  /* Typography */
  --ff-body:    'DM Sans', sans-serif;
  --ff-display: 'DM Serif Display', serif;
  --ff-mono:    'JetBrains Mono', monospace;

  /* Shadows */
  --sh1: 0 2px 8px  rgba(10,60,30,.07);
  --sh2: 0 4px 18px rgba(10,60,30,.10);
  --sh3: 0 8px 36px rgba(10,60,30,.14);
}

/* ── Reset & base ─────────────────────────────────── */
html, body, [class*="css"] {
  font-family: var(--ff-body) !important;
  background:  var(--bg) !important;
  color:       var(--txt) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: var(--bg); }
.main .block-container { padding: 1.5rem 2rem 4rem; max-width: 1380px; }

::-webkit-scrollbar            { width: 5px; height: 5px; }
::-webkit-scrollbar-track      { background: var(--bg); }
::-webkit-scrollbar-thumb      { background: var(--bord2); border-radius: 3px; }

/* ── Ambient background ───────────────────────────── */
.stApp::before {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(ellipse 800px 600px at 0% 0%,   rgba(12,155,88,.07) 0%, transparent 60%),
    radial-gradient(ellipse 600px 500px at 100% 100%, rgba(8,145,178,.05) 0%, transparent 60%);
  animation: bgBreathe 12s ease-in-out infinite alternate;
}
@keyframes bgBreathe { 0%{opacity:.7;} 100%{opacity:1;} }

.stApp::after {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image: radial-gradient(circle 1px at 1px 1px, rgba(12,155,88,.07) 1px, transparent 0);
  background-size: 36px 36px;
}

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f4faf6 100%) !important;
  border-right: 1px solid var(--bord) !important;
  min-width: 260px !important; max-width: 260px !important;
  box-shadow: 2px 0 16px rgba(10,60,30,.07) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.25rem 1rem 2rem; }
[data-testid="collapsedControl"]            { display: none !important; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] { display: none !important; }

/* ── Buttons ──────────────────────────────────────── */
.stButton > button {
  font-family: var(--ff-body) !important;
  font-weight: 600 !important;
  border-radius: var(--r2) !important;
  font-size: .875rem !important;
  transition: all .22s cubic-bezier(.22,1,.36,1) !important;
  border: none !important;
  width: 100%;
  background: linear-gradient(135deg, var(--green), var(--green2)) !important;
  color: #fff !important;
  padding: .7rem 1.5rem !important;
  box-shadow: 0 4px 14px rgba(12,155,88,.28) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(12,155,88,.38) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.ghost .stButton > button {
  background: var(--surf) !important; color: var(--txt2) !important;
  border: 1.5px solid var(--bord) !important; box-shadow: var(--sh1) !important;
  padding: .5rem 1rem !important; font-size: .8rem !important; width: auto !important;
}
.ghost .stButton > button:hover {
  background: var(--surf2) !important; color: var(--green) !important;
  border-color: var(--green) !important; transform: none !important;
}

.nav-btn .stButton > button {
  background: transparent !important; color: var(--txt2) !important;
  border: none !important; box-shadow: none !important;
  text-align: left !important; padding: .6rem .9rem !important;
  font-size: .84rem !important; border-radius: var(--r2) !important; width: 100% !important;
}
.nav-btn .stButton > button:hover {
  background: var(--green-s) !important; color: var(--green) !important; transform: none !important;
}
.nav-btn-active .stButton > button {
  background: var(--green-g) !important; color: var(--green) !important;
  box-shadow: none !important; font-weight: 700 !important;
  border-left: 3px solid var(--green) !important;
}
.nav-btn-active .stButton > button:hover {
  background: var(--green-g) !important; transform: none !important;
}

.run-btn-wrap .stButton > button {
  background: linear-gradient(135deg, var(--green3), var(--green), var(--emerald)) !important;
  font-size: 1rem !important; font-weight: 700 !important;
  padding: 1rem 2.5rem !important; border-radius: 14px !important;
  width: 100% !important; animation: glow 3s ease infinite !important;
}
.run-btn-wrap .stButton > button:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 12px 36px rgba(12,155,88,.45) !important;
}

/* ── Inputs ───────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: var(--surf) !important; border: 1.5px solid var(--bord) !important;
  color: var(--txt) !important; border-radius: var(--r2) !important;
  font-family: var(--ff-body) !important; transition: all .2s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--green) !important;
  box-shadow: 0 0 0 3px var(--green-g) !important;
  outline: none !important;
}
[data-testid="stSelectbox"] * {
  background: var(--surf) !important; color: var(--txt) !important;
  border-color: var(--bord) !important;
}
label[data-testid="stWidgetLabel"] p {
  font-size: .75rem !important; color: var(--txt2) !important; font-weight: 500 !important;
}

[data-testid="stNumberInput"]               { margin-bottom: 0 !important; }
[data-testid="stNumberInput"] input {
  background: var(--surf) !important; border: 1.5px solid var(--bord) !important;
  color: var(--txt) !important; border-radius: var(--r2) !important;
  font-family: var(--ff-mono) !important; font-size: .9rem !important;
  font-weight: 500 !important; padding: .55rem .75rem !important; transition: all .2s !important;
}
[data-testid="stNumberInput"] input:focus {
  border-color: var(--green) !important; box-shadow: 0 0 0 3px var(--green-g) !important; outline: none !important;
}
[data-testid="stNumberInput"] button {
  background: var(--surf2) !important; border-color: var(--bord) !important;
  color: var(--txt2) !important; border-radius: var(--r3) !important;
}

/* ── Radio (Yes/No toggles) ───────────────────────── */
div[data-testid="stRadio"] > div {
  display: flex !important; flex-direction: row !important; gap: 0 !important;
  background: var(--surf3) !important; border: 1.5px solid var(--bord) !important;
  border-radius: 10px !important; padding: 3px !important; width: fit-content !important;
}
div[data-testid="stRadio"] > div > label {
  display: flex !important; align-items: center !important; justify-content: center !important;
  padding: .45rem 1.4rem !important; border-radius: 7px !important; cursor: pointer !important;
  font-family: var(--ff-mono) !important; font-size: .72rem !important;
  font-weight: 600 !important; letter-spacing: .08em !important; transition: all .18s !important;
  color: var(--txt3) !important; background: transparent !important; min-width: 68px !important;
}
div[data-testid="stRadio"] > div > label:has(input:checked) {
  background: var(--surf) !important; box-shadow: 0 1px 6px rgba(10,60,30,.10) !important;
}
div[data-testid="stRadio"] > div > label:first-of-type:has(input:checked) {
  color: var(--green) !important; border: 1px solid rgba(12,155,88,.35) !important;
}
div[data-testid="stRadio"] > div > label:last-of-type:has(input:checked) {
  color: var(--rose) !important; border: 1px solid rgba(232,72,85,.3) !important;
}
div[data-testid="stRadio"] input[type="radio"] { display: none !important; }
div[data-testid="stRadio"] > label            { display: none !important; }

/* ── Expander ─────────────────────────────────────── */
[data-testid="stExpander"]                        { background: transparent !important; border: none !important; margin-bottom: 0 !important; }
[data-testid="stExpander"] summary                {
  background: var(--surf2) !important; border: 1px solid var(--bord) !important;
  border-radius: var(--r2) !important; color: var(--txt2) !important;
  font-size: .8rem !important; font-weight: 500 !important;
  padding: .7rem 1rem !important; margin-bottom: .5rem !important;
}
[data-testid="stExpander"] > div:last-child       { background: transparent !important; border-top: none !important; padding: .25rem 0 0 !important; }

/* ── Alerts ───────────────────────────────────────── */
.stSuccess > div { background: rgba(12,155,88,.07)  !important; border: 1px solid rgba(12,155,88,.25) !important; border-radius: var(--r2) !important; color: var(--green) !important; }
.stError   > div { background: rgba(232,72,85,.07)  !important; border: 1px solid rgba(232,72,85,.25) !important; border-radius: var(--r2) !important; }
.stInfo    > div { background: var(--green-s)        !important; border: 1px solid rgba(12,155,88,.2)  !important; border-radius: var(--r2) !important; }
.stWarning > div { background: rgba(217,119,6,.07)   !important; border: 1px solid rgba(217,119,6,.2)  !important; border-radius: var(--r2) !important; }

/* ── Animations ───────────────────────────────────── */
@keyframes fadeUp   { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:none} }
@keyframes fadeIn   { from{opacity:0} to{opacity:1} }
@keyframes glow     { 0%,100%{box-shadow:0 4px 18px rgba(12,155,88,.28)} 50%{box-shadow:0 4px 32px rgba(12,155,88,.50)} }
@keyframes pulse2   { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.45;transform:scale(.8)} }
@keyframes spin2    { to{transform:rotate(360deg)} }
@keyframes floatCard{ 0%,100%{transform:translateY(0)} 50%{transform:translateY(-7px)} }
@keyframes cardEnter{ from{opacity:0;transform:translateY(22px) scale(.97)} to{opacity:1;transform:none} }
@keyframes ecg      { 0%{background-position:0 0} 100%{background-position:-1440px 0} }
@keyframes dotPop   { 0%{transform:scale(0);opacity:0} 70%{transform:scale(1.2)} 100%{transform:scale(1);opacity:1} }
@keyframes shimmerBar{ 0%{background-position:-600px 0} 100%{background-position:600px 0} }

.au  { animation: fadeUp .42s cubic-bezier(.22,1,.36,1) both; }
.au2 { animation: fadeUp .42s cubic-bezier(.22,1,.36,1) .08s both; }
.au3 { animation: fadeUp .42s cubic-bezier(.22,1,.36,1) .16s both; }
.au4 { animation: fadeUp .42s cubic-bezier(.22,1,.36,1) .24s both; }

/* ── Ring gauge ───────────────────────────────────── */
.ring-svg        { transform: rotate(-90deg); }
.ring-track      { fill: none; stroke: var(--surf3); stroke-width: 14; }
.ring-fill-high  { fill: none; stroke: var(--rose);  stroke-width: 14; stroke-linecap: round; stroke-dasharray: 339; transition: stroke-dashoffset 1.2s cubic-bezier(.22,1,.36,1); }
.ring-fill-low   { fill: none; stroke: var(--green); stroke-width: 14; stroke-linecap: round; stroke-dasharray: 339; transition: stroke-dashoffset 1.2s cubic-bezier(.22,1,.36,1); }

/* ── Section cards ────────────────────────────────── */
.section-card {
  background: var(--surf); border: 1.5px solid var(--bord); border-radius: 16px;
  padding: 1.1rem 1.3rem .85rem; margin-bottom: .7rem; position: relative; overflow: hidden;
  box-shadow: var(--sh1); transition: box-shadow .2s, border-color .2s;
}
.section-card:hover { box-shadow: var(--sh2); border-color: var(--green); }
.sc-stripe { position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.sc-blue    { background: linear-gradient(90deg, var(--teal),   var(--green),  transparent 75%); }
.sc-smoke   { background: linear-gradient(90deg, #94a3b8,       #64748b,       transparent 75%); }
.sc-contra  { background: linear-gradient(90deg, var(--purple), var(--rose),   transparent 75%); }
.sc-std     { background: linear-gradient(90deg, var(--green),  var(--teal),   transparent 75%); }
.sc-dx      { background: linear-gradient(90deg, var(--amber),  var(--rose),   transparent 75%); }
.section-eyebrow { font-family: var(--ff-mono); font-size: .54rem; letter-spacing: .22em; text-transform: uppercase; color: var(--txt3); margin-bottom: .15rem; }
.section-title   { font-family: var(--ff-display); font-size: 1rem; font-weight: 400; color: var(--txt); display: flex; align-items: baseline; gap: .5rem; margin-bottom: .05rem; }
.section-subtitle{ font-size: .72rem; font-weight: 400; color: var(--txt3); font-family: var(--ff-body); }
.section-desc    { font-size: .72rem; color: var(--txt3); line-height: 1.55; margin-bottom: .75rem; margin-top: .05rem; }

/* ── Field rows ───────────────────────────────────── */
.field-row {
  display: grid; grid-template-columns: 1fr auto; align-items: center;
  background: var(--surf); border: 1.5px solid var(--bord); border-radius: 10px;
  padding: .55rem .85rem; margin-bottom: .4rem; gap: 1rem;
  transition: border-color .18s, box-shadow .18s;
}
.field-row:hover { border-color: var(--green); box-shadow: 0 0 0 3px var(--green-g); }
.field-label      { font-size: .76rem; font-weight: 500; color: var(--txt2); }
.field-label small{ display: block; font-size: .65rem; color: var(--txt3); font-family: var(--ff-mono); margin-top: 1px; }
.yn-wrap {
  background: var(--surf); border: 1.5px solid var(--bord); border-radius: 10px;
  padding: .5rem .85rem; margin-bottom: .4rem;
  display: flex; align-items: center; justify-content: space-between;
  transition: border-color .18s;
}
.yn-wrap:hover { border-color: var(--green); }
.yn-label { font-size: .76rem; font-weight: 500; color: var(--txt2); }

/* ── Progress bar ─────────────────────────────────── */
.prog-wrap {
  background: var(--surf); border: 1px solid var(--bord); border-radius: var(--r2);
  padding: .6rem 1.2rem; margin-bottom: 1.2rem;
  display: flex; align-items: center; gap: 1rem; box-shadow: var(--sh1);
}
.prog-lbl  { font-family: var(--ff-mono); font-size: .57rem; letter-spacing: .12em; text-transform: uppercase; color: var(--txt3); white-space: nowrap; }
.prog-bar  { flex: 1; background: var(--surf3); border-radius: 100px; height: 5px; overflow: hidden; }
.prog-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, var(--green), var(--emerald), var(--green)); background-size: 200% 100%; animation: shimmerBar 2s linear infinite; transition: width .6s cubic-bezier(.22,1,.36,1); }
.prog-pct  { font-family: var(--ff-mono); font-size: .7rem; color: var(--green); white-space: nowrap; font-weight: 600; }

/* ── Cards ────────────────────────────────────────── */
.card    { background: var(--surf); border: 1px solid var(--bord); border-radius: var(--r); padding: 1.5rem; box-shadow: var(--sh1); transition: box-shadow .2s, transform .2s; }
.card:hover { box-shadow: var(--sh2); transform: translateY(-1px); }
.card-sm { background: var(--surf); border: 1px solid var(--bord); border-radius: var(--r); padding: 1rem 1.25rem; box-shadow: var(--sh1); }

/* ── Top-bar ──────────────────────────────────────── */
.topbar {
  display: flex; align-items: center; gap: 1rem; padding: .75rem 1.4rem;
  background: rgba(255,255,255,.95); backdrop-filter: blur(20px);
  border: 1px solid var(--bord); border-radius: var(--r);
  margin-bottom: 1.75rem; position: sticky; top: .5rem; z-index: 99;
  box-shadow: 0 2px 14px rgba(10,60,30,.07); animation: fadeIn .4s ease both;
}
.topbar-logo  { font-family: var(--ff-display); font-size: 1.1rem; color: var(--txt); white-space: nowrap; }
.topbar-logo em{ color: var(--green); font-style: italic; }
.topbar-sep   { width: 1px; height: 16px; background: var(--bord2); }
.topbar-page  { font-family: var(--ff-mono); font-size: .6rem; letter-spacing: .18em; text-transform: uppercase; color: var(--txt3); }
.topbar-right { margin-left: auto; display: flex; align-items: center; gap: .6rem; }
.tb-pill      { font-family: var(--ff-mono); font-size: .54rem; letter-spacing: .12em; text-transform: uppercase; padding: 3px 11px; border-radius: 20px; background: rgba(12,155,88,.08); color: var(--green); border: 1px solid rgba(12,155,88,.2); }
.tb-ico       { width: 32px; height: 32px; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: .9rem; background: var(--surf2); border: 1px solid var(--bord); transition: all .2s; position: relative; }
.tb-ico:hover { background: var(--surf3); border-color: var(--green); }
.tb-badge     { position: absolute; top: -4px; right: -4px; width: 16px; height: 16px; border-radius: 50%; background: var(--rose); color: #fff; font-size: .48rem; font-weight: 700; display: flex; align-items: center; justify-content: center; border: 2px solid var(--bg); animation: dotPop .4s cubic-bezier(.22,1,.36,1); }
.tb-avatar    { width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, var(--green), var(--emerald)); display: flex; align-items: center; justify-content: center; font-size: .72rem; font-weight: 700; color: #fff; border: 2px solid var(--bord2); cursor: pointer; transition: all .2s; }
.tb-avatar:hover { border-color: var(--green); box-shadow: 0 0 0 3px var(--green-g); }

/* ── Sidebar components ───────────────────────────── */
.sb-logo  { font-family: var(--ff-display); font-size: 1.5rem; color: var(--txt); padding-bottom: 1rem; margin-bottom: 1rem; border-bottom: 1px solid var(--bord); }
.sb-logo em{ color: var(--green); font-style: italic; }
.sb-user  { display: flex; align-items: center; gap: .75rem; background: linear-gradient(135deg, var(--green-g), rgba(12,155,88,.05)); border: 1px solid rgba(12,155,88,.2); border-radius: var(--r2); padding: .75rem; margin-bottom: 1.25rem; }
.sb-avatar{ width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, var(--green), var(--emerald)); display: flex; align-items: center; justify-content: center; font-size: .92rem; font-weight: 700; color: #fff; flex-shrink: 0; box-shadow: 0 2px 10px rgba(12,155,88,.30); }
.sb-name  { font-size: .82rem; font-weight: 600; color: var(--txt); }
.sb-role  { font-size: .67rem; color: var(--txt3); font-family: var(--ff-mono); }
.sb-section{ font-family: var(--ff-mono); font-size: .54rem; letter-spacing: .2em; text-transform: uppercase; color: var(--txt3); padding: .3rem .6rem; margin-bottom: .35rem; }
.sb-divider{ border: none; border-top: 1px solid var(--bord); margin: .85rem 0; }
.sb-status { display: flex; align-items: center; gap: .5rem; font-size: .7rem; color: var(--txt3); padding: .4rem .9rem; }
.sb-dot    { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px rgba(12,155,88,.6); flex-shrink: 0; animation: pulse2 2.5s ease infinite; }

/* ── Shared components ────────────────────────────── */
.team-card  { display: flex; align-items: center; gap: .9rem; background: var(--surf2); border: 1px solid var(--bord); border-radius: var(--r2); padding: .9rem 1.1rem; margin-bottom: .6rem; transition: all .2s; }
.team-card:hover { border-color: var(--green); transform: translateX(4px); box-shadow: var(--sh1); }
.team-av    { width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0; }
.team-name  { font-size: .84rem; font-weight: 600; color: var(--txt); }
.team-role  { font-size: .7rem; color: var(--txt3); font-family: var(--ff-mono); }
.tech-pill  { display: inline-flex; align-items: center; gap: .3rem; background: var(--surf2); border: 1px solid var(--bord); border-radius: 6px; padding: 4px 10px; font-size: .73rem; color: var(--txt2); margin: .2rem; transition: all .2s; }
.tech-pill:hover { border-color: var(--green); color: var(--green); background: var(--green-g); }
.metric-grid{ display: grid; grid-template-columns: repeat(3,1fr); gap: .65rem; margin-top: .75rem; }
.metric-box { background: var(--surf2); border: 1px solid var(--bord); border-radius: var(--r2); padding: 1.1rem; text-align: center; transition: all .2s; }
.metric-box:hover { border-color: var(--green); transform: scale(1.03); box-shadow: var(--sh2); }
.metric-v   { font-family: var(--ff-display); font-size: 1.9rem; color: var(--green); }
.metric-l   { font-size: .68rem; color: var(--txt3); }
.contact-method { display: flex; align-items: center; gap: .9rem; background: var(--surf2); border: 1px solid var(--bord); border-radius: var(--r2); padding: .9rem 1.1rem; margin-bottom: .55rem; transition: all .2s; }
.contact-method:hover { border-color: var(--green); transform: translateX(4px); box-shadow: var(--sh1); }
.cm-ico  { width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0; }
.cm-lbl  { font-size: .67rem; color: var(--txt3); margin-bottom: .1rem; }
.cm-val  { font-size: .83rem; font-weight: 600; color: var(--txt); }
.page-hdr{ display: flex; align-items: flex-start; gap: 1.25rem; margin-bottom: 1.5rem; }
.hdr-ico { width: 50px; height: 50px; flex-shrink: 0; background: linear-gradient(135deg, var(--green-g), rgba(16,185,129,.12)); border: 1.5px solid rgba(12,155,88,.3); border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 1.35rem; box-shadow: 0 2px 12px rgba(12,155,88,.12); }
.hdr-h   { font-family: var(--ff-display) !important; font-size: 1.75rem; color: var(--txt); margin: 0 0 .2rem; line-height: 1.2; }
.hdr-p   { font-size: .82rem; color: var(--txt3); margin: 0; line-height: 1.6; }
.lbl     { font-family: var(--ff-mono); font-size: .55rem; letter-spacing: .2em; text-transform: uppercase; color: var(--txt3); margin-bottom: .55rem; padding-left: 1px; }
.notice  { display: flex; gap: .8rem; background: rgba(12,155,88,.06); border: 1px solid rgba(12,155,88,.2); border-left: 3px solid var(--green); border-radius: var(--r2); padding: .75rem 1rem; margin-bottom: 1.2rem; font-size: .78rem; color: var(--txt2); line-height: 1.65; }
.notice.err{ border-left-color: var(--rose); background: rgba(232,72,85,.05); border-color: rgba(232,72,85,.2); }
.notice code{ background: var(--surf3); padding: 2px 6px; border-radius: 4px; font-family: var(--ff-mono); font-size: .76em; color: var(--green); }
.disclaimer{ background: rgba(217,119,6,.05); border: 1px solid rgba(217,119,6,.18); border-left: 3px solid var(--amber); border-radius: var(--r2); padding: .8rem 1.1rem; font-size: .76rem; color: var(--txt2); line-height: 1.7; }
.summary-g { display: grid; grid-template-columns: repeat(auto-fill,minmax(175px,1fr)); gap: .55rem; margin-bottom: 1.25rem; }
.sum-item  { background: var(--surf2); border: 1px solid var(--bord); border-radius: var(--r2); padding: .65rem .9rem; transition: border-color .2s; }
.sum-item:hover{ border-color: var(--green); }
.sum-lbl   { font-size: .65rem; color: var(--txt3); margin-bottom: .15rem; }
.sum-val   { font-family: var(--ff-mono); font-size: .84rem; color: var(--txt); font-weight: 600; }
.dropdown  { background: var(--surf); border: 1px solid var(--bord2); border-radius: var(--r); padding: .5rem; min-width: 260px; box-shadow: var(--sh3); animation: fadeUp .2s ease both; }
.dropdown-item { display: flex; align-items: center; gap: .75rem; padding: .6rem .75rem; border-radius: var(--r3); font-size: .82rem; color: var(--txt2); transition: all .15s; }
.dropdown-item:hover { background: var(--surf2); color: var(--txt); }
.dropdown-sep { height: 1px; background: var(--bord); margin: .3rem 0; }
.dropdown-header { font-family: var(--ff-mono); font-size: .54rem; letter-spacing: .15em; text-transform: uppercase; color: var(--txt3); padding: .4rem .75rem .2rem; }
.notif-unread { background: var(--green-g); border-left: 2px solid var(--green); }
.notif-dot    { width: 7px; height: 7px; border-radius: 50%; background: var(--green); flex-shrink: 0; box-shadow: 0 0 6px rgba(12,155,88,.5); }
.hist-item    { display: flex; align-items: center; gap: 1rem; background: var(--surf); border: 1px solid var(--bord); border-radius: var(--r2); padding: .9rem 1.25rem; margin-bottom: .5rem; transition: all .2s; box-shadow: var(--sh1); }
.hist-item:hover { border-color: var(--green); transform: translateX(4px); box-shadow: var(--sh2); }
.hist-risk-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.hist-risk-dot.high { background: var(--rose);  box-shadow: 0 0 8px rgba(232,72,85,.5); }
.hist-risk-dot.low  { background: var(--green); box-shadow: 0 0 8px rgba(12,155,88,.5); }
.hist-info    { flex: 1; }
.hist-name    { font-size: .84rem; font-weight: 600; color: var(--txt); }
.hist-meta    { font-size: .7rem; color: var(--txt3); font-family: var(--ff-mono); margin-top: .1rem; }
.hist-badge   { font-family: var(--ff-mono); font-size: .62rem; font-weight: 600; letter-spacing: .08em; padding: 3px 10px; border-radius: 20px; }
.hist-badge.high { background: rgba(232,72,85,.10); color: var(--rose);  border: 1px solid rgba(232,72,85,.25); }
.hist-badge.low  { background: rgba(12,155,88,.10);  color: var(--green); border: 1px solid rgba(12,155,88,.2);  }
.hist-score   { font-family: var(--ff-display); font-size: 1.4rem; }
.hist-score.high { color: var(--rose);  }
.hist-score.low  { color: var(--green); }
.cal-grid { display: grid; grid-template-columns: repeat(7,1fr); gap: 4px; margin: .75rem 0; }
.cal-dh   { text-align: center; font-family: var(--ff-mono); font-size: .57rem; letter-spacing: .08em; text-transform: uppercase; color: var(--txt3); padding: 5px 0; font-weight: 600; }
.cal-day  { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; border-radius: var(--r3); font-size: .78rem; color: var(--txt2); transition: all .15s; font-family: var(--ff-mono); cursor: pointer; border: 1px solid transparent; }
.cal-day:hover  { background: var(--green-g); color: var(--green); border-color: rgba(12,155,88,.3); }
.cal-day.today  { background: var(--green); color: #fff; font-weight: 700; box-shadow: 0 2px 10px rgba(12,155,88,.35); }
.cal-day.has-appt{ background: rgba(8,145,178,.10); color: var(--teal); border-color: rgba(8,145,178,.3); font-weight: 600; }
.cal-day.empty  { cursor: default; opacity: 0; pointer-events: none; }
.appt-row { display: flex; align-items: center; gap: .75rem; background: var(--surf2); border: 1px solid var(--bord); border-radius: var(--r2); padding: .7rem 1rem; margin-bottom: .4rem; transition: all .2s; }
.appt-row:hover { border-color: var(--green); box-shadow: var(--sh1); }
.appt-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px rgba(12,155,88,.6); flex-shrink: 0; }
.about-hero { background: linear-gradient(135deg, #edfaf3, #f0fdf9); border: 1px solid var(--bord); border-radius: var(--r); padding: 2.5rem; margin-bottom: 1.25rem; position: relative; overflow: hidden; }
.about-hero::before { content: ''; position: absolute; top: 0; right: 0; bottom: 0; width: 45%; background: radial-gradient(ellipse at right center, rgba(12,155,88,.08), transparent 70%); }
hr { border-color: var(--bord) !important; }

/* ── Home hero ────────────────────────────────────── */
.hero2 { position: relative; border-radius: 24px; overflow: hidden; border: 1.5px solid rgba(12,155,88,.25); margin-bottom: 1.25rem; min-height: 330px; background: linear-gradient(135deg, #f0fdf7 0%, #e8f7ef 50%, #f4fdf9 100%); box-shadow: 0 8px 40px rgba(10,60,30,.10); }
.hero2-grid { position: absolute; inset: 0; background-image: linear-gradient(rgba(12,155,88,.06) 1px,transparent 1px), linear-gradient(90deg,rgba(12,155,88,.06) 1px,transparent 1px); background-size: 40px 40px; }
.hero2-glow-l { position: absolute; top: -100px; left: -80px; width: 500px; height: 500px; border-radius: 50%; background: radial-gradient(circle, rgba(12,155,88,.10) 0%, transparent 65%); }
.hero2-glow-r { position: absolute; bottom: -80px; right: 0; width: 400px; height: 400px; border-radius: 50%; background: radial-gradient(circle, rgba(8,145,178,.08) 0%, transparent 65%); }
.hero2-ecg { position: absolute; bottom: 0; left: 0; right: 0; height: 45px; opacity: .22; background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 50'%3E%3Cpolyline points='0,25 200,25 230,5 245,45 260,5 275,45 300,25 500,25 530,5 545,45 560,5 575,45 600,25 800,25 830,5 845,45 860,5 875,45 900,25 1100,25 1130,5 1145,45 1160,5 1175,45 1200,25 1440,25' fill='none' stroke='%230c9b58' stroke-width='2'/%3E%3C/svg%3E") repeat-x; animation: ecg 5s linear infinite; }
.hero2-inner { position: relative; z-index: 2; padding: 2.5rem 3rem; display: grid; grid-template-columns: 1fr 280px; gap: 3rem; align-items: center; }
.hero2-badge { display: inline-flex; align-items: center; gap: .5rem; font-family: var(--ff-mono); font-size: .57rem; letter-spacing: .24em; text-transform: uppercase; color: var(--green); background: rgba(12,155,88,.10); border: 1px solid rgba(12,155,88,.3); padding: 5px 16px; border-radius: 20px; margin-bottom: 1.1rem; }
.hero2-badge-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px rgba(12,155,88,.6); animation: pulse2 2s ease infinite; }
.hero2-title { font-family: var(--ff-display); font-size: 3rem; color: var(--txt); line-height: 1.1; letter-spacing: -.01em; margin-bottom: .75rem; }
.hero2-title .accent-teal { color: var(--green); font-style: italic; }
.hero2-sub   { font-size: .88rem; color: var(--txt2); max-width: 460px; line-height: 1.85; }
.hero2-visual { position: relative; display: flex; align-items: center; justify-content: center; }
.hero2-orb    { position: absolute; inset: 0; margin: auto; width: 220px; height: 220px; border-radius: 50%; background: radial-gradient(circle, rgba(12,155,88,.10) 0%, transparent 70%); border: 1px solid rgba(12,155,88,.15); }
.hero2-orb2   { position: absolute; inset: -20px; margin: auto; width: 260px; height: 260px; border-radius: 50%; border: 1px solid rgba(12,155,88,.08); animation: spin2 20s linear infinite; }
.hero2-img    { width: 180px; height: 180px; border-radius: 50%; object-fit: cover; position: relative; z-index: 1; border: 3px solid rgba(12,155,88,.3); box-shadow: 0 0 30px rgba(12,155,88,.18), 0 12px 40px rgba(10,60,30,.12); }
.hero2-img-tag{ position: absolute; bottom: -8px; left: 50%; transform: translateX(-50%); background: rgba(255,255,255,.96); backdrop-filter: blur(12px); border: 1px solid rgba(12,155,88,.35); border-radius: 20px; padding: 4px 14px; font-size: .67rem; color: var(--green); font-family: var(--ff-mono); white-space: nowrap; font-weight: 700; z-index: 2; box-shadow: 0 2px 12px rgba(12,155,88,.15); }
.hero2-float-card { position: absolute; background: rgba(255,255,255,.93); backdrop-filter: blur(12px); border: 1px solid rgba(12,155,88,.2); border-radius: 12px; padding: .6rem .9rem; z-index: 3; box-shadow: 0 4px 18px rgba(10,60,30,.12); }
.hero2-float-card.tl { top: 10px;    left: -15px;  animation: floatCard 4s ease-in-out infinite; }
.hero2-float-card.br { bottom: 10px; right: -15px; animation: floatCard 4s ease-in-out 2s infinite; }
.hfc-val { font-family: var(--ff-display); font-size: 1.3rem; color: var(--txt); }
.hfc-lbl { font-size: .62rem; color: var(--txt3); }

/* ── Stat cards ───────────────────────────────────── */
.stats2 { display: grid; grid-template-columns: repeat(4,1fr); gap: .85rem; margin-bottom: 1.25rem; }
.stat2  { position: relative; border-radius: 16px; padding: 1.4rem 1.5rem; overflow: hidden; border: 1.5px solid var(--bord); background: var(--surf); transition: all .28s cubic-bezier(.22,1,.36,1); cursor: pointer; box-shadow: var(--sh1); animation: cardEnter .5s ease both; }
.stat2:hover { transform: translateY(-7px); box-shadow: var(--sh3); }
.stat2::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 16px 16px 0 0; }
.stat2-bg   { position: absolute; bottom: -20px; right: -20px; width: 90px; height: 90px; border-radius: 50%; opacity: .08; }
.stat2-ico-wrap { width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; margin-bottom: .9rem; position: relative; z-index: 1; }
.stat2-val  { font-family: var(--ff-display); font-size: 2.4rem; line-height: 1; margin-bottom: .2rem; position: relative; z-index: 1; }
.stat2-lbl  { font-size: .7rem; color: var(--txt3); position: relative; z-index: 1; font-weight: 500; }
.stat2-trend{ position: absolute; top: 1rem; right: 1rem; font-family: var(--ff-mono); font-size: .58rem; padding: 2px 8px; border-radius: 20px; z-index: 1; font-weight: 600; }
.stat2.blue::before   { background: linear-gradient(90deg, var(--green),  var(--emerald)); }
.stat2.blue .stat2-val{ color: var(--green); }
.stat2.blue .stat2-ico-wrap{ background: rgba(12,155,88,.09); border: 1px solid rgba(12,155,88,.2); }
.stat2.blue .stat2-trend   { background: rgba(12,155,88,.09); color: var(--green); border: 1px solid rgba(12,155,88,.2); }
.stat2.teal::before   { background: linear-gradient(90deg, var(--teal),   var(--green)); }
.stat2.teal .stat2-val{ color: var(--teal); }
.stat2.teal .stat2-ico-wrap{ background: rgba(8,145,178,.09); border: 1px solid rgba(8,145,178,.2); }
.stat2.teal .stat2-trend   { background: rgba(8,145,178,.09); color: var(--teal); border: 1px solid rgba(8,145,178,.2); }
.stat2.rose::before   { background: linear-gradient(90deg, var(--rose),   #f97316); }
.stat2.rose .stat2-val{ color: var(--rose); }
.stat2.rose .stat2-ico-wrap{ background: rgba(232,72,85,.08); border: 1px solid rgba(232,72,85,.2); }
.stat2.rose .stat2-trend   { background: rgba(232,72,85,.08); color: var(--rose); border: 1px solid rgba(232,72,85,.2); }
.stat2.amber::before  { background: linear-gradient(90deg, var(--amber),  #f59e0b); }
.stat2.amber .stat2-val{ color: var(--amber); }
.stat2.amber .stat2-ico-wrap{ background: rgba(217,119,6,.09); border: 1px solid rgba(217,119,6,.2); }
.stat2.amber .stat2-trend   { background: rgba(217,119,6,.09); color: var(--amber); border: 1px solid rgba(217,119,6,.2); }

/* ── Feature cards ────────────────────────────────── */
.feats2  { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.feat2   { position: relative; border-radius: 16px; padding: 1.5rem; overflow: hidden; background: var(--surf); border: 1.5px solid var(--bord); transition: all .28s cubic-bezier(.22,1,.36,1); cursor: pointer; box-shadow: var(--sh1); }
.feat2:hover { border-color: var(--green); transform: translateY(-5px); box-shadow: var(--sh3); }
.feat2::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; transform: scaleX(0); transform-origin: left; transition: transform .35s; }
.feat2:hover::before { transform: scaleX(1); }
.feat2.f-blue::before  { background: linear-gradient(90deg, var(--green),  var(--teal));   }
.feat2.f-teal::before  { background: linear-gradient(90deg, var(--teal),   var(--emerald));}
.feat2.f-purple::before{ background: linear-gradient(90deg, var(--purple), var(--teal));   }
.feat2.f-amber::before { background: linear-gradient(90deg, var(--amber),  #f59e0b);       }
.feat2.f-rose::before  { background: linear-gradient(90deg, var(--rose),   var(--purple)); }
.feat2.f-green::before { background: linear-gradient(90deg, var(--emerald),var(--green));  }
.feat2-num  { font-family: var(--ff-mono); font-size: .54rem; letter-spacing: .2em; text-transform: uppercase; color: var(--txt3); margin-bottom: .9rem; }
.feat2-ico  { font-size: 1.9rem; margin-bottom: .75rem; display: block; }
.feat2-title{ font-family: var(--ff-display); font-size: 1rem; color: var(--txt); margin-bottom: .5rem; line-height: 1.3; }
.feat2-desc { font-size: .76rem; color: var(--txt3); line-height: 1.7; }
.feat2-link { position: absolute; top: 1.1rem; right: 1.1rem; width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: .8rem; background: var(--surf2); border: 1px solid var(--bord); color: var(--txt3); transition: all .2s; }
.feat2:hover .feat2-link { background: var(--green-g); border-color: rgba(12,155,88,.3); color: var(--green); }

/* ── CTA buttons ──────────────────────────────────── */
.btn-primary .stButton > button {
  background: linear-gradient(135deg, var(--green2), var(--green), var(--emerald)) !important;
  color: #fff !important; border: none !important; font-weight: 700 !important;
  box-shadow: 0 4px 18px rgba(12,155,88,.35) !important;
  padding: .78rem 1.5rem !important; font-size: .88rem !important; width: 100% !important;
  animation: glow 2.5s ease infinite !important;
}
.btn-primary .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(12,155,88,.50) !important; }
.btn-ghost2  .stButton > button {
  background: var(--surf) !important; color: var(--txt2) !important;
  border: 1.5px solid var(--bord) !important; box-shadow: var(--sh1) !important;
  padding: .78rem 1.5rem !important; font-size: .88rem !important; width: 100% !important;
}
.btn-ghost2 .stButton > button:hover { background: var(--surf2) !important; color: var(--green) !important; border-color: var(--green) !important; transform: translateY(-1px) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def goto(page):
    st.session_state.page = page
    st.session_state.show_notif = False
    st.session_state.show_profile_menu = False
    st.rerun()

def unread_count():
    return sum(1 for n in st.session_state.notifications if not n["read"])

# ── Field renderers ────────────────────────────────────────────────
def render_number(label, key, mn, mx, default, step, unit=""):
    saved = st.session_state.input_values.get(key, default)
    if isinstance(step, float):
        saved = float(saved); mn, mx = float(mn), float(mx)
    else:
        saved = int(saved); mn, mx, step = int(mn), int(mx), int(step)
    unit_str = f" ({unit})" if unit else ""
    # Label row
    st.markdown(
        f'<div class="field-row">'
        f'<div class="field-label">{label}<small>Range: {mn}–{mx}{unit_str}</small></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    # Input rendered natively (not inside HTML div — avoids broken DOM)
    val = st.number_input(
        label, min_value=mn, max_value=mx, value=saved, step=step,
        key=f"ni_{key}", label_visibility="collapsed",
    )
    return val

def render_yesno(label, key, default=0):
    saved     = st.session_state.input_values.get(key, default)
    saved_idx = 1 if int(saved) == 1 else 0
    col_lbl, col_radio = st.columns([3, 2])
    with col_lbl:
        st.markdown(f'<div class="yn-label" style="padding-top:.55rem;">{label}</div>', unsafe_allow_html=True)
    with col_radio:
        choice = st.radio(
            label, options=["No", "Yes"], index=saved_idx,
            key=f"yn_{key}", horizontal=True, label_visibility="collapsed",
        )
    return 1 if choice == "Yes" else 0

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
def render_sidebar():
    u  = st.session_state.get("user_info", {})
    pg = st.session_state.page
    with st.sidebar:
        st.markdown('<div class="sb-logo">Cerv<em>AI</em></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sb-user">'
            f'<div class="sb-avatar">{u.get("initials","?")}</div>'
            f'<div><div class="sb-name">{u.get("name","User")}</div>'
            f'<div class="sb-role">{u.get("role","Clinician")}</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
        for pid, ico, lbl in [
            ("home",       "🏠", "Dashboard"),
            ("classifier", "📋", "Patient Classifier"),
            ("history",    "🕐", "History"),
            ("calendar",   "📅", "Calendar"),
        ]:
            cls = "nav-btn-active" if pg == pid else "nav-btn"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(f"{ico}  {lbl}", key=f"sb_{pid}"): goto(pid)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Info</div>', unsafe_allow_html=True)
        for pid, ico, lbl in [("about", "ℹ️", "About"), ("contact", "✉️", "Contact Us"), ("guide", "📖", "Medical Guide")]:
            cls = "nav-btn-active" if pg == pid else "nav-btn"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(f"{ico}  {lbl}", key=f"sb_{pid}"): goto(pid)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Account</div>', unsafe_allow_html=True)
        for pid, ico, lbl in [("profile", "👤", "My Profile"), ("settings", "⚙️", "Settings")]:
            cls = "nav-btn-active" if pg == pid else "nav-btn"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(f"{ico}  {lbl}", key=f"sb_{pid}"): goto(pid)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sb-status"><div class="sb-dot"></div>'
            f'{"Model Active" if model else "Demo Mode · No Model"}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
        if st.button("🚪  Sign Out", key="sb_logout"):
            st.session_state.logged_in  = False
            st.session_state.page       = "login"
            st.session_state.user_info  = {}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════
def render_topbar():
    pg = st.session_state.page
    ico_lbl = {
        "home": "🏠 Dashboard", "classifier": "📋 Classifier",
        "history": "🕐 History",  "calendar": "📅 Calendar",
        "about": "ℹ️ About",      "contact": "✉️ Contact Us",
        "profile": "👤 Profile",   "settings": "⚙️ Settings",
        "results": "📊 Results",   "guide": "📖 Medical Guide",
    }.get(pg, "🔬 Page")

    u  = st.session_state.get("user_info", {})
    nc = unread_count()

    c_left, c_notif, c_search, c_prof = st.columns([6, 0.7, 0.7, 0.7])
    with c_left:
        st.markdown(
            f'<div class="topbar">'
            f'<div class="topbar-logo">Cerv<em>AI</em></div>'
            f'<div class="topbar-sep"></div>'
            f'<div class="topbar-page">{ico_lbl}</div>'
            f'<div class="topbar-right"><div class="tb-pill">Research Use Only</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c_notif:
        badge_html = f'<div class="tb-badge">{nc}</div>' if nc else ""
        # Streamlit button — the HTML div is decorative only
        if st.button("🔔", key="tb_notif", help="Notifications"):
            st.session_state.show_notif        = not st.session_state.show_notif
            st.session_state.show_profile_menu = False
            st.rerun()
    with c_search:
        if st.button("🔍", key="tb_search", help="Go to Classifier"):
            goto("classifier")
    with c_prof:
        if st.button(u.get("initials", "?"), key="tb_avatar", help="Profile menu"):
            st.session_state.show_profile_menu = not st.session_state.show_profile_menu
            st.session_state.show_notif        = False
            st.rerun()

    # ── Notification dropdown ──────────────────────────────────────
    if st.session_state.show_notif:
        st.markdown('<div class="dropdown" style="max-width:320px;">', unsafe_allow_html=True)
        st.markdown('<div class="dropdown-header">Notifications</div>', unsafe_allow_html=True)
        for n in st.session_state.notifications:
            cls  = "notif-unread" if not n["read"] else ""
            dot  = '<div class="notif-dot"></div>' if not n["read"] else '<div style="width:7px"></div>'
            st.markdown(
                f'<div class="dropdown-item {cls}">{dot}'
                f'<div style="flex:1;">'
                f'<div style="font-size:.8rem;color:var(--txt);">{n["icon"]} {n["text"]}</div>'
                f'<div style="font-size:.65rem;color:var(--txt3);font-family:var(--ff-mono);">{n["time"]}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("✓ Mark all read", key="mark_read"):
            for n in st.session_state.notifications:
                n["read"] = True
            st.session_state.show_notif = False
            st.rerun()

    # ── Profile dropdown ───────────────────────────────────────────
    if st.session_state.show_profile_menu:
        u2 = st.session_state.get("user_info", {})
        st.markdown(
            f'<div class="dropdown">'
            f'<div style="display:flex;align-items:center;gap:.75rem;padding:.75rem;background:var(--surf3);border-radius:var(--r3);margin-bottom:.35rem;">'
            f'<div class="sb-avatar" style="width:36px;height:36px;font-size:.85rem;">{u2.get("initials","?")}</div>'
            f'<div><div style="font-size:.82rem;font-weight:600;color:var(--txt);">{u2.get("name","User")}</div>'
            f'<div style="font-size:.67rem;color:var(--txt3);">{u2.get("email","")}</div></div>'
            f'</div><div class="dropdown-sep"></div></div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("👤 Profile",  key="pm_profile"):  goto("profile")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("⚙️ Settings", key="pm_settings"): goto("settings")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="pm_logout"):
            st.session_state.logged_in = False
            st.session_state.page      = "login"
            st.session_state.user_info = {}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE · LOGIN
# ══════════════════════════════════════════════════════════════════
def page_login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, cc, _ = st.columns([1, 1.4, 1])
    with cc:
        st.markdown(
            '<div style="text-align:center;margin-bottom:1.75rem;">'
            '<div style="font-family:\'DM Serif Display\',serif;font-size:3rem;color:var(--txt);letter-spacing:-.02em;line-height:1;">Cerv<span style="color:var(--green);font-style:italic;">AI</span></div>'
            '<div style="font-family:var(--ff-mono);font-size:.58rem;letter-spacing:.25em;text-transform:uppercase;color:var(--txt3);margin-top:.4rem;">Clinical Cervical Cancer Risk Platform</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        # Card
        st.markdown(
            '<div style="background:var(--surf);border:1.5px solid var(--bord2);border-radius:20px;padding:2rem 2rem 1.75rem;box-shadow:var(--sh2);animation:fadeUp .5s cubic-bezier(.22,1,.36,1) .1s both;">'
            '<div style="font-family:var(--ff-display);font-size:1.15rem;color:var(--txt);margin-bottom:1.5rem;">🔐 Sign in to your account</div>',
            unsafe_allow_html=True,
        )
        email = st.text_input("Email Address", placeholder="you@example.com", key="li_email")
        pwd   = st.text_input("Password",       type="password", placeholder="••••••••",    key="li_pwd")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In  →", key="li_btn"):
            user = check_login(email, pwd)
            if user:
                st.session_state.logged_in  = True
                st.session_state.user_info  = {**user, "email": email.strip().lower()}
                st.session_state.page       = "home"
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please check your email and password.")
        st.markdown('</div>', unsafe_allow_html=True)   # close card

        st.markdown(
            '<div style="background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);'
            'padding:.75rem 1rem;margin-top:1rem;font-family:var(--ff-mono);font-size:.67rem;color:var(--txt3);line-height:2;">'
            '<strong style="color:var(--txt2);">Demo accounts</strong><br>'
            'bakraouladomor@gmail.com / cerv2024<br>'
            'yassirjbili@gmail.com / cerv2024<br>'
            'ilyaselhadad@gmail.com / cerv2024<br>'
            'mohammedelhadad@gmail.com / cerv2024<br>'
            'yahyaelomari@gmail.com / cerv2024<br>'
            'admin@cerv.ai / admin123'
            '</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════
# PAGE · HOME
# ══════════════════════════════════════════════════════════════════
def page_home():
    u          = st.session_state.get("user_info", {})
    total_hist = len(st.session_state.history)
    high_risk  = sum(1 for h in st.session_state.history if h["pred"] == 1)
    low_risk   = total_hist - high_risk
    today      = datetime.date.today()
    upcoming   = sum(
        1 for a in st.session_state.appointments
        if datetime.datetime.strptime(a["date"], "%Y-%m-%d").date() >= today
    )

    # ── Hero ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="hero2 au">'
        f'<div class="hero2-grid"></div>'
        f'<div class="hero2-glow-l"></div><div class="hero2-glow-r"></div>'
        f'<div class="hero2-ecg"></div>'
        f'<div class="hero2-inner">'
        f'  <div>'
        f'    <div class="hero2-badge"><span class="hero2-badge-dot"></span>AI-Powered · Clinical Grade · Research Platform</div>'
        f'    <div class="hero2-title">Cervical Cancer<br>Risk <span class="accent-teal">Classifier</span></div>'
        f'    <div class="hero2-sub">Welcome back, <strong style="color:var(--txt);">{u.get("name","Doctor")}</strong>.<br>'
        f'      Evidence-informed biopsy risk prediction trained on <strong style="color:var(--txt);">858 clinical records</strong> across <strong style="color:var(--txt);">36 risk factors</strong>.</div>'
        f'  </div>'
        f'  <div class="hero2-visual">'
        f'    <div class="hero2-orb"></div>'
        f'    <div class="hero2-orb2"></div>'
        f'    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Papilloma_Virus_%28HPV%29_EM.jpg/480px-Papilloma_Virus_%28HPV%29_EM.jpg" class="hero2-img" alt="HPV"/>'
        f'    <div class="hero2-img-tag">🔬 HPV · Primary Causal Agent</div>'
        f'    <div class="hero2-float-card tl"><div class="hfc-val" style="color:var(--green);">{total_hist}</div><div class="hfc-lbl">Assessments</div></div>'
        f'    <div class="hero2-float-card br"><div class="hfc-val" style="color:var(--teal);">99%</div><div class="hfc-lbl">HPV-linked</div></div>'
        f'  </div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── CTA buttons ────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("🔬  Run Assessment", key="h_assess"): goto("classifier")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-ghost2">', unsafe_allow_html=True)
        if st.button("🕐  View History",   key="h_hist"):   goto("history")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="btn-ghost2">', unsafe_allow_html=True)
        if st.button("📅  Calendar",       key="h_cal"):    goto("calendar")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stat cards ─────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4, gap="small")
    stats = [
        (s1, "blue",  "🔬", total_hist, "Assessments Run",  "All time"),
        (s2, "teal",  "✅", low_risk,   "Low Risk Results",  "✓ Safe"),
        (s3, "rose",  "⚠️", high_risk,  "High Risk Flagged", "⚠ Review"),
        (s4, "amber", "📅", upcoming,   "Appointments",      "Upcoming"),
    ]
    for col, cls, ico, val, lbl, trend in stats:
        with col:
            st.markdown(
                f'<div class="stat2 {cls}">'
                f'<div class="stat2-bg"></div>'
                f'<div class="stat2-trend">{trend}</div>'
                f'<div class="stat2-ico-wrap">{ico}</div>'
                f'<div class="stat2-val">{val}</div>'
                f'<div class="stat2-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature cards (clickable) ─────────────────────────────────
    FEAT_DATA = [
        ("f-blue",   "01 · Profiling",  "📋", "Comprehensive Patient Profiling",
         "5 clinical modules · 36 risk factors covering demographics, STDs, contraceptives and prior diagnoses.",
         "classifier"),
        ("f-teal",   "02 · Engine",     "⚡", "Instant AI Prediction",
         "XGBoost + SMOTE oversampling. Sub-second inference with probability scoring.",
         "classifier"),
        ("f-purple", "03 · Insight",    "📊", "Explainable AI Results",
         "Feature importance charts reveal which clinical factors drove each prediction.",
         "results"),
        ("f-amber",  "04 · Audit",      "🕐", "Full Assessment History",
         "Every prediction logged with patient data, risk scores and timestamps.",
         "history"),
        ("f-rose",   "05 · Schedule",   "📅", "Appointment Calendar",
         "Schedule follow-ups, biopsies and consultations with the interactive calendar.",
         "calendar"),
        ("f-green",  "06 · Privacy",    "🔒", "Secure & Private",
         "Zero data persistence. All inputs processed in-memory — cleared on session end.",
         "about"),
    ]
    row1 = st.columns(3, gap="medium")
    row2 = st.columns(3, gap="medium")
    for idx, (cls, num, ico, title, desc, target) in enumerate(FEAT_DATA):
        col = row1[idx] if idx < 3 else row2[idx-3]
        with col:
            safe_title = title.replace("&","&amp;")
            st.markdown(
                f'<div class="feat2 {cls}" style="cursor:default;">'
                f'<div class="feat2-num">{num}</div>'
                f'<span class="feat2-ico">{ico}</span>'
                f'<div class="feat2-title">{safe_title}</div>'
                f'<div class="feat2-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if target == "results" and not st.session_state.prediction:
                st.markdown('<div class="ghost">', unsafe_allow_html=True)
                if st.button(f"→ {title}", key=f"feat_{idx}"): goto("classifier")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ghost">', unsafe_allow_html=True)
                if st.button(f"→ Open", key=f"feat_{idx}"): goto(target)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer au4"><strong>⚕ Research &amp; Educational Use Only.</strong> '
        'CervAI is a machine learning prototype. It is <em>not a certified medical device</em>.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# PAGE · CLASSIFIER
# ══════════════════════════════════════════════════════════════════
def page_classifier():
    # ── Result overlay ─────────────────────────────────────────────
    if st.session_state.get("show_result_overlay") and st.session_state.prediction:
        p        = st.session_state.prediction
        pred     = p["pred"]; pct = p["pct"]
        accent   = "var(--rose)"  if pred == 1 else "var(--teal)"
        verdict  = "Biopsy Indicated"  if pred == 1 else "Biopsy Not Indicated"
        badge_t  = "⚠ HIGH RISK"       if pred == 1 else "✓ LOW RISK"
        detail   = (
            "The model predicts a <strong>positive biopsy result</strong>. Warrants immediate specialist referral."
            if pred == 1 else
            "The model predicts a <strong>negative biopsy result</strong>. Continue routine screening per guidelines."
        )
        badge_bg  = "rgba(232,72,85,.12)" if pred == 1 else "rgba(12,155,88,.10)"
        badge_bd  = "rgba(232,72,85,.30)" if pred == 1 else "rgba(12,155,88,.25)"
        bar_grad  = "linear-gradient(90deg,#ff8fa0,#cc1133)" if pred == 1 else "linear-gradient(90deg,#7ee8be,#22c55e)"
        ring_cls  = "ring-fill-high" if pred == 1 else "ring-fill-low"
        offset    = round(339 * (1 - pct / 100), 1)
        top_col   = "rgba(232,72,85,.08)" if pred == 1 else "rgba(12,155,88,.08)"
        bord_c    = "rgba(232,72,85,.30)" if pred == 1 else "rgba(12,155,88,.30)"
        top_bg    = "linear-gradient(90deg,var(--rose),#cc1133)" if pred==1 else "linear-gradient(90deg,var(--teal),#22c55e)"

        st.markdown(
            f'<div style="background:linear-gradient(135deg,{top_col},rgba(246,250,247,0.7));'
            f'border:1.5px solid {bord_c};border-radius:20px;max-width:720px;margin:0 auto 1.5rem;'
            f'animation:fadeUp .4s ease both;overflow:hidden;box-shadow:var(--sh3);">'
            f'<div style="height:3px;background:{top_bg};"></div>'
            f'<div style="padding:2rem 2rem 0;">'
            f'  <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1.25rem;flex-wrap:wrap;">'
            f'    <span style="font-family:var(--ff-mono);font-size:.58rem;letter-spacing:.2em;text-transform:uppercase;'
            f'          color:{accent};padding:4px 14px;border-radius:20px;background:{badge_bg};border:1px solid {badge_bd};">{badge_t}</span>'
            f'    <span style="font-family:var(--ff-mono);font-size:.65rem;color:var(--txt3);">'
            f'      {p.get("patient_id","—")} &nbsp;·&nbsp; {p.get("timestamp","—")}</span>'
            f'  </div>'
            f'  <div style="display:grid;grid-template-columns:1fr 140px;gap:2rem;align-items:center;">'
            f'    <div>'
            f'      <div style="font-family:var(--ff-display);font-size:2.2rem;color:{accent};line-height:1.1;margin-bottom:.5rem;">{verdict}</div>'
            f'      <div style="font-size:.84rem;color:var(--txt2);line-height:1.75;">{detail}</div>'
            f'    </div>'
            f'    <div style="position:relative;width:140px;height:140px;flex-shrink:0;">'
            f'      <svg class="ring-svg" width="140" height="140" viewBox="0 0 120 120">'
            f'        <circle class="ring-track" cx="60" cy="60" r="54"/>'
            f'        <circle class="{ring_cls}" cx="60" cy="60" r="54" style="stroke-dashoffset:{offset};"/>'
            f'      </svg>'
            f'      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">'
            f'        <div style="font-family:var(--ff-display);font-size:2.6rem;line-height:1;color:{accent};">{pct}%</div>'
            f'        <div style="font-family:var(--ff-mono);font-size:.52rem;letter-spacing:.14em;text-transform:uppercase;color:var(--txt3);">Risk Score</div>'
            f'      </div>'
            f'    </div>'
            f'  </div>'
            f'</div>'
            f'<div style="padding:1.25rem 2rem 2rem;">'
            f'  <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem;">'
            f'    <div style="font-family:var(--ff-mono);font-size:.57rem;letter-spacing:.14em;text-transform:uppercase;color:var(--txt3);margin-bottom:.5rem;">Probability</div>'
            f'    <div style="background:var(--surf4);border-radius:100px;height:10px;overflow:hidden;">'
            f'      <div style="width:{pct}%;height:100%;border-radius:100px;background:{bar_grad};"></div>'
            f'    </div>'
            f'    <div style="display:flex;justify-content:space-between;font-family:var(--ff-mono);font-size:.63rem;color:var(--txt3);margin-top:.3rem;">'
            f'      <span>0%</span><span style="color:{accent};font-weight:600;">{pct}% predicted</span><span>100%</span>'
            f'    </div>'
            f'  </div>'
            f'  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-bottom:1rem;">'
            f'    <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:.8rem;text-align:center;">'
            f'      <div style="font-family:var(--ff-mono);font-size:.54rem;color:var(--txt3);margin-bottom:.25rem;">MODEL</div>'
            f'      <div style="font-size:.82rem;font-weight:600;color:var(--txt);">XGBoost</div>'
            f'    </div>'
            f'    <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:.8rem;text-align:center;">'
            f'      <div style="font-family:var(--ff-mono);font-size:.54rem;color:var(--txt3);margin-bottom:.25rem;">AGE</div>'
            f'      <div style="font-size:.82rem;font-weight:600;color:var(--txt);">{p.get("age","?")} yrs</div>'
            f'    </div>'
            f'    <div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:.8rem;text-align:center;">'
            f'      <div style="font-family:var(--ff-mono);font-size:.54rem;color:var(--txt3);margin-bottom:.25rem;">VERDICT</div>'
            f'      <div style="font-size:.82rem;font-weight:600;color:{accent};">{"POSITIVE" if pred==1 else "NEGATIVE"}</div>'
            f'    </div>'
            f'  </div>'
            f'  <div style="background:rgba(217,119,6,.04);border:1px solid rgba(217,119,6,.15);'
            f'       border-left:3px solid var(--amber);border-radius:10px;padding:.7rem 1rem;'
            f'       font-size:.73rem;color:var(--txt2);line-height:1.6;">'
            f'    <strong style="color:var(--amber);">⚕ Clinical Disclaimer:</strong> AI prediction only. Consult a qualified provider.'
            f'  </div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="text-align:center;font-family:var(--ff-mono);font-size:.58rem;letter-spacing:.15em;'
            'text-transform:uppercase;color:var(--txt3);margin-bottom:.75rem;">— Choose an action —</p>',
            unsafe_allow_html=True,
        )
        b1, b2, b3 = st.columns(3, gap="medium")
        with b1:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("← Return to Form", key="ov_close"):
                st.session_state.show_result_overlay = False; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("📊  Full Report",   key="ov_full"):
                st.session_state.show_result_overlay = False; goto("results")
            st.markdown('</div>', unsafe_allow_html=True)
        with b3:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("🔄  New Assessment", key="ov_new"):
                st.session_state.show_result_overlay = False
                st.session_state.prediction   = None
                st.session_state.input_values = {}
                for k in [k2 for k2 in st.session_state.keys() if k2.startswith(("ni_", "yn_"))]:
                    del st.session_state[k]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Normal form ────────────────────────────────────────────────
    st.markdown(
        '<div class="page-hdr au">'
        '<div class="hdr-ico">📋</div>'
        '<div><div class="hdr-h">Patient Risk Profile</div>'
        '<p class="hdr-p">Fill in the patient\'s clinical data below, then run the prediction.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    c_pid, c_gender, c_guide_btn, c_status = st.columns([2, 1.2, 1, 2.5])
    with c_pid:
        pat_id = st.text_input("Patient ID / Reference", placeholder="e.g. PAT-2024-0042", key="pat_id_field")
    with c_gender:
        st.markdown('<div style="padding-top:.1rem;">', unsafe_allow_html=True)
        gender = st.radio(
            "Patient gender", options=["Female", "Male"],
            index=0 if st.session_state.get("patient_gender","Female")=="Female" else 1,
            key="gender_radio", horizontal=True,
        )
        st.session_state["patient_gender"] = gender
        st.markdown('</div>', unsafe_allow_html=True)
    with c_guide_btn:
        st.markdown('<div style="padding-top:1.55rem;" class="ghost">', unsafe_allow_html=True)
        if st.button("📖  Medical Guide", key="open_guide_main"):
            goto("guide")
        st.markdown('</div>', unsafe_allow_html=True)
    with c_status:
        if model is None:
            st.markdown(
                '<div class="notice err" style="margin-top:1.7rem;">'
                '<span>⚠️</span><div><strong>Model not loaded.</strong> Place <code>xgboost_model.joblib</code> in <code>models/</code>.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="notice" style="margin-top:1.7rem;">'
                f'<span>✅</span><div>Model ready · <code>{model_path}</code></div></div>',
                unsafe_allow_html=True,
            )

    total_f = 36
    filled  = sum(1 for v in st.session_state.input_values.values() if v not in (0, 0.0))
    pct_p   = int(filled / total_f * 100)
    st.markdown(
        f'<div class="prog-wrap">'
        f'<div class="prog-lbl">Profile Completion</div>'
        f'<div class="prog-bar"><div class="prog-fill" style="width:{pct_p}%;"></div></div>'
        f'<div class="prog-pct">{pct_p}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    input_values = {}
    col1, col2   = st.columns(2, gap="large")

    with col1:
        # Module 1 — Demographics
        st.markdown(
            '<div class="section-card">'
            '<div class="sc-stripe sc-blue"></div>'
            '<div class="section-eyebrow">👤 · Module 01</div>'
            '<div class="section-title">Demographics <span class="section-subtitle">Patient background</span></div>'
            '<div class="section-desc">Age, sexual history and reproductive background.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        input_values["Age"]                     = render_number("Age",                      "Age",                     13,   84,   25,   1,   "yrs")
        input_values["Num_sexual_partners"]      = render_number("Lifetime sexual partners", "Num_sexual_partners",     0,    28,   2,    1)
        input_values["First_sexual_intercourse"] = render_number("Age at first intercourse", "First_sexual_intercourse",10,   32,   17,   1,   "yrs")
        input_values["Num_of_pregnancies"]       = render_number("Number of pregnancies",    "Num_of_pregnancies",      0,    11,   1,    1)
        st.markdown("<br>", unsafe_allow_html=True)

        # Module 2 — Smoking
        st.markdown(
            '<div class="section-card">'
            '<div class="sc-stripe sc-smoke"></div>'
            '<div class="section-eyebrow">🚬 · Module 02</div>'
            '<div class="section-title">Smoking History <span class="section-subtitle">Tobacco exposure</span></div>'
            '<div class="section-desc">Smoking is a significant co-factor in HPV persistence.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        input_values["Smokes"]            = render_yesno("Smoker (current or former)",   "Smokes")
        input_values["Smokes_years"]      = render_number("Smoking duration (years)",    "Smokes_years",      0.0, 37.0, 0.0, 0.5)
        input_values["Smokes_packs_year"] = render_number("Intensity (packs/year)",      "Smokes_packs_year", 0.0, 40.0, 0.0, 0.5)
        st.markdown("<br>", unsafe_allow_html=True)

        # Module 5 — Diagnoses
        sc3, sg3 = st.columns([6,1])
        with sc3:
            st.markdown(
                '<div class="section-card">'
                '<div class="sc-stripe sc-dx"></div>'
                '<div class="section-eyebrow">🔬 · Module 05</div>'
                '<div class="section-title">Prior Diagnoses &amp; Tests</div>'
                '<div class="section-desc">Previous diagnoses and gynecological screening results.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with sg3:
            st.markdown('<div class="ghost" style="padding-top:1.2rem;">', unsafe_allow_html=True)
            if st.button("📖 Guide", key="guide_dx"): goto("guide")
            st.markdown('</div>', unsafe_allow_html=True)
        for lbl, key in [
            ("Prior cancer diagnosis",   "Dx_Cancer"),
            ("Prior CIN diagnosis",       "Dx_CIN"),
            ("Prior HPV diagnosis",       "Dx_HPV"),
            ("General diagnosis flag",    "Dx"),
            ("Hinselmann test positive",  "Hinselmann"),
            ("Schiller test positive",    "Schiller"),
            ("Cytology positive",         "Citology"),
        ]:
            input_values[key] = render_yesno(lbl, key)

    with col2:
        # Module 3 — Contraceptives
        sc4, sg4 = st.columns([6,1])
        with sc4:
            st.markdown(
                '<div class="section-card">'
                '<div class="sc-stripe sc-contra"></div>'
                '<div class="section-eyebrow">💊 · Module 03</div>'
                '<div class="section-title">Contraceptive Use <span class="section-subtitle">Hormonal &amp; IUD</span></div>'
                '<div class="section-desc">Long-term hormonal use is associated with increased cervical cancer risk.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with sg4:
            st.markdown('<div class="ghost" style="padding-top:1.2rem;">', unsafe_allow_html=True)
            if st.button("📖 Guide", key="guide_contra"): goto("guide")
            st.markdown('</div>', unsafe_allow_html=True)
        input_values["Hormonal_Contraceptives"]       = render_yesno("Using hormonal contraceptives", "Hormonal_Contraceptives")
        input_values["Hormonal_Contraceptives_years"] = render_number("Duration of use (years)",      "Hormonal_Contraceptives_years", 0.0, 30.0, 0.0, 0.5)
        input_values["IUD"]                           = render_yesno("IUD in use or history",          "IUD")
        input_values["IUD_years"]                     = render_number("IUD duration (years)",          "IUD_years",                    0.0, 19.0, 0.0, 0.5)
        st.markdown("<br>", unsafe_allow_html=True)

        # Module 4 — STDs
        sc2, sg2 = st.columns([6,1])
        with sc2:
            st.markdown(
                '<div class="section-card">'
                '<div class="sc-stripe sc-std"></div>'
                '<div class="section-eyebrow">🦠 · Module 04</div>'
                '<div class="section-title">STD History <span class="section-subtitle">Sexually transmitted infections</span></div>'
                '<div class="section-desc">STD co-infections modulate HPV persistence and dysplasia progression.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with sg2:
            st.markdown('<div class="ghost" style="padding-top:1.2rem;">', unsafe_allow_html=True)
            if st.button("📖 Guide", key="guide_std"): goto("guide")
            st.markdown('</div>', unsafe_allow_html=True)
        input_values["STDs"]        = render_yesno("History of STD(s)",          "STDs")
        input_values["STDs_number"] = render_number("Number of distinct STDs",   "STDs_number",               0, 4, 0, 1)
        with st.expander("🔽  Specific STD types", expanded=False):
            for lbl, key in [
                ("Condylomatosis",               "STDs_condylomatosis"),
                ("Cervical condylomatosis",       "STDs_cervical_condylomatosis"),
                ("Vaginal condylomatosis",        "STDs_vaginal_condylomatosis"),
                ("Vulvo-perineal condylomatosis", "STDs_vulvo_perineal_condylomatosis"),
                ("Syphilis",                      "STDs_syphilis"),
                ("Pelvic inflammatory disease",   "STDs_pelvic_inflammatory_disease"),
                ("Genital herpes",                "STDs_genital_herpes"),
                ("Molluscum contagiosum",         "STDs_molluscum_contagiosum"),
                ("AIDS",                          "STDs_AIDS"),
                ("HIV",                           "STDs_HIV"),
                ("Hepatitis B",                   "STDs_Hepatitis_B"),
                ("HPV",                           "STDs_HPV"),
            ]:
                input_values[key] = render_yesno(lbl, key)
        input_values["STDs_Number_of_diagnosis"]        = render_number("Total STD diagnoses",          "STDs_Number_of_diagnosis",        0,   3,   0,   1)
        input_values["STDs_Time_since_first_diagnosis"] = render_number("Years since first STD dx",     "STDs_Time_since_first_diagnosis",  0.0, 29.0,0.0, 0.5)
        input_values["STDs_Time_since_last_diagnosis"]  = render_number("Years since last STD dx",      "STDs_Time_since_last_diagnosis",   0.0, 29.0,0.0, 0.5)

    # Ensure all STD sub-keys are present (even if expander wasn't opened)
    for key in STD_KEYS:
        if key not in input_values:
            input_values[key] = st.session_state.input_values.get(key, 0)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(12,155,88,.06),rgba(123,111,255,.04));'
        'border:1px solid rgba(12,155,88,.14);border-radius:16px;padding:1.4rem;text-align:center;">'
        '<div style="font-family:var(--ff-display);font-size:1.1rem;color:var(--txt);margin-bottom:.3rem;">Ready to Run Assessment?</div>'
        '<div style="font-size:.8rem;color:var(--txt2);margin-bottom:.25rem;">Review the profile above, then click below.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _, bcol, _ = st.columns([1, 2, 1])
    with bcol:
        st.markdown('<div class="run-btn-wrap">', unsafe_allow_html=True)
        run_clicked = st.button("🔬  Run Biopsy Risk Prediction", disabled=(model is None), key="predict_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer" style="margin-top:1rem;">'
        '<strong>⚕ Research Use Only.</strong> Does not replace clinical judgment.</div>',
        unsafe_allow_html=True,
    )

    if run_clicked:
        st.session_state.input_values = dict(input_values)
        row = {KEY_TO_COL[k]: v for k, v in input_values.items() if k in KEY_TO_COL}
        df  = pd.DataFrame([row])

        with st.expander("🔍 Debug — Values sent to model", expanded=False):
            non_zero = {k: v for k, v in row.items() if v not in (0, 0.0)}
            st.write("**Non-zero inputs:**", non_zero)

        # Pre-processing
        try:
            tc  = assets["columns"]
            imp = assets["imputer"]
            sc  = assets["scaler"]
            df_aligned = df.reindex(columns=tc, fill_value=0)
            df_imp     = pd.DataFrame(imp.transform(df_aligned), columns=tc)
            df_scaled  = pd.DataFrame(sc.transform(df_imp),      columns=tc)
            df         = df_scaled
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
                fn      = list(assets.get("columns", COLUMN_ORDER))[:len(raw_imps)]
                top_n   = 12
                top_idx = np.argsort(raw_imps)[::-1][:top_n]
                fi = {
                    "names": [fn[i] for i in top_idx][::-1],
                    "vals":  [float(raw_imps[i]) for i in top_idx][::-1],
                    "mean":  float(np.mean(raw_imps)),
                }
        except Exception:
            fi = None

        new_id = st.session_state.last_prediction_id + 1
        entry  = {
            "id":         new_id,
            "pred":       pred,
            "prob":       prob,
            "pct":        pct,
            "feature_importance": fi,
            "input_summary": {KEY_TO_COL[k]: v for k, v in input_values.items()
                              if v not in (0, 0.0) and k in KEY_TO_COL},
            "patient_id": (
                pat_id.strip()
                or f"PAT-{datetime.date.today().strftime('%Y%m%d')}-{len(st.session_state.history)+1:03d}"
            ),
            "timestamp":  datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
            "age":        input_values.get("Age", "?"),
            "gender":     st.session_state.get("patient_gender", "Female"),
        }
        st.session_state.prediction          = entry
        st.session_state.last_prediction_id  = new_id
        st.session_state.history.append(entry)
        st.session_state.show_result_overlay = True
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# PAGE · RESULTS
# ══════════════════════════════════════════════════════════════════
def page_results():
    p = st.session_state.prediction
    if not p:
        st.warning("No prediction found. Run an assessment first.")
        if st.button("← Go to Classifier"): goto("classifier")
        return

    pred     = p["pred"]; pct = p["pct"]
    accent   = "var(--rose)"  if pred == 1 else "var(--teal)"
    verdict  = "Biopsy Indicated"  if pred == 1 else "Biopsy Unlikely"
    badge_t  = "⚠ High Risk"       if pred == 1 else "✓ Low Risk"
    detail   = (
        "The model predicts a <strong>positive biopsy result</strong>. Warrants further evaluation."
        if pred == 1 else
        "The model predicts a <strong>negative biopsy result</strong>. Continue routine screening."
    )
    offset   = round(339 * (1 - pct / 100), 1)
    ring_cls = "ring-fill-high" if pred == 1 else "ring-fill-low"
    bg_grad  = "rgba(232,72,85,.07)" if pred==1 else "rgba(12,155,88,.07)"
    bord_c   = "rgba(232,72,85,.30)" if pred==1 else "rgba(12,155,88,.30)"
    top_bg   = "linear-gradient(90deg,var(--rose),#cc1133)"   if pred==1 else "linear-gradient(90deg,var(--teal),#22c55e)"
    bar_grad = "linear-gradient(90deg,#ff8fa0,#cc1133)"       if pred==1 else "linear-gradient(90deg,#7ee8be,#22c55e)"
    badge_bg = "rgba(232,72,85,.12)" if pred==1 else "rgba(12,155,88,.10)"
    badge_bd = "rgba(232,72,85,.25)" if pred==1 else "rgba(12,155,88,.2)"

    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(
            f'<div class="hdr-h au">Prediction Results</div>'
            f'<p style="font-size:.78rem;color:var(--txt3);margin-bottom:1.5rem;">'
            f'{p.get("patient_id","—")} &nbsp;·&nbsp; {p.get("timestamp","—")}</p>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("← Edit", key="res_back"): goto("classifier")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="background:linear-gradient(135deg,{bg_grad},rgba(246,250,247,0.8));'
        f'border:1.5px solid {bord_c};border-radius:18px;padding:2.25rem;margin-bottom:1.5rem;'
        f'position:relative;overflow:hidden;animation:fadeUp .5s ease both;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;border-radius:18px 18px 0 0;background:{top_bg};"></div>'
        f'<div style="display:grid;grid-template-columns:1fr auto;gap:2rem;align-items:center;">'
        f'  <div>'
        f'    <div style="display:inline-flex;align-items:center;font-size:.67rem;font-weight:600;letter-spacing:.1em;'
        f'         text-transform:uppercase;padding:4px 12px;border-radius:20px;margin-bottom:.8rem;'
        f'         font-family:var(--ff-mono);color:{accent};background:{badge_bg};border:1px solid {badge_bd};">{badge_t}</div>'
        f'    <div style="font-family:var(--ff-display);font-size:2rem;line-height:1.15;margin-bottom:.5rem;color:{accent};">{verdict}</div>'
        f'    <div style="font-size:.84rem;color:var(--txt2);line-height:1.75;margin-bottom:1.4rem;">{detail}</div>'
        f'    <div style="background:var(--surf3);border-radius:100px;height:9px;overflow:hidden;border:1px solid var(--bord);">'
        f'      <div style="width:{pct}%;height:100%;border-radius:100px;background:{bar_grad};"></div>'
        f'    </div>'
        f'    <div style="display:flex;justify-content:space-between;font-family:var(--ff-mono);font-size:.65rem;color:var(--txt3);margin-top:.3rem;">'
        f'      <span>Predicted biopsy probability</span><span style="font-weight:600;">{pct}%</span>'
        f'    </div>'
        f'  </div>'
        f'  <div style="position:relative;width:130px;height:130px;flex-shrink:0;">'
        f'    <svg class="ring-svg" width="130" height="130" viewBox="0 0 120 120">'
        f'      <circle class="ring-track" cx="60" cy="60" r="54"/>'
        f'      <circle class="{ring_cls}" cx="60" cy="60" r="54" style="stroke-dashoffset:{offset};"/>'
        f'    </svg>'
        f'    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">'
        f'      <div style="font-family:var(--ff-display);font-size:2.4rem;line-height:1;color:{accent};">{pct}%</div>'
        f'      <div style="font-family:var(--ff-mono);font-size:.52rem;letter-spacing:.14em;text-transform:uppercase;color:var(--txt3);">Risk Score</div>'
        f'    </div>'
        f'  </div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    summary = {k: v for k, v in p.get("input_summary", {}).items() if v not in (0, 0.0)}
    if summary:
        st.markdown('<p class="lbl">Notable Input Values</p>', unsafe_allow_html=True)
        html = '<div class="summary-g">'
        for k, v in list(summary.items())[:12]:
            html += f'<div class="sum-item"><div class="sum-lbl">{k}</div><div class="sum-val">{v}</div></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # ── Feature importance chart ───────────────────────────────────
    fi = p.get("feature_importance")
    if fi and fi.get("vals"):
        try:
            import matplotlib
            matplotlib.use("Agg")
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
            fig.patch.set_facecolor("#f6faf7")
            ax.set_facecolor("#f6faf7")

            for i, (val, nv) in enumerate(zip(vals, norm)):
                color = tuple(LOW_C + nv * (HIGH_C - LOW_C))
                ax.barh(i, max_val, height=0.56, color="#e8f5ed", edgecolor="none", zorder=1)
                ax.barh(i, val,     height=0.56, color=color,      edgecolor="none", zorder=3, alpha=0.9)
                ax.text(val + max_val * 0.018, i, f"{val:.3f}",
                        va="center", ha="left", fontsize=8.5, color="#2d6649", fontfamily="monospace")

            ax.set_yticks(range(n))
            ax.set_yticklabels(names, fontsize=9.5, color="#2d6649")
            ax.tick_params(axis="y", length=0, pad=6)
            ax.tick_params(axis="x", colors="#cce4d6", labelsize=8, labelcolor="#6b9e80")
            ax.set_xlim(0, max_val * 1.30)
            ax.set_ylim(-0.7, n - 0.3)
            for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#cce4d6")
            ax.xaxis.grid(True, color="#d9eee3", linewidth=0.7, linestyle="--", zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel("Feature Importance Score", fontsize=9, color="#6b9e80", labelpad=10, fontfamily="monospace")
            ax.set_title(
                "Top Contributing Features — Ranked by Predictive Strength",
                fontsize=11.5, color="#0e2e1e", pad=14, loc="left",
            )
            legend_handles = [
                mpatches.Patch(color=tuple(LOW_C),  alpha=0.9, label="Lower importance"),
                mpatches.Patch(color=tuple(HIGH_C), alpha=0.9, label="Highest importance"),
            ]
            ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
                      frameon=True, framealpha=0.9, edgecolor="#cce4d6",
                      facecolor="#ffffff", labelcolor="#2d6649")
            plt.tight_layout(pad=1.6)
            st.markdown('<p class="lbl" style="margin-top:.5rem;">Feature importance analysis</p>', unsafe_allow_html=True)
            st.caption("Bars show each feature's contribution to the prediction. Longer bar = stronger influence. Blue → rose gradient shows relative rank.")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.info(f"Chart unavailable: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("← Edit Profile",    key="r_edit"): goto("classifier")
        st.markdown('</div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("🕐 View History",   key="r_hist"): goto("history")
        st.markdown('</div>', unsafe_allow_html=True)
    with a3:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("📖 Medical Guide",  key="r_guide"): goto("guide")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer">'
        '<strong>⚕ Clinical Disclaimer:</strong> AI prediction only. Not a substitute for professional medical judgment.'
        '</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# PAGE · HISTORY
# ══════════════════════════════════════════════════════════════════
def page_history():
    st.markdown(
        '<div class="page-hdr au"><div class="hdr-ico">🕐</div><div>'
        '<div class="hdr-h">Assessment History</div>'
        '<p class="hdr-p">Complete log of all biopsy risk predictions.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    hist = st.session_state.history
    if not hist:
        st.markdown(
            '<div class="card" style="text-align:center;padding:3rem;">'
            '<div style="font-size:3rem;margin-bottom:1rem;">📭</div>'
            '<div style="font-family:var(--ff-display);font-size:1.1rem;color:var(--txt);margin-bottom:.5rem;">No assessments yet</div>'
            '<div style="font-size:.82rem;color:var(--txt3);">Run your first prediction to see results here.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Run First Assessment", key="hist_cta"): goto("classifier")
        return

    total = len(hist)
    highs = sum(1 for h in hist if h["pred"] == 1)
    avg   = np.mean([h["pct"] for h in hist])
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.9rem;margin-bottom:1.5rem;">'
        f'<div class="card-sm"><div class="lbl">Total</div><div style="font-family:var(--ff-display);font-size:1.8rem;color:var(--green);">{total}</div></div>'
        f'<div class="card-sm"><div class="lbl">High Risk</div><div style="font-family:var(--ff-display);font-size:1.8rem;color:var(--rose);">{highs}</div></div>'
        f'<div class="card-sm"><div class="lbl">Avg Score</div><div style="font-family:var(--ff-display);font-size:1.8rem;color:var(--amber);">{avg:.1f}%</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    f1, f2, _ = st.columns([1, 1, 3])
    with f1:
        filt = st.selectbox("Filter", ["All", "High Risk", "Low Risk"], key="hist_filter")
    with f2:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("🗑 Clear All", key="hist_clear"):
            st.session_state.history = []; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    filtered = [
        h for h in reversed(hist)
        if filt == "All"
        or (filt == "High Risk" and h["pred"] == 1)
        or (filt == "Low Risk"  and h["pred"] == 0)
    ]
    for i, h in enumerate(filtered):
        rc2 = "high" if h["pred"] == 1 else "low"
        c1, c2 = st.columns([9, 1])
        with c1:
            st.markdown(
                f'<div class="hist-item">'
                f'<div class="hist-risk-dot {rc2}"></div>'
                f'<div class="hist-info">'
                f'  <div class="hist-name">{h.get("patient_id","—")}</div>'
                f'  <div class="hist-meta">{h.get("timestamp","—")} · Age: {h.get("age","?")}</div>'
                f'</div>'
                f'<div class="hist-badge {rc2}">{"⚠ High" if rc2=="high" else "✓ Low"}</div>'
                f'<div class="hist-score {rc2}">{h["pct"]}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("View", key=f"hv_{i}"):
                st.session_state.prediction = h; goto("results")

# ══════════════════════════════════════════════════════════════════
# PAGE · CALENDAR
# ══════════════════════════════════════════════════════════════════
def page_calendar():
    st.markdown(
        '<div class="page-hdr au"><div class="hdr-ico">📅</div><div>'
        '<div class="hdr-h">Appointment Calendar</div>'
        '<p class="hdr-p">Schedule and manage patient appointments.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    today = datetime.date.today()
    y, m  = st.session_state.cal_y, st.session_state.cal_m

    month_name = datetime.date(y, m, 1).strftime("%B %Y")
    appt_days  = {
        datetime.datetime.strptime(a["date"], "%Y-%m-%d").date().day
        for a in st.session_state.appointments
        if datetime.datetime.strptime(a["date"], "%Y-%m-%d").date().year  == y
        and datetime.datetime.strptime(a["date"], "%Y-%m-%d").date().month == m
    }

    col_cal, col_form = st.columns([1.3, 1])

    with col_cal:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        n1, n2, n3 = st.columns([1, 4, 1])
        with n1:
            if st.button("‹", key="cp"):
                if m == 1: st.session_state.cal_m = 12; st.session_state.cal_y -= 1
                else:      st.session_state.cal_m -= 1
                st.rerun()
        with n2:
            st.markdown(
                f'<div style="text-align:center;font-family:var(--ff-display);font-size:1rem;padding:.35rem 0;">{month_name}</div>',
                unsafe_allow_html=True,
            )
        with n3:
            if st.button("›", key="cn"):
                if m == 12: st.session_state.cal_m = 1; st.session_state.cal_y += 1
                else:       st.session_state.cal_m += 1
                st.rerun()

        cal_html = '<div class="cal-grid">'
        for dh in ["M", "T", "W", "T", "F", "S", "S"]:
            cal_html += f'<div class="cal-dh">{dh}</div>'
        for week in calendar.monthcalendar(y, m):
            for day in week:
                if day == 0:
                    cal_html += '<div class="cal-day empty"></div>'
                else:
                    cls = "cal-day"
                    if day == today.day and m == today.month and y == today.year:
                        cls += " today"
                    if day in appt_days:
                        cls += " has-appt"
                    cal_html += f'<div class="{cls}">{day}</div>'
        cal_html += '</div>'
        st.markdown(cal_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:var(--ff-display);font-size:.95rem;color:var(--txt);margin-bottom:1rem;">➕ New Appointment</div>',
            unsafe_allow_html=True,
        )
        pn = st.text_input("Patient Name", placeholder="Full name", key="ap_n")
        at = st.selectbox("Type", [
            "Initial Consultation", "Follow-up", "Biopsy Review",
            "Screening", "Post-Treatment", "Emergency",
        ], key="ap_t")
        ad = st.date_input("Date",  value=today, key="ap_d")
        ah = st.selectbox("Time", [
            "08:00","08:30","09:00","09:30","10:00","10:30",
            "11:00","11:30","14:00","14:30","15:00","15:30","16:00","17:00",
        ], key="ap_h")
        an = st.text_input("Notes", placeholder="Optional", key="ap_note")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("📅 Schedule Appointment", key="ap_add"):
            if pn.strip():
                st.session_state.appointments.append({
                    "patient": pn.strip(), "type": at,
                    "date": ad.strftime("%Y-%m-%d"), "time": ah, "note": an,
                })
                st.success(f"✅ Scheduled for {pn.strip()}")
                st.rerun()
            else:
                st.error("Patient name is required.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="lbl">Upcoming Appointments</p>', unsafe_allow_html=True)
    upcoming = sorted(
        [a for a in st.session_state.appointments
         if datetime.datetime.strptime(a["date"], "%Y-%m-%d").date() >= today],
        key=lambda x: x["date"] + x["time"],
    )
    if not upcoming:
        st.markdown(
            '<div class="card" style="text-align:center;padding:1.5rem;color:var(--txt3);font-size:.82rem;">No upcoming appointments.</div>',
            unsafe_allow_html=True,
        )
    else:
        for i, a in enumerate(upcoming):
            d    = datetime.datetime.strptime(a["date"], "%Y-%m-%d").date()
            dstr = d.strftime("%a, %d %b %Y")
            c1, c2 = st.columns([9, 1])
            with c1:
                st.markdown(
                    f'<div class="appt-row"><div class="appt-dot"></div>'
                    f'<div style="flex:1;">'
                    f'  <div style="font-size:.83rem;font-weight:500;color:var(--txt);">{a["patient"]}</div>'
                    f'  <div style="font-size:.7rem;color:var(--txt3);font-family:var(--ff-mono);">{a["type"]} · {dstr} at {a["time"]}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown('<div class="ghost">', unsafe_allow_html=True)
                if st.button("🗑", key=f"da_{i}"):
                    st.session_state.appointments.remove(a); st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE · ABOUT
# ══════════════════════════════════════════════════════════════════
def page_about():
    st.markdown(
        '<div class="page-hdr au"><div class="hdr-ico">ℹ️</div><div>'
        '<div class="hdr-h">About CervAI</div>'
        '<p class="hdr-p">Mission, methodology, and the team behind the platform.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="about-hero au">'
        '<div style="font-family:var(--ff-display);font-size:2rem;color:var(--txt);margin-bottom:.75rem;position:relative;z-index:1;">'
        'Advancing Cervical Cancer<br><em style="color:var(--green);">Early Detection</em> with AI</div>'
        '<p style="font-size:.86rem;color:var(--txt2);max-width:580px;line-height:1.85;position:relative;z-index:1;">'
        'CervAI bridges advanced machine learning and gynecological oncology, built during Coding Week 2026 at Centrale Casablanca.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown(
            '<div class="card au2" style="margin-bottom:1rem;">'
            '<div style="font-family:var(--ff-display);font-size:1rem;color:var(--txt);margin-bottom:1rem;">🧬 The Dataset</div>'
            '<p style="font-size:.82rem;color:var(--txt2);line-height:1.8;">UCI Cervical Cancer (Risk Factors) Dataset — 858 patients, 36 clinical attributes.</p>'
            '<div class="metric-grid">'
            '<div class="metric-box"><div class="metric-v">858</div><div class="metric-l">Patients</div></div>'
            '<div class="metric-box"><div class="metric-v">36</div><div class="metric-l">Features</div></div>'
            '<div class="metric-box"><div class="metric-v">4</div><div class="metric-l">Target Tests</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card au3">'
            '<div style="font-family:var(--ff-display);font-size:1rem;color:var(--txt);margin-bottom:1rem;">⚙️ Technology Stack</div>'
            '<div>'
            '<span class="tech-pill">🐍 Python 3.11</span>'
            '<span class="tech-pill">🎈 Streamlit</span>'
            '<span class="tech-pill">🚀 XGBoost</span>'
            '<span class="tech-pill">⚖️ SMOTE</span>'
            '<span class="tech-pill">🔢 NumPy</span>'
            '<span class="tech-pill">🐼 Pandas</span>'
            '<span class="tech-pill">🎯 Scikit-learn</span>'
            '<span class="tech-pill">📊 Matplotlib</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="card au2">'
            '<div style="font-family:var(--ff-display);font-size:1rem;color:var(--txt);margin-bottom:1rem;">👥 Research Team — Coding Week 2026</div>'
            '<div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#1a3a5c,#0e2240);">👨‍⚕️</div>'
            '<div><div class="team-name">Bakr Aoulad Omar</div><div class="team-role">Lead Researcher · Clinical Advisor</div></div></div>'
            '<div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#1a2a50,#0a1830);">👨‍💻</div>'
            '<div><div class="team-name">Yassir Jbili</div><div class="team-role">ML Engineer · Model Architecture</div></div></div>'
            '<div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#1a1a40,#0a0a28);">🔬</div>'
            '<div><div class="team-name">Ilyas El Hadad</div><div class="team-role">Data Scientist · Feature Engineering</div></div></div>'
            '<div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#102a30,#081820);">📊</div>'
            '<div><div class="team-name">Mohamed El Hadad</div><div class="team-role">Biostatistics · Validation</div></div></div>'
            '<div class="team-card"><div class="team-av" style="background:linear-gradient(135deg,#2a1a40,#180a28);">🏥</div>'
            '<div><div class="team-name">Yahya El Omari</div><div class="team-role">Clinical Review · QA</div></div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════
# PAGE · CONTACT
# ══════════════════════════════════════════════════════════════════
def page_contact():
    st.markdown(
        '<div class="page-hdr au"><div class="hdr-ico">✉️</div><div>'
        '<div class="hdr-h">Contact Us</div>'
        '<p class="hdr-p">Reach our team for support or research inquiries.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown('<div class="card au">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:var(--ff-display);font-size:.95rem;color:var(--txt);margin-bottom:1.25rem;">📬 Send Us a Message</div>',
            unsafe_allow_html=True,
        )
        cn  = st.text_input("Full Name",     placeholder="Your name",         key="ct_n")
        ce  = st.text_input("Your Email",    placeholder="you@example.com",   key="ct_e")
        cto = st.selectbox("Contact",  ["Bakr Aoulad Omar","Yassir Jbili","Ilyas El Hadad","Mohamed El Hadad","Yahya El Omari"], key="ct_to")
        cs  = st.selectbox("Subject",  ["Technical Support","Clinical Query","Research Collaboration","Bug Report","General Inquiry"], key="ct_s")
        cm  = st.text_area("Message",  placeholder="Describe your inquiry…", height=130, key="ct_m")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📨  Send Message", key="ct_send"):
            if cn.strip() and ce.strip() and cm.strip():
                st.success("✅ Message sent! We'll respond within 48 hours.")
            else:
                st.error("Please fill in name, email, and message.")
    with col2:
        st.markdown(
            '<div class="card au2">'
            '<div style="font-family:var(--ff-display);font-size:.95rem;color:var(--txt);margin-bottom:1.1rem;">👥 Team Contacts</div>'
            '<div class="contact-method"><div class="cm-ico" style="background:rgba(12,155,88,.10);border:1px solid rgba(12,155,88,.2);">👨‍⚕️</div>'
            '<div><div class="cm-lbl">Lead Researcher</div><div class="cm-val">Bakr Aoulad Omar</div>'
            '<div style="font-size:.7rem;color:var(--green);font-family:var(--ff-mono);">bakraouladomor@gmail.com</div></div></div>'
            '<div class="contact-method"><div class="cm-ico" style="background:rgba(8,145,178,.10);border:1px solid rgba(8,145,178,.2);">👨‍💻</div>'
            '<div><div class="cm-lbl">ML Engineer</div><div class="cm-val">Yassir Jbili</div>'
            '<div style="font-size:.7rem;color:var(--green);font-family:var(--ff-mono);">yassirjbili@gmail.com</div></div></div>'
            '<div class="contact-method"><div class="cm-ico" style="background:rgba(124,58,237,.10);border:1px solid rgba(124,58,237,.2);">🔬</div>'
            '<div><div class="cm-lbl">Data Scientist</div><div class="cm-val">Ilyas El Hadad</div>'
            '<div style="font-size:.7rem;color:var(--green);font-family:var(--ff-mono);">ilyaselhadad@gmail.com</div></div></div>'
            '<div class="contact-method"><div class="cm-ico" style="background:rgba(217,119,6,.10);border:1px solid rgba(217,119,6,.2);">📊</div>'
            '<div><div class="cm-lbl">Biostatistics</div><div class="cm-val">Mohamed El Hadad</div>'
            '<div style="font-size:.7rem;color:var(--green);font-family:var(--ff-mono);">mohammedelhadad@gmail.com</div></div></div>'
            '<div class="contact-method"><div class="cm-ico" style="background:rgba(232,72,85,.10);border:1px solid rgba(232,72,85,.2);">🏥</div>'
            '<div><div class="cm-lbl">Clinical Review</div><div class="cm-val">Yahya El Omari</div>'
            '<div style="font-size:.7rem;color:var(--green);font-family:var(--ff-mono);">yahyaelomari@gmail.com</div></div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════
# PAGE · PROFILE
# ══════════════════════════════════════════════════════════════════
def page_profile():
    u = st.session_state.get("user_info", {})
    st.markdown(
        '<div class="page-hdr au"><div class="hdr-ico">👤</div><div>'
        '<div class="hdr-h">My Profile</div>'
        '<p class="hdr-p">Manage your account information.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown(
            f'<div class="card au" style="text-align:center;padding:2rem;">'
            f'<div style="width:80px;height:80px;border-radius:50%;background:linear-gradient(135deg,var(--green),var(--teal));'
            f'display:flex;align-items:center;justify-content:center;font-size:1.8rem;font-weight:700;'
            f'color:#fff;margin:0 auto 1rem;border:3px solid var(--bord2);">{u.get("initials","?")}</div>'
            f'<div style="font-family:var(--ff-display);font-size:1.2rem;color:var(--txt);">{u.get("name","User")}</div>'
            f'<div style="font-size:.76rem;color:var(--txt3);font-family:var(--ff-mono);margin-top:.25rem;">{u.get("role","—")} · {u.get("dept","—")}</div>'
            f'<div style="font-size:.76rem;color:var(--txt3);margin-top:.5rem;">{u.get("email","—")}</div>'
            f'<div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--bord);display:grid;grid-template-columns:1fr 1fr;gap:.5rem;">'
            f'<div style="background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:.75rem;">'
            f'<div style="font-family:var(--ff-display);font-size:1.4rem;color:var(--green);">{len(st.session_state.history)}</div>'
            f'<div style="font-size:.65rem;color:var(--txt3);">Assessments</div></div>'
            f'<div style="background:var(--surf2);border:1px solid var(--bord);border-radius:var(--r2);padding:.75rem;">'
            f'<div style="font-family:var(--ff-display);font-size:1.4rem;color:var(--teal);">{len(st.session_state.appointments)}</div>'
            f'<div style="font-size:.65rem;color:var(--txt3);">Appointments</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown('<div class="card au2">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:var(--ff-display);font-size:.95rem;color:var(--txt);margin-bottom:1.1rem;">✏️ Edit Profile</div>',
            unsafe_allow_html=True,
        )
        new_name = st.text_input("Full Name",   value=u.get("name", ""),    key="prof_name")
        st.text_input("Email",                  value=u.get("email", ""),   disabled=True, key="prof_email")
        new_dept = st.text_input("Department",  value=u.get("dept", ""),    key="prof_dept")
        role_opts = ["Oncologist","Gynecologist","Radiologist","Researcher","Administrator","Lead Researcher","ML Engineer","Data Scientist","Biostatistics","Clinical Review"]
        cur_role  = u.get("role", "Researcher")
        role_idx  = role_opts.index(cur_role) if cur_role in role_opts else 0
        new_role  = st.selectbox("Role", role_opts, index=role_idx, key="prof_role_sel")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Changes", key="prof_save"):
            st.session_state.user_info["name"] = new_name
            st.session_state.user_info["dept"] = new_dept
            st.session_state.user_info["role"] = new_role
            st.session_state.user_info["initials"] = (
                "".join(w[0].upper() for w in new_name.split()[:2]) if new_name else u.get("initials", "?")
            )
            st.success("✅ Profile updated!")
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE · SETTINGS
# ══════════════════════════════════════════════════════════════════
def page_settings():
    st.markdown(
        '<div class="page-hdr au"><div class="hdr-ico">⚙️</div><div>'
        '<div class="hdr-h">Settings</div>'
        '<p class="hdr-p">Configure your platform preferences.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    CARD  = "background:var(--surf);border:1px solid var(--bord);border-radius:14px;padding:1.5rem 1.6rem;margin-bottom:1rem;box-shadow:var(--sh1);"
    TITLE = "font-family:var(--ff-display);font-size:.95rem;color:var(--txt);margin-bottom:1.1rem;"

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(f'<div style="{CARD}"><div style="{TITLE}">🔬 Classifier</div></div>', unsafe_allow_html=True)
        autosave = st.checkbox("Auto-save inputs",                  value=True,  key="s_autosave")
        show_fi  = st.checkbox("Show feature importance chart",     value=True,  key="s_fi")
        thresh   = st.selectbox(
            "Risk threshold",
            ["50% (Standard)", "40% (Sensitive)", "60% (Specific)"],
            key="s_thresh",
        )
        tv = {"50% (Standard)": 50, "40% (Sensitive)": 40, "60% (Specific)": 60}.get(thresh, 50)
        st.markdown(
            f'<div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;'
            f'padding:.7rem 1rem;margin-top:.5rem;display:flex;align-items:center;gap:.75rem;">'
            f'<div style="flex:1;background:var(--surf4);border-radius:100px;height:5px;">'
            f'<div style="width:{tv}%;height:100%;border-radius:100px;background:linear-gradient(90deg,var(--green),var(--teal));"></div>'
            f'</div>'
            f'<span style="font-family:var(--ff-mono);font-size:.72rem;color:var(--green);font-weight:600;">{tv}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(f'<div style="{CARD}"><div style="{TITLE}">🔔 Notifications</div></div>', unsafe_allow_html=True)
        email_on  = st.checkbox("Email alerts for high-risk patients", value=True,  key="s_email")
        appt_on   = st.checkbox("Appointment reminders",               value=True,  key="s_appt_rem")
        weekly_on = st.checkbox("Weekly summary report",               value=False, key="s_weekly")

        def badge(active, label):
            if active:
                return (f'<span style="background:rgba(12,155,88,.10);color:var(--green);'
                        f'border:1px solid rgba(12,155,88,.25);border-radius:20px;padding:2px 10px;'
                        f'font-family:var(--ff-mono);font-size:.6rem;font-weight:600;">● {label}</span>')
            return (f'<span style="background:var(--surf3);color:var(--txt3);'
                    f'border:1px solid var(--bord);border-radius:20px;padding:2px 10px;'
                    f'font-family:var(--ff-mono);font-size:.6rem;">○ {label}</span>')

        st.markdown(
            f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.6rem;">'
            f'{badge(email_on,"Alerts")}{badge(appt_on,"Reminders")}{badge(weekly_on,"Weekly")}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="{CARD}"><div style="{TITLE}">🗂 Data Management</div></div>', unsafe_allow_html=True)
        n_hist = len(st.session_state.history)
        n_appt = len(st.session_state.appointments)
        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin-bottom:1rem;">'
            f'<div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:.8rem;text-align:center;">'
            f'<div style="font-family:var(--ff-display);font-size:1.6rem;color:var(--green);">{n_hist}</div>'
            f'<div style="font-size:.65rem;color:var(--txt3);">Saved assessments</div></div>'
            f'<div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:.8rem;text-align:center;">'
            f'<div style="font-family:var(--ff-display);font-size:1.6rem;color:var(--teal);">{n_appt}</div>'
            f'<div style="font-size:.65rem;color:var(--txt3);">Appointments</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        cc1, cc2 = st.columns(2, gap="small")
        with cc1:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("🗑  Clear History",     key="s_clr_h"):
                st.session_state.history = []; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with cc2:
            st.markdown('<div class="ghost">', unsafe_allow_html=True)
            if st.button("🗑  Clear Appointments", key="s_clr_a"):
                st.session_state.appointments = []; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="background:rgba(217,119,6,.04);border:1px solid rgba(217,119,6,.15);'
            'border-left:3px solid var(--amber);border-radius:8px;padding:.6rem .9rem;'
            'margin-top:.75rem;font-size:.72rem;color:var(--txt2);line-height:1.6;">'
            '⚠ Clearing data is permanent and cannot be undone.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("💾  Save All Settings", key="s_save"):
            st.session_state.settings = {
                "autosave": autosave,
                "fi":       show_fi,
                "alerts":   email_on,
                "thresh":   thresh,
            }
            st.success("✅ Settings saved!")


# ══════════════════════════════════════════════════════════════════
# PAGE · MEDICAL GUIDE
# ══════════════════════════════════════════════════════════════════
def page_guide():
    st.markdown(
        '<div class="page-hdr au"><div class="hdr-ico">📖</div><div>'
        '<div class="hdr-h">Medical Guide</div>'
        '<p class="hdr-p">Learn about every condition, test, and risk factor used in this assessment.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Search / filter bar
    search = st.text_input("🔍  Search conditions, tests, or risk factors…", key="guide_search",
                           placeholder="e.g. HPV, syphilis, Pap smear…")
    search_lower = search.strip().lower()

    # Category filter chips via selectbox
    all_cats = sorted({v["category"] for v in MEDICAL_GUIDE.values()})
    cat_filter = st.selectbox("Filter by category", ["All categories"] + all_cats, key="guide_cat")

    filtered = {
        k: v for k, v in MEDICAL_GUIDE.items()
        if (search_lower == "" or search_lower in k.lower() or search_lower in v["description"].lower()
            or search_lower in v["category"].lower() or search_lower in v.get("symptoms","").lower())
        and (cat_filter == "All categories" or v["category"] == cat_filter)
    }

    if not filtered:
        st.markdown(
            '<div class="card" style="text-align:center;padding:2.5rem;">'
            '<div style="font-size:2.5rem;margin-bottom:.75rem;">🔍</div>'
            '<div style="font-family:var(--ff-display);font-size:1rem;color:var(--txt);">No results found</div>'
            '<div style="font-size:.8rem;color:var(--txt3);margin-top:.4rem;">Try a different search term or category.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(f'<p style="font-family:var(--ff-mono);font-size:.65rem;color:var(--txt3);margin-bottom:1rem;">{len(filtered)} entries found</p>', unsafe_allow_html=True)

    for topic, info in filtered.items():
        c_val = info["color"]
        with st.expander(f"{info['icon']}  {topic}  ·  {info['short']}", expanded=False):
            st.markdown(
                f'<div style="background:#fff;border-left:3px solid {c_val};border-radius:0 12px 12px 0;'
                f'padding:1.25rem 1.5rem;margin-top:.25rem;">'
                f'<div style="font-family:var(--ff-mono);font-size:.54rem;letter-spacing:.18em;text-transform:uppercase;'
                f'color:{c_val};margin-bottom:.6rem;">{info["category"]}</div>'
                f'<div style="font-size:.84rem;color:var(--txt2);line-height:1.8;margin-bottom:1rem;">{info["description"]}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.65rem;margin-bottom:.8rem;">'
                f'<div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:.75rem;">'
                f'<div style="font-family:var(--ff-mono);font-size:.52rem;letter-spacing:.14em;text-transform:uppercase;color:var(--txt3);margin-bottom:.3rem;">Symptoms</div>'
                f'<div style="font-size:.77rem;color:var(--txt2);line-height:1.6;">{info["symptoms"]}</div></div>'
                f'<div style="background:var(--surf2);border:1px solid var(--bord);border-radius:10px;padding:.75rem;">'
                f'<div style="font-family:var(--ff-mono);font-size:.52rem;letter-spacing:.14em;text-transform:uppercase;color:var(--txt3);margin-bottom:.3rem;">Prevention</div>'
                f'<div style="font-size:.77rem;color:var(--txt2);line-height:1.6;">{info["prevention"]}</div></div>'
                f'<div style="background:rgba(12,155,88,.06);border:1px solid rgba(12,155,88,.2);border-left:3px solid {c_val};border-radius:0 10px 10px 0;padding:.75rem;">'
                f'<div style="font-family:var(--ff-mono);font-size:.52rem;letter-spacing:.14em;text-transform:uppercase;color:var(--txt3);margin-bottom:.3rem;">Cancer relevance</div>'
                f'<div style="font-size:.77rem;color:var(--txt2);line-height:1.6;">{info["relevance"]}</div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ghost">', unsafe_allow_html=True)
    if st.button("← Back to Classifier", key="guide_back"):
        goto("classifier")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════
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
    elif pg == "guide":      page_guide()
    else:                    page_home()

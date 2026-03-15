import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from logic import (
    kopparapu_hz, habitability_verdict,
    clear_search, apply_search_result, resolve_active_data,
    COLS, EARTH, PRESETS,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ExoHabit — Explore Distant Worlds",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="auto"
)

BASE_DIR = os.path.dirname(__file__)
def get_rel_path(rel_path): return os.path.join(BASE_DIR, rel_path)

# ── Data & Model Loading ──────────────────────────────────────────────────────
@st.cache_data
def load_planet_db():
    try: return pd.read_csv(get_rel_path("data/processed_exoplanet_data.csv"))
    except Exception: return pd.DataFrame()

@st.cache_resource
def load_model():
    path = get_rel_path("models/final_habitable_model.joblib")
    return joblib.load(path) if os.path.exists(path) else None

# ── HZ Physics (Kopparapu 2013) ───────────────────────────────────────────────
def kopparapu_hz(teff):
    T = teff - 5780.0
    inner = 1.107 + 1.332e-4*T + 1.580e-8*T**2 - 8.308e-12*T**3 - 1.931e-15*T**4
    outer = 0.356 + 6.171e-5*T + 1.698e-9*T**2 - 3.198e-12*T**3 - 5.573e-16*T**4
    return inner, outer

# ── Constants ─────────────────────────────────────────────────────────────────
COLS  = ['pl_orbper','pl_orbsmax','pl_rade','pl_insol','st_teff',
         'st_logg','st_rad','sy_dist','sy_vmag']
EARTH = {"pl_orbper":365.25,"pl_orbsmax":1.0,"pl_rade":1.0,"pl_insol":1.0,
         "st_teff":5778.0,"st_logg":4.44,"st_rad":1.0,"sy_dist":8.0,"sy_vmag":4.8}
PRESETS = {
    "Earth (reference)": EARTH,
    "TRAPPIST-1e":       {"pl_orbper":6.1,   "pl_orbsmax":0.029, "pl_rade":0.91, "pl_insol":0.66,
                          "st_teff":2566.0,  "st_logg":5.24,    "st_rad":0.12,  "sy_dist":12.0,   "sy_vmag":18.8},
    "Kepler-442 b":      {"pl_orbper":112.3, "pl_orbsmax":0.409, "pl_rade":1.34, "pl_insol":0.66,
                          "st_teff":4402.0,  "st_logg":4.67,    "st_rad":0.60,  "sy_dist":365.96, "sy_vmag":15.32},
}

# ── Session State ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        'page': 'landing',
        'found_planet_data': None,
        'search_key': 'default',
        # FIX: track which input source is active so preset can evict search
        'active_source': 'preset',   # 'preset' | 'search'
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Styles ────────────────────────────────────────────────────────────────────
STYLES = """
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp .stMarkdown {
    font-family: 'Inter', sans-serif !important; color: #e8e8f0 !important;
}
.stApp h1, .stApp h2, .stApp h3, [data-testid="stHeader"] h1, .hero-title {
    font-family: 'Playfair Display', serif !important; font-weight: 600 !important;
    color: #ffffff !important; letter-spacing: -0.015em !important;
}
.stApp {
    background-color: #0c0c16;
    background-image: url("./app/static/background.gif");
    background-size: cover; background-position: center; background-attachment: fixed;
}
[data-testid="stAppViewContainer"] { background: rgba(8, 8, 20, 0.72); }
.card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 28px; }
.pill { display: inline-block; padding: 4px 12px; background: rgba(108,99,255,0.15);
        color: #9f8fff; border-radius: 20px; font-size: 0.75rem; font-weight: 500;
        margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.05em; }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
section[data-testid="stSidebar"] {
    background-color: #0a0a14 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stMetric"] { background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 14px 16px !important; }
.stSlider [data-baseweb="slider"] [role="slider"] { background-color: #6c63ff !important; }
div.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #9f8fff) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 12px 28px !important; font-weight: 500 !important; transition: opacity 0.2s !important;
}
div.stButton > button:hover { opacity: 0.8 !important; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
def go_to_predictor(): st.session_state.page = 'predictor'
def go_to_landing():   st.session_state.page = 'landing'

# ── FIX: explicit clear so preset can always take back control ────────────────
def clear_search():
    st.session_state.found_planet_data = None
    st.session_state.search_key        = 'default'
    st.session_state.active_source     = 'preset'

# ═════════════════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ═════════════════════════════════════════════════════════════════════════════
def show_landing():
    st.markdown("<br><br>", unsafe_allow_html=True)
    hero_l, hero_r = st.columns([1.6, 1], gap="large")
    with hero_l:
        st.markdown('<div class="pill">NASA Exoplanet Archive · 6,147 confirmed worlds</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">Could another world support life?</h1>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:1.15rem;color:#aaaacc;line-height:1.75;max-width:520px;">
        Astronomers have confirmed thousands of planets beyond our Solar System.
        ExoHabit lets you explore them — and predict whether any could harbour life —
        using the same physics models used by NASA researchers.
        </p>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Open the Habitability Predictor →", on_click=go_to_predictor, key="hero_btn")
    with hero_r:
        st.markdown("""
        <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:24px;padding:48px 0;">
            <div style="width:60px;height:60px;border-radius:50%;background:linear-gradient(135deg,#0a3d62,#1e6fa5);box-shadow:0 0 30px rgba(93,232,176,0.4);flex-shrink:0;"></div>
            <div style="width:100px;height:100px;border-radius:50%;background:linear-gradient(135deg,#1a4a3a,#2d7a5f);box-shadow:0 0 50px rgba(93,232,176,0.6);flex-shrink:0;"></div>
            <div style="width:45px;height:45px;border-radius:50%;background:linear-gradient(135deg,#5a1a00,#c04000);box-shadow:0 0 30px rgba(255,100,50,0.4);flex-shrink:0;"></div>
        </div>""", unsafe_allow_html=True)
    st.divider()
    sc1, sc2, sc3 = st.columns(3, gap="medium")
    with sc1:
        st.markdown('<div class="card"><h3>What is an exoplanet?</h3><p style="font-size:0.9rem;color:#aaaacc;">A planet located outside our solar system, usually orbiting another star.</p></div>', unsafe_allow_html=True)
    with sc2:
        st.markdown('<div class="card"><h3>How do we find them?</h3><p style="font-size:0.9rem;color:#aaaacc;">By watching for star "wobbles" or a slight dimming as a planet passes in front.</p></div>', unsafe_allow_html=True)
    with sc3:
        st.markdown('<div class="card"><h3>Habitability 101</h3><p style="font-size:0.9rem;color:#aaaacc;">Planets need to be rocky and at the "Goldilocks" distance to host liquid water.</p></div>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;"><h2 style="font-size:1.8rem;">Ready to find a new world?</h2></div>', unsafe_allow_html=True)
    _, cent, _ = st.columns([1, 2, 1])
    with cent:
        st.button("Launch the Habitability Predictor →", on_click=go_to_predictor, key="cta_btn", use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
#  PREDICTOR PAGE
# ═════════════════════════════════════════════════════════════════════════════
def show_predictor():
    planet_db = load_planet_db()
    model     = load_model()

    MED = dict(EARTH)
    if not planet_db.empty:
        for c in COLS:
            if c in planet_db.columns:
                m = planet_db[c].median()
                if not np.isnan(m): MED[c] = m

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ExoHabit")
        st.button("← Back to Landing Page", on_click=go_to_landing)
        st.divider()

        # ── Search block ──────────────────────────────────────────────────────
        st.markdown("#### Search a Planet")
        st.markdown("<br>", unsafe_allow_html=True)
        search_query = st.text_input("planet_search", placeholder="e.g. KELT-9b",
                                     label_visibility="collapsed")

        if search_query.strip() and not planet_db.empty:
            def norm(s): return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()
            u_norm  = norm(search_query.strip())
            mask    = planet_db['pl_name'].apply(norm).str.contains(u_norm, case=False, na=False)
            matches = planet_db[mask]

            if len(matches) == 1:
                row = matches.iloc[0]
                # Only update state if this is a different planet than currently loaded
                if st.session_state.search_key != row['pl_name']:
                    st.session_state.found_planet_data = row
                    st.session_state.search_key        = row['pl_name']
                    st.session_state.active_source     = 'search'
                st.success(f"Loaded: {row['pl_name']}")

            elif len(matches) > 1:
                names  = matches['pl_name'].dropna().tolist()[:10]
                chosen = st.selectbox("Multiple matches:", names)
                row    = matches[matches['pl_name'] == chosen].iloc[0]
                if st.session_state.search_key != chosen:
                    st.session_state.found_planet_data = row
                    st.session_state.search_key        = chosen
                    st.session_state.active_source     = 'search'

            else:
                st.warning("No planets found.")

        elif not search_query.strip() and st.session_state.active_source == 'search':
            # FIX: user cleared the search box — revert to preset mode
            clear_search()

        st.divider()

        # ── Preset block ──────────────────────────────────────────────────────
        preset_name = st.selectbox("Presets", list(PRESETS.keys()))

        # FIX: explicit "Use this preset" button so intent is unambiguous
        if st.button("Apply Preset", use_container_width=True):
            clear_search()

        st.divider()

        # ── Resolve active data source ────────────────────────────────────────
        # active_source == 'search' → use found_planet_data
        # active_source == 'preset' → use selected preset
        if st.session_state.active_source == 'search' and st.session_state.found_planet_data is not None:
            def _v(c):
                val = st.session_state.found_planet_data.get(c)
                return float(val) if pd.notna(val) else MED[c]
            D  = {c: _v(c) for c in COLS}
            sk = st.session_state.search_key
        else:
            D  = PRESETS[preset_name]
            sk = f"preset_{preset_name}"   # unique key per preset so sliders reset

        orb_per  = st.slider("Year length (days)",          0.1,   5000.0, float(D["pl_orbper"]),  key=f"per_{sk}")
        orb_max  = st.slider("Orbital Distance (AU)",       0.01,  10.0,   float(D["pl_orbsmax"]), key=f"max_{sk}")
        pl_rade  = st.slider("Planet Size (× Earth)",       0.1,   10.0,   float(D["pl_rade"]),    key=f"rad_{sk}")
        pl_insol = st.slider("Sunlight Received (× Earth)", 0.01,  100.0,  float(D["pl_insol"]),   key=f"ins_{sk}")
        st_teff  = st.slider("Star Temperature (K)",        2000,  10000,  int(D["st_teff"]),      key=f"teff_{sk}")

    # ── Main UI ───────────────────────────────────────────────────────────────
    c1, c2 = st.columns([1.4, 1], gap="large")

    with c1:
        fp    = st.session_state.found_planet_data if st.session_state.active_source == 'search' else None
        title = fp['pl_name'] if fp is not None else preset_name
        st.markdown(f"# {title}")

        if pl_insol < 0.30:  glow, color = "#a0d8ef", "linear-gradient(135deg,#0a3d62,#1e6fa5)"
        elif pl_insol < 2.0: glow, color = "#5de8b0", "linear-gradient(135deg,#1a4a3a,#2d7a5f)"
        else:                glow, color = "#ff6b35", "linear-gradient(135deg,#5a1a00,#c04000)"

        vsz = max(60, min(240, int(pl_rade * 80)))
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                    border-radius:16px;padding:60px;display:flex;justify-content:center;
                    align-items:center;min-height:360px;">
            <div style="width:{vsz}px;height:{vsz}px;background:{color};border-radius:50%;
                        box-shadow:0 0 50px {glow}44,inset -20px -20px 40px rgba(0,0,0,0.5);"></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("# Planet Analysis")
        input_data = pd.DataFrame([{c: float(D[c]) for c in COLS}])
        input_data.at[0, 'pl_orbper']  = orb_per
        input_data.at[0, 'pl_orbsmax'] = orb_max
        input_data.at[0, 'pl_rade']    = pl_rade
        input_data.at[0, 'pl_insol']   = pl_insol
        input_data.at[0, 'st_teff']    = st_teff

        hz_inner, hz_outer = kopparapu_hz(st_teff)
        in_hz = hz_outer <= pl_insol <= hz_inner
        rocky = 0.5 <= pl_rade <= 1.6

        if in_hz and rocky:
            st.success("**Potentially Habitable** — Matches physical criteria for liquid water.")
        else:
            msg = "Outside Habitable Zone" if not in_hz else "Too large/small for rocky surface"
            st.error(f"**Not Habitable** — {msg}")

        if model:
            prob = model.predict_proba(input_data)[0][1]
            st.metric("Habitability Likelihood", f"{prob * 100:.1f}%")
            st.progress(min(float(prob), 0.9999))

# ── Render ────────────────────────────────────────────────────────────────────
if st.session_state.page == 'landing':
    show_landing()
else:
    show_predictor()
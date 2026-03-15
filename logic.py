"""
Pure logic — no Streamlit imports.
Imported by both app.py and test_exohabit.py.
"""
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
COLS = [
    'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_insol', 'st_teff',
    'st_logg',   'st_rad',     'sy_dist', 'sy_vmag',
]

EARTH = {
    "pl_orbper": 365.25, "pl_orbsmax": 1.0,  "pl_rade": 1.0,   "pl_insol": 1.0,
    "st_teff":   5778.0, "st_logg":    4.44, "st_rad":  1.0,   "sy_dist":  8.0,
    "sy_vmag":   4.8,
}

PRESETS = {
    "Earth (reference)": EARTH,
    "TRAPPIST-1e": {
        "pl_orbper": 6.1,   "pl_orbsmax": 0.029, "pl_rade": 0.91, "pl_insol": 0.66,
        "st_teff":   2566.0,"st_logg":    5.24,  "st_rad":  0.12, "sy_dist":  12.0,
        "sy_vmag":   18.8,
    },
    "Kepler-442 b": {
        "pl_orbper": 112.3,  "pl_orbsmax": 0.409, "pl_rade": 1.34, "pl_insol": 0.66,
        "st_teff":   4402.0, "st_logg":    4.67,  "st_rad":  0.60, "sy_dist":  365.96,
        "sy_vmag":   15.32,
    },
}

# ── HZ Physics (Kopparapu 2013) ───────────────────────────────────────────────
def kopparapu_hz(teff: float) -> tuple[float, float]:
    T = teff - 5780.0
    inner = 1.107 + 1.332e-4*T + 1.580e-8*T**2 - 8.308e-12*T**3 - 1.931e-15*T**4
    outer = 0.356 + 6.171e-5*T + 1.698e-9*T**2 - 3.198e-12*T**3 - 5.573e-16*T**4
    return inner, outer

# ── Habitability verdict ──────────────────────────────────────────────────────
def habitability_verdict(pl_insol: float, pl_rade: float, st_teff: float) -> tuple[bool, bool]:
    hz_inner, hz_outer = kopparapu_hz(st_teff)
    in_hz = hz_outer <= pl_insol <= hz_inner
    rocky = 0.5 <= pl_rade <= 1.6
    return in_hz, rocky

# ── Session-state helpers (operate on a dict-like object) ────────────────────
def clear_search(state) -> None:
    """Reset search state, reverting control to preset."""
    state['found_planet_data'] = None
    state['search_key']        = 'default'
    state['active_source']     = 'preset'

def apply_search_result(state, row: pd.Series) -> None:
    """Load a searched planet into state (no-op if same planet already loaded)."""
    if state['search_key'] != row['pl_name']:
        state['found_planet_data'] = row
        state['search_key']        = row['pl_name']
        state['active_source']     = 'search'

def resolve_active_data(state, preset_name: str, med: dict) -> tuple[dict, str]:
    """
    Return (D, slider_key) where D is the dict of column values to seed
    sliders from, and slider_key is the Streamlit widget key suffix.
    """
    if state['active_source'] == 'search' and state['found_planet_data'] is not None:
        row = state['found_planet_data']
        def _v(c):
            val = row.get(c)
            return float(val) if pd.notna(val) else med[c]
        D  = {c: _v(c) for c in COLS}
        sk = state['search_key']
    else:
        D  = PRESETS[preset_name]
        sk = f"preset_{preset_name}"
    return D, sk
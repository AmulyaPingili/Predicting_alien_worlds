import pandas as pd
import numpy as np

def calculate_luminosity(st_teff, st_rad):
    """
    Calculates Stellar Luminosity in Solar Units (L_sun).
    Using Stefan-Boltzmann Law: L = 4 * pi * R^2 * sigma * T^4
    Relatively: L/L_sun = (R/R_sun)^2 * (T/T_sun)^4
    """
    T_sun = 5780.0
    # L_star = (R_star/R_sun)^2 * (T_eff/T_sun)^4
    return (st_rad ** 2) * ((st_teff / T_sun) ** 4)

def get_hz_boundaries(st_teff):
    """
    Calculates Habitable Zone (HZ) boundaries using Kopparapu et al. (2013).
    S_eff = Seff_sun + a*T + b*T^2 + c*T^3 + d*T^4
    where T = Teff - 5780
    """
    T_diff = st_teff - 5780.0
    
    # Coefficients for Runaway Greenhouse (Inner Boundary)
    S_eff_sun_inner = 1.107
    a_inner = 1.332e-4
    b_inner = 1.580e-8
    c_inner = -8.308e-12
    d_inner = -1.931e-15
    
    S_eff_inner = (S_eff_sun_inner + a_inner * T_diff + b_inner * (T_diff**2) + 
                   c_inner * (T_diff**3) + d_inner * (T_diff**4))
    
    # Coefficients for Maximum Greenhouse (Outer Boundary)
    S_eff_sun_outer = 0.356
    a_outer = 6.171e-5
    b_outer = 1.698e-9
    c_outer = -3.198e-12
    d_outer = -5.573e-16
    
    S_eff_outer = (S_eff_sun_outer + a_outer * T_diff + b_outer * (T_diff**2) + 
                   c_outer * (T_diff**3) + d_outer * (T_diff**4))
    
    return S_eff_inner, S_eff_outer

def engineer_features(df):
    """
    Performs physics-first feature engineering on the Planetary Systems (ps) dataset.
    """
    print("Beginning Physics-First Feature Engineering...")
    df = df.copy()

    # 1. Fill critical gaps: Luminosity
    # If st_rad or st_teff is missing, we can't do much for HZ, but we fill what we can.
    df['st_lum'] = calculate_luminosity(df['st_teff'], df['st_rad'])

    # 2. Manual Insolation Flux (if missing)
    # S = L / d^2 (where d is semi-major axis in AU)
    # This fills the 85% missing 'pl_insol' identified in EDA
    df['pl_insol_calc'] = df['st_lum'] / (df['pl_orbsmax'] ** 2)
    df['pl_insol'] = df['pl_insol'].fillna(df['pl_insol_calc'])

    # 3. HZ Boundary Calculation
    # We define boundaries for each star based on its temperature
    boundaries = df['st_teff'].apply(get_hz_boundaries)
    df['hz_inner_limit'], df['hz_outer_limit'] = zip(*boundaries)

    # 4. Target Generation: Scientifically Habitable Candidate
    # A planet is a "Habitable Candidate" if:
    # - It is in the HZ (S_eff_outer <= S_calc <= S_eff_inner)
    # - It is Earth-sized (0.5 <= pl_rade <= 1.6) - Conservative limit
    df['is_habitable'] = (
        (df['pl_insol'] >= df['hz_outer_limit']) & 
        (df['pl_insol'] <= df['hz_inner_limit']) &
        (df['pl_rade'] >= 0.5) & 
        (df['pl_rade'] <= 1.6)
    ).astype(int)

    # 5. Handle extreme outliers identified in EDA (Clipping)
    # Orbital periods of millions of days are physically real but can skew models.
    df['pl_orbper'] = df['pl_orbper'].clip(upper=10000) # Clip at ~27 years
    
    print(f"Feature engineering complete. Found {df['is_habitable'].sum()} potential habitable candidates.")
    return df

if __name__ == "__main__":
    try:
        data_path = "data/latest_exoplanet_data.csv"
        df = pd.read_csv(data_path)
        processed_df = engineer_features(df)
        
        # Save processed data
        output_path = "data/processed_exoplanet_data.csv"
        processed_df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        
        # Quick check for habitable candidates
        habitable = processed_df[processed_df['is_habitable'] == 1]
        if not habitable.empty:
            print("\nPreview of Habitable Candidates:")
            print(habitable[['pl_name', 'pl_rade', 'pl_insol', 'st_teff']].head())
            
    except Exception as e:
        print(f"Feature engineering failed: {e}")

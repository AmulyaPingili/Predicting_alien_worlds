import pandas as pd
import io
import requests
import os

def fetch_planetary_systems_data():
    """
    Fetches the latest planetary systems data from NASA's TAP service.
    This replaces the mission-specific 'CUMULATIVE' table with 'ps',
    which includes Kepler, K2, and TESS discoveries as of March 2026.
    """
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # Mapping physical features to the 'ps' table schema
    # 'default_flag' ensures we get one primary row per planet
    query = (
        "SELECT pl_name, default_flag, disc_facility, disc_year, pl_orbper, pl_orbsmax, "
        "pl_rade, pl_eqt, pl_insol, st_teff, st_logg, st_rad, ra, dec, sy_dist, sy_vmag "
        "FROM ps "
        "WHERE default_flag = 1"
    )

    
    params = {
        "query": query,
        "format": "csv"
    }
    
    print("Connecting to NASA Exoplanet Archive (Planetary Systems Table)...")
    try:
        response = requests.get(base_url, params=params, timeout=60)
        
        if response.status_code == 200:
            if "Error" in response.text:
                raise Exception(f"TAP Service Error: {response.text}")
                
            df = pd.read_csv(io.StringIO(response.text))
            
            if df.empty:
                raise Exception("Received empty dataset from NASA.")
                
            print(f"Successfully downloaded {len(df)} confirmed records.")
            return df
        else:
            raise Exception(f"HTTP Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        raise

if __name__ == "__main__":
    try:
        # 1. Fetch data
        df = fetch_planetary_systems_data()

        # 2. Habitability Logic Preparation
        # We keep confirmed planets only (as 'ps' is a confirmed planet table).
        # We will create our habitability 'target' in the next step using HZ math.
        
        # 3. Save to local CSV (Zero-Cost approach)
        output_path = "data/latest_exoplanet_data.csv"
        os.makedirs("data", exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\nData saved to {output_path}")
        print(f"Mission Breakdown:\n{df['disc_facility'].value_counts().head()}")
        
    except Exception as e:
        print(f"\nExecution failed: {e}")
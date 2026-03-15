import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_data(path="data/latest_exoplanet_data.csv"):
    """Loads the latest exoplanet discovery dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Run data_ingestion.py first.")
    return pd.read_csv(path)

def generate_visualizations(df):
    """Generates a comprehensive scientific EDA suite."""
    os.makedirs("reports/plots", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # --- 1. Missing Value Analysis (Scientific Justification) ---
    plt.figure(figsize=(10, 6))
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        sns.barplot(x=missing.values, y=missing.index, hue=missing.index, palette="Reds_r", legend=False)
        plt.title("Percentage of Missing Data by Feature\n(Justifies Physics-First Derived Features)")
        plt.xlabel("Percentage Missing (%)")
        plt.tight_layout()
        plt.savefig("reports/plots/missing_data.png")
    plt.close()

    # --- 2. Discovery Chronology (1990 - 2026) ---
    plt.figure(figsize=(12, 6))
    yearly_discoveries = df.groupby('disc_year').size()
    plt.plot(yearly_discoveries.index, yearly_discoveries.values, marker='o', linestyle='-', color='darkblue')
    plt.fill_between(yearly_discoveries.index, yearly_discoveries.values, alpha=0.2, color='blue')
    plt.title("The Golden Age of Discovery: Exoplanets Found per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Confirmed Planets")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/plots/discovery_chronology.png")
    plt.close()

    # --- 3. Planetary Parameter Outliers (Boxplots) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(y=df['pl_orbper'], ax=axes[0], color='skyblue')
    axes[0].set_yscale('log')
    axes[0].set_title("Orbital Period Outliers (Log Scale)")
    
    sns.boxplot(y=df['pl_rade'], ax=axes[1], color='salmon')
    axes[1].set_yscale('log')
    axes[1].set_title("Planetary Radius Outliers (Log Scale)")
    plt.tight_layout()
    plt.savefig("reports/plots/outlier_analysis.png")
    plt.close()

    # --- 4. HR-Diagram (Stellar Temp vs. Gravity) ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['st_teff'], df['st_logg'], c=df['st_rad'], cmap='YlOrRd', alpha=0.6, s=10)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.colorbar(scatter, label='Stellar Radius (Solar Radii)')
    plt.title("Stellar Context: Temperature vs. Surface Gravity")
    plt.xlabel("Effective Temperature (K)")
    plt.ylabel("Surface Gravity (log10(g))")
    plt.tight_layout()
    plt.savefig("reports/plots/stellar_context_hr.png")
    plt.close()

    # --- 5. Correlation Heatmap ---
    plt.figure(figsize=(12, 10))
    # Select original physical columns
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['default_flag'], errors='ignore')
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title("Global Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("reports/plots/correlation_matrix.png")
    plt.close()

    print("Comprehensive EDA visualizations saved to reports/plots/")

def data_summary(df):
    """Prints a statistical summary of the dataset."""
    print("\n" + "="*40)
    print("Academic Data Summary")
    print("="*40)
    print(f"Total Confirmed Worlds: {len(df)}")
    print(f"Time Range: {int(df['disc_year'].min())} - {int(df['disc_year'].max())}")
    
    # Calculate sparsity
    sparsity = df.isnull().mean().mean() * 100
    print(f"Dataset Sparsity: {sparsity:.2f}%")
    print("="*40)

if __name__ == "__main__":
    try:
        data = load_data()
        data_summary(data)
        generate_visualizations(data)
    except Exception as e:
        print(f"EDA failed: {e}")


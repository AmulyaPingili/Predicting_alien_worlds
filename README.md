# ExoHabit: AI Exoplanet Habitability Predictor

A machine learning application that uses a **Physics-First** approach to identify potentially habitable exoplanets from the NASA Planetary Systems Archive.

![Dashboard Preview](assets/hero.png)

## 🚀 Instant Setup (Zero-Cost Deployment)

This repository is designed to be fully portable. The final trained model is pre-included in the `models/` directory, so you do **not** need to re-run the training or data ingestion scripts to explore the dashboard.

### 1. Prerequisites
- Python 3.9+
- Pip

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd Predicting_alien_worlds
pip install -r requirements.txt
```

### 3. Launch the Dashboard
Run the following command to start the interactive Streamlit server:

```bash
streamlit run app.py
```

## 🧠 Scientific Methodology
- **Target Architecture**: LightGBM (Tuned via Optuna Bayesian Optimization).
- **Physics Features**: Derived Insolation Flux (`pl_insol`), Stellar Luminosity, and Habitable Zone boundaries.
- **Performance**: Achieved **1.0 F1-score** on the hold-out vetting set.

## 📁 Repository Structure
- `app.py`: Main Streamlit dashboard source.
- `src/model.py`: Final production pipeline definition.
- `models/`: Contains the serialized `.joblib` model file.
- `src/features.py`: Logic for manual physics calculations.
- `notebooks/`: Research logs and model benchmarking history.

---
*Data Source: NASA Exoplanet Archive (Live TAP Service)*

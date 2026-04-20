<div align="center">

# Used Car Price Prediction (CarDekho)

![GitHub repo size](https://img.shields.io/github/repo-size/avijit-jana/used-car-price-prediction?style=plastic)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-f7931e.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)
![GitHub language count](https://img.shields.io/github/languages/count/avijit-jana/used-car-price-prediction?style=plastic)
![GitHub top language](https://img.shields.io/github/languages/top/avijit-jana/used-car-price-prediction?style=plastic)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/avijit-jana/used-car-price-prediction?color=red\&style=plastic)

End-to-end **used car price prediction** project for estimating **car resale value** using a clean **data preprocessing + feature engineering + regression modeling** workflow in **Python**. Includes an interactive **Streamlit** web app for price inference and reproducible notebooks for **EDA (Exploratory Data Analysis)** and model development.

![Car](app/car.png)

</div>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Run the Streamlit app](#run-the-streamlit-app)
  - [Reproduce preprocessing + EDA](#reproduce-preprocessing--eda)
  - [Reproduce model training](#reproduce-model-training)
  - [Programmatic prediction (Python)](#programmatic-prediction-python)
- [Results Summary](#results-summary)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository demonstrates a practical **machine learning regression** pipeline for **used car resale price prediction**. It covers:

- Multi-city dataset consolidation (Bangalore, Chennai, Delhi, Hyderabad, Jaipur, Kolkata)
- Robust cleaning of messy/real-world marketplace data (units, missing values, outliers)
- Feature engineering and encoding of categorical variables
- Model benchmarking and selection (baseline algorithms -> tuned model)
- Deployment-ready inference via a **Streamlit price predictor**

Data is collected from CarDekho used car listings (see `Others/Info.txt`).

## Features

- **Preprocessing**
  - Dataset merge across cities
  - Data cleaning (types, units, missing values)
  - Outlier handling
  - Persisted preprocessing artifacts (encoders/scalers)
- **Feature Engineering**
  - Structured feature extraction from nested fields (flattening to tabular data)
  - Categorical encoding and numerical scaling for modeling
  - Feature selection notes and documentation (`Others/Feature Description.pdf`)
- **Regression Modeling**
  - Baseline comparison across multiple regressors
  - Cross-validation and test-set evaluation (MAE / RMSE / R^2)
  - Hyperparameter tuning for the selected model
  - Saved model artifact for inference (`Utility Files/model.pkl`)

## Tech Stack

- **Language:** Python
- **Core:** Pandas, NumPy
- **ML:** scikit-learn (regression models, preprocessing, CV, metrics)
- **EDA:** notebooks + visual analysis (Exploratory Data Analysis)
- **App:** Streamlit

## Project Structure

```text
used-car-price-prediction/
|-- app/
|   |-- Price_Prediction.py          # Streamlit app (inference UI)
|   `-- car.png
|-- DataSets/                        # Raw city-level datasets (Excel)
|-- NoteBooks/
|   |-- Preprocessing & EDA.ipynb     # Data prep + EDA pipeline
|   `-- Model Development.ipynb       # Baselines, tuning, evaluation, exports
|-- Utility Files/                   # Processed data + trained artifacts
|   |-- car_data.xlsx
|   |-- encoded_car_data.xlsx
|   |-- label_encoder.pkl
|   |-- scaler.pkl
|   |-- model.pkl
|   `-- selected_features.txt
|-- requirements.txt
`-- README.md
```

## Dataset

- **Source:** CarDekho used car listings (`Others/Info.txt`)
- **Raw files:** `DataSets/*.xlsx` (city-wise)
- **Processed dataset:** `Utility Files/car_data.xlsx`
- **Encoded dataset used for modeling:** `Utility Files/encoded_car_data.xlsx`

From `NoteBooks/Preprocessing & EDA.ipynb`, the city-wise data is concatenated (combined shape reported as **(8369, 6)** in the notebook output) and then flattened/cleaned into a modeling-ready table.

From `NoteBooks/Model Development.ipynb`, the encoded modeling dataset shape is reported as **(7853, 16)** with `price` as the target.

**Target:** `price` (INR)

**Example modeling features** (see the Streamlit app's `FEATURE_ORDER`):

- `Fuel type`, `Body type`, `transmission`, `model`, `variantName`, `Insurance Validity`, `City`
- `Kilometers driven`, `ownerNo`, `modelYear`, `Registration Year`
- `Mileage(kmpl)`, `Engine(CC)`, `Max Power(bhp)`, `Torque(Nm)`

## Installation

```bash
# (optional) create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

### Run the Streamlit app

```bash
streamlit run "app/Price_Prediction.py"
```

The app loads:

- Trained model: `Utility Files/model.pkl`
- Preprocessors: `Utility Files/label_encoder.pkl`, `Utility Files/scaler.pkl`
- Reference data for categories: `Utility Files/car_data.xlsx`

Note: `Utility Files/model.pkl` is a large binary file. If you publish this repository on GitHub, consider using **Git LFS** for model artifacts or attaching them as **Release assets**.

### Reproduce preprocessing     + EDA

Open and run:

- `NoteBooks/Preprocessing & EDA.ipynb`

This notebook demonstrates dataset concatenation, cleaning, missing-value handling, outlier removal, categorical encoding, and EDA.

### Reproduce model training

Open and run:

- `NoteBooks/Model Development.ipynb`

This notebook benchmarks multiple regression algorithms, performs cross-validation, tunes the best model, and exports artifacts used by the app.

## Results Summary

Metrics reported in `NoteBooks/Model Development.ipynb` (test-set comparison):

- **Random Forest Regressor:** MAE **89,905**, RMSE **167,212**, R^2 **0.9416**
- Gradient Boosting Regressor: MAE 114,597, RMSE 202,607, R^2 0.9143
- Decision Tree Regressor: MAE 118,577, RMSE 228,176, R^2 0.8913
- Linear Regression: MAE 223,371, RMSE 348,656, R^2 0.7462

Baseline 5-fold CV (R^2) reported in the same notebook:

- Linear Regression: **0.7355** (+/- 0.0126)
- Decision Tree: **0.8606** (+/- 0.0298)
- Random Forest: **0.9219** (+/- 0.0124)
- Gradient Boosting: **0.8996** (+/- 0.0111)

Note: Exact results can vary if you retrain (different splits, preprocessing choices, or scikit-learn versions).

## Contributing

Contributions are welcome—especially improvements to data cleaning, feature engineering, model evaluation, and Streamlit UX.

- Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and follow [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
- For security issues, see [`SECURITY.md`](SECURITY.md).

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

![Developer](https://img.shields.io/badge/Developed%20By-Avijit_Jana-navy?style=for-the-badge)

</div>

<div align="center">

# 🚗 Used Car Price Prediction

![GitHub repo size](https://img.shields.io/github/repo-size/avijit-jana/used-car-price-prediction?style=plastic)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-f7931e.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)
![GitHub language count](https://img.shields.io/github/languages/count/avijit-jana/used-car-price-prediction?style=plastic)
![GitHub top language](https://img.shields.io/github/languages/top/avijit-jana/used-car-price-prediction?style=plastic)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/avijit-jana/used-car-price-prediction?color=red\&style=plastic)

**A production-minded machine learning project that estimates resale value of used cars with a clean, reproducible pipeline and interpretable results.**

![Car](app/car.png)

</div>

## 📌 Executive Summary

The determination of an equitable resale valuation for motor vehicles necessitates a sophisticated analysis of multi-dimensional variables. Beyond simple age-based depreciation, values are influenced by temporal factors, mechanical utilization (mileage), fuel efficiency standards, and the fluctuating sentiments of prevailing market dynamics.

This repository presents a comprehensive, end-to-end computational pipeline designed to bridge the gap between raw automotive data and high-fidelity price predictions. By leveraging state-of-the-art machine learning algorithms, the framework provides not only numerical accuracy but also deep model interpretability, ensuring that valuation outputs are both precise and justifiable to stakeholders.

## ✨ Why This Project Matters

Pricing used cars is inherently noisy—depreciation, mileage, ownership, and market sentiment all interact in non-linear ways. This project demonstrates how to translate that complexity into a **robust regression system** that is:

* **Accurate** – leverages tree-based models for non-linear relationships
* **Explainable** – focuses on feature importance and model behavior
* **Reproducible** – clean pipeline from raw data → prediction

## 🏗️ Project Architecture

```
Raw Data → Cleaning → Feature Engineering → Model Training → Evaluation → Prediction
```

Key design goals:

* Minimal leakage
* Consistent preprocessing
* Model comparability

## Project Structure

```text
used-car-price-prediction/
|-- app/
|   |-- price_prediction.py        # Streamlit app (inference UI)
|   |-- car.png
|
|-- notebooks/
|   |-- preprocessing_eda.ipynb    # Data prep + EDA pipeline
|   |-- model_development.ipynb    # Baselines, tuning, evaluation, exports
|
|-- artifacts/                     # Processed data + trained artifacts
|   |-- car_data.xlsx
|   |-- encoded_car_data.xlsx
|   |-- label_encoder.pkl
|   |-- scaler.pkl
|   |-- model.pkl
|   |-- selected_features.txt
|
|-- requirements.txt
|-- CODE_OF_CONDUCT.md
|-- CONTRIBUTING.md
|-- LICENSE
|-- README.md
```

## 🔬 Exploratory Data Analysis (EDA)

EDA focuses on uncovering signal and validating assumptions:

* Distribution of selling price
* Correlation between price and age/mileage
* Categorical feature impact (fuel, transmission, seller)
* Outlier detection 

## Results Summary

Metrics reported in [`NoteBooks/Model Development.ipynb`](NoteBooks/Model%20Development.ipynb) (test-set comparison):

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

- Please read [**`CONTRIBUTING.md`**](CONTRIBUTING.md) and follow [**`CODE_OF_CONDUCT.md`**](CODE_OF_CONDUCT.md).
- For security issues, see [**`SECURITY.md`**](SECURITY.md).

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [**`LICENSE`**](LICENSE) for details.

---

<div align="center">

![Developer](https://img.shields.io/badge/Developed%20By-Avijit_Jana-navy?style=for-the-badge)

</div>

import os
import pickle
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Environment-based Paths (Deployment Friendly)
# ─────────────────────────────────────────────────────────────
CAR_DATA_PATH = os.getenv("CAR_DATA_PATH", "Utility Files/car_data.xlsx")
MODEL_PATH = os.getenv("MODEL_PATH", "Utility Files/model.pkl")
LABEL_ENCODER_PATH = os.getenv("ENCODER_PATH", "Utility Files/label_encoder.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "Utility Files/scaler.pkl")

CAR_IMAGE_PATH = "app/car.png"

# ─────────────────────────────────────────────────────────────
# Streamlit Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarDekho Price Predictor",
    page_icon="🚗",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Load Resources
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_car_data():
    if os.path.exists(CAR_DATA_PATH):
        return pd.read_excel(CAR_DATA_PATH)

    else:
        path = hf_hub_download(
            repo_id=st.secrets["huggingface"]["repo_id"],
            filename="car_data.xlsx",
            token=st.secrets["huggingface"]["hf_token"]
        )
        return pd.read_excel(path)

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    else:
        path = hf_hub_download(
            repo_id=st.secrets["huggingface"]["repo_id"],
            filename="model.pkl",
            token=st.secrets["huggingface"]["hf_token"]
        )
        return pickle.load(open(path, "rb"))

@st.cache_resource(show_spinner="Loading preprocessors...")
def load_preprocessors():
    if os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(SCALER_PATH):
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            sc = pickle.load(f)
        return le, sc
    else:
        le_path = hf_hub_download(
            repo_id=st.secrets["huggingface"]["repo_id"],
            filename="label_encoder.pkl",
            token=st.secrets["huggingface"]["hf_token"]
        )
        sc_path = hf_hub_download(
            repo_id=st.secrets["huggingface"]["repo_id"],
            filename="scaler.pkl",
            token=st.secrets["huggingface"]["hf_token"]
        )
        le = pickle.load(open(le_path, "rb"))
        sc = pickle.load(open(sc_path, "rb"))
        return le, sc

# ─────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "Fuel type", "Body type", "transmission",
    "model", "variantName", "Insurance Validity", "City",
]

FEATURE_ORDER = [
    "Fuel type", "Body type", "Kilometers driven", "transmission",
    "ownerNo", "model", "modelYear", "variantName",
    "Registration Year", "Insurance Validity",
    "Mileage(kmpl)", "Engine(CC)", "Max Power(bhp)", "Torque(Nm)", "City",
]


# ─────────────────────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────────────────────
def predict_price(features, car_data):
    model = load_model()
    label_encoder, scaler = load_preprocessors()

    # Validate schema
    if set(FEATURE_ORDER) != set(features.keys()):
        raise ValueError("Feature mismatch detected")

    df = pd.DataFrame([features])[FEATURE_ORDER]

    # Encode categorical features safely
    for col in CATEGORICAL_COLS:
        try:
            df[col] = label_encoder.transform(df[col].astype(str))
        except Exception:
            df[col] = -1  # fallback for unseen values

    df_scaled = scaler.transform(df)
    return model.predict(df_scaled)[0]


# ─────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────
def main():
    try:
        df = load_car_data()
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

    def opts(col):
        return sorted(df[col].dropna().unique())

    # Sidebar
    with st.sidebar:
        st.header('About')
        st.write(
            'This app uses a tuned **Random Forest** model to estimate the '
            'resale price of a used car based on vehicle features.'
        )
        st.markdown('**- Developed by Avijit Jana**')
        try:
            st.image(CAR_IMAGE_PATH, width=250)
        except Exception:
            pass

        with st.expander('Model Info'):
            st.markdown(
                '- Algorithm: Random Forest Regressor\n'
                '- Tuning: RandomizedSearchCV (50 iterations, 5-fold CV)\n'
                '- Scaling: MinMaxScaler\n'
                '- Encoding: LabelEncoder (per column)\n'
            )

    # Header
    st.header("🚗 CarDekho Price Prediction", divider="red")

  # ────────────────────────────────────────────────────────────
  # Input form 
  # ────────────────────────────────────────────────────────────
    with st.form('prediction_form'):
        st.subheader('Car Details')

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            owner    = int(st.selectbox('Owner Number',        opts('ownerNo')))
            reg_year = int(st.selectbox('Registration Year',   opts('Registration Year')))
        with col2:
            fuel    = st.selectbox('Fuel Type',                opts('Fuel type'))
            mileage = float(st.selectbox('Mileage (kmpl)',     opts('Mileage(kmpl)')))
        with col3:
            model_year = int(st.selectbox('Model Year',        opts('modelYear')))
            engine     = float(st.selectbox('Engine (CC)',     opts('Engine(CC)')))
        with col4:
            km_driven = int(st.selectbox('Kilometers Driven',  opts('Kilometers driven')))
            torque    = float(st.selectbox('Torque (Nm)',      opts('Torque(Nm)')))
        with col5:
            city         = st.selectbox('City',                opts('City'))
            transmission = st.selectbox('Transmission',        opts('transmission'))

        col6, col7, col8 = st.columns(3)
        with col6:
            body      = st.selectbox('Body Type',              opts('Body type'))
        with col7:
            insurance = st.selectbox('Insurance Validity',     opts('Insurance Validity'))
        with col8:
            max_power = float(st.selectbox('Max Power (bhp)',  opts('Max Power(bhp)')))

        col9, col10 = st.columns(2)
        with col9:
            model_name = st.selectbox('Model Name',            opts('model'))
        with col10:
            variant_opts = sorted(
                df.loc[df['model'] == model_name, 'variantName'].dropna().unique()
            ) or opts('variantName')
            variant = st.selectbox('Variant Name', variant_opts)

        submitted = st.form_submit_button('🔍 Predict Price', use_container_width=True)

    if submitted:
        if km_driven < 0:
            st.error("Kilometers cannot be negative")
            st.stop()
        features = {
            'Fuel type'         : fuel,
            'Body type'         : body,
            'Kilometers driven' : km_driven,
            'transmission'      : transmission,
            'ownerNo'           : owner,
            'model'             : model_name,
            'modelYear'         : model_year,
            'variantName'       : variant,
            'Registration Year' : reg_year,
            'Insurance Validity': insurance,
            'Mileage(kmpl)'     : mileage,
            'Engine(CC)'        : engine,
            'Max Power(bhp)'    : max_power,
            'Torque(Nm)'        : torque,
            'City'              : city,
        }

        with st.spinner('Predicting …'):
            try:
                prediction = predict_price(features, df)
            except FileNotFoundError as e:
                st.error(f'❌ Model file not found: {e}')
                st.stop()
            except Exception as e:
                st.error(f'❌ Prediction failed: {e}')
                st.stop()

        # ── Result display ────────────────────────────────────────────────────
        st.divider()
        _, res_col, _ = st.columns([1, 2, 1])
        with res_col:
            st.success('Prediction complete!')
            st.metric(
                label='💰 Estimated Resale Price',
                value=f'₹ {prediction:,.0f}',
            )
            lakh = prediction / 1e5
            if lakh < 5:
                band = '🟢 Budget segment (< ₹5 Lakh)'
            elif lakh < 15:
                band = '🟡 Mid-range segment (₹5 – 15 Lakh)'
            else:
                band = '🔴 Premium segment (> ₹15 Lakh)'
            st.caption(band)

# ─────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

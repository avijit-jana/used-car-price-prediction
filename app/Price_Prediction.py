import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ── Local file paths ──────────────────────────────────────────────────────────
CAR_DATA_PATH      = 'Utility Files/car_data.xlsx'
MODEL_PATH         = 'Utility Files/model.pkl'
LABEL_ENCODER_PATH = 'Utility Files/label_encoder.pkl'
SCALER_PATH        = 'Utility Files/scaler.pkl'
CAR_IMAGE_PATH     = 'app/car.png'

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = 'CarDekho Price Predictor',
    page_icon  = '🚗',
    layout     = 'wide',
)

# ── Resource loading (cached) ─────────────────────────────────────────────────
@st.cache_data(show_spinner='Loading dataset …')
def load_car_data() -> pd.DataFrame:
    return pd.read_excel(CAR_DATA_PATH)

@st.cache_resource(show_spinner='Loading model …')
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource(show_spinner='Loading preprocessors …')
def load_preprocessors():
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return label_encoder, scaler

# ── Prediction helper ─────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    'Fuel type', 'Body type', 'transmission',
    'model', 'variantName', 'Insurance Validity', 'City',
]

# Must match the column order used during model training
FEATURE_ORDER = [
    'Fuel type', 'Body type', 'Kilometers driven', 'transmission',
    'ownerNo', 'model', 'modelYear', 'variantName',
    'Registration Year', 'Insurance Validity',
    'Mileage(kmpl)', 'Engine(CC)', 'Max Power(bhp)', 'Torque(Nm)', 'City',
]

def predict_price(features: dict, car_data: pd.DataFrame) -> float:
    model = load_model()
    label_encoder, scaler = load_preprocessors()

    features_df = pd.DataFrame([features])[FEATURE_ORDER]

    # Encode each categorical column using the distribution from car_data
    for col in CATEGORICAL_COLS:
        label_encoder.fit(car_data[col].astype(str))
        features_df[col] = label_encoder.transform(features_df[col].astype(str))

    features_scaled = scaler.transform(features_df)
    return model.predict(features_scaled)[0]

# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    try:
        df = load_car_data()
    except FileNotFoundError:
        st.error(f'❌ Data file not found: `{CAR_DATA_PATH}`. Make sure it is in the same directory as this script.')
        st.stop()
    except Exception as e:
        st.error(f'❌ Failed to load car data: {e}')
        st.stop()

    def opts(col):
        return sorted(df[col].dropna().unique())

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header('About')
        st.write(
            'This tool uses a tuned **Random Forest** model to estimate the '
            'resale price of a used car based on vehicle features.'
        )
        st.markdown('**Developed by Avijit Jana**')
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

        # with st.expander('Dataset Info'):
        #     st.write(df.shape)
        #     st.write(df.dtypes)

    # ── Header ────────────────────────────────────────────────────────────────
    st.header("🚗 :orange[_CarDekho_] Resale Car Price Prediction 🚗", divider="red")
    st.write('Fill in the car details below and click **Predict Price**.')

    # ── Input form ────────────────────────────────────────────────────────────
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

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
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

        with st.expander('📋 Input Summary'):
            st.dataframe(
                pd.DataFrame([features]).T.rename(columns={0: 'Selected Value'}),
                use_container_width=True,
            )

if __name__ == '__main__':
    main()
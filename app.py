import streamlit as st
import pandas as pd
import joblib

# Load trained Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Load label encoders
le_property = joblib.load('le_property.pkl')
le_location = joblib.load('le_location.pkl')
le_city = joblib.load('le_city.pkl')
le_province = joblib.load('le_province.pkl')
le_purpose = joblib.load('le_purpose.pkl')

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 Pakistan House Price Prediction")
st.markdown(
    "<h4 style='text-align: center; color: 	#FFFFFF;'>Trained on Real Data from <b>Zameen.com</b></h4>",
    unsafe_allow_html=True
)

st.markdown("Predict house prices based on full property details using a trained machine learning model.")

# Input features
area = st.number_input("Area (sqft)", min_value=100.0, value=3000.0, step=50.0)
bedrooms = st.number_input("Bedrooms", min_value=1, value=4)
baths = st.number_input("Bathrooms", min_value=1, value=3)
house_age = st.slider("House Age (years)", 0, 100, value=5)
latitude = st.number_input("Latitude", value=33.6844, format="%.4f")
longitude = st.number_input("Longitude", value=73.0479, format="%.4f")

# Dropdowns for categorical features
property_type = st.selectbox("Property Type", le_property.classes_)
location = st.selectbox("Location", le_location.classes_)
city = st.selectbox("City", le_city.classes_)
province = st.selectbox("Province", le_province.classes_)
purpose = st.selectbox("Purpose", le_purpose.classes_)

# Predict button
if st.button("📊 Predict Price"):
    try:
        # Encode categorical inputs
        input_data = pd.DataFrame({
            'area': [area],
            'bedrooms': [bedrooms],
            'baths': [baths],
            'house_age': [house_age],
            'latitude': [latitude],
            'longitude': [longitude],
            'property_type': [le_property.transform([property_type])[0]],
            'location': [le_location.transform([location])[0]],
            'city': [le_city.transform([city])[0]],
            'province_name': [le_province.transform([province])[0]],
            'purpose': [le_purpose.transform([purpose])[0]]
        })

        # Ensure columns match model input
        input_data = input_data[rf_model.feature_names_in_]

        # Make prediction
        predicted_price = rf_model.predict(input_data)[0]

        # Display result
        st.success(f"🏷️ Predicted Price: PKR {predicted_price:,.0f}")
        st.info(f"💰 Estimated Price in Millions: {predicted_price / 1_000_000:.2f} million PKR")

    except Exception as e:
        st.error(f"❌ Error: {e}")
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f0f2f6;
            padding: 10px;
            text-align: center;
            color: #4b4b4b;
            font-size: 14px;
        }
    </style>
    <div class="footer">
        Made with ❤️ by <b>Shahwaiz</b>
    </div>
""", unsafe_allow_html=True)

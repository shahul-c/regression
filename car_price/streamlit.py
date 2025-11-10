import streamlit as st
import pickle 
import numpy as np


#load trained model and encoders

model = pickle.load(open('model.pkl', 'rb'))
Fuel_Type_en = pickle.load(open('Fuel_Type.pkl', 'rb'))
Transmission_en = pickle.load(open('Transmission.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


st.set_page_config(page_title="Car Price Prediction")

st.title("Car Price Prediction App")
st.write("Enter the car details below to estimate its selling price:")

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    Car_Age = st.text_input("Enter the Age of Car", "5")
    Present_Price = st.text_input("Present Price (in lakhs)", "5.0")
    Kms_Driven = st.text_input("Kilometers Driven", "30000")

with col2:
    Fuel_Type = st.selectbox("Fuel Type", ("Petrol", "Diesel", "CNG"))
    Transmission = st.selectbox("Transmission", ("Manual", "Automatic"))
    Owner = st.text_input("Number of Previous Owners", "0")
    Seller_Type_Individual = st.selectbox("Seller Type", ("Dealer", "Individual"))



try:
    Car_Age = int(Car_Age)
    Present_Price = float(Present_Price)
    Kms_Driven = float(Kms_Driven)
    Owner = int(Owner)

    # Encode categorical variables
    Fuel_Type_val = Fuel_Type_en.transform([Fuel_Type])[0]
    Transmission_val = Transmission_en.transform([Transmission])[0]
    Seller_Type_Individual_val = 1 if Seller_Type_Individual == "Individual" else 0

     # Prepare input data
    details = [Car_Age, Present_Price, Kms_Driven, Fuel_Type_val, Transmission_val, Owner, Seller_Type_Individual_val]
    data_out = np.array(details).reshape(1, -1)
    data_scaled = scaler.transform(data_out)

    if st.button("Predict Car Price"):
        prediction = model.predict(data_scaled)[0]
        st.success(f"Estimated Car Price:{round(prediction, 2)}")


except ValueError:
    st.warning("Please enter valid numeric values for Year, Price, KM Driven, and Owner.")


    
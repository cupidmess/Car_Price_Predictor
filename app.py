import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

model = pk.load(open('model.pkl','rb'))

st.header('Car Price Prediction ML Model')

cars_data = pd.read_csv('Cardetails.csv')
# Extract numeric mileage from strings like "23.4 kmpl"
cars_data['mileage'] = cars_data['mileage'].astype(str).str.extract('(\d+\.?\d*)').astype(float)

# Use median mileage as default (since users don't input it)
default_mileage = cars_data['mileage'].median()

# Extract brand and model
def get_brand(name):
    return name.split(" ")[0]

def get_model(name):
    return " ".join(name.split(" ")[1:])

cars_data['brand'] = cars_data['name'].apply(get_brand)
cars_data['model'] = cars_data['name'].apply(get_model)

# Brand dropdown (ONLY ONE)
brand = st.selectbox('Select Car Brand', cars_data['brand'].unique())

# Model dropdown based on selected brand 
filtered_models = cars_data[cars_data['brand'] == brand]['model'].unique()
model_name = st.selectbox('Select Car Model', filtered_models)

# Use brand as 'name' for ML model
name = brand

year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())

engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)

# Predict button
if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[name, year, km_driven, fuel, seller_type, transmission, owner, default_mileage, engine, max_power, seats]],
    columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner',
             'mileage', 'engine', 'max_power', 'seats']
)


    
    # Encode categorical values
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                       'Fourth & Above Owner', 'Test Drive Car'],
                                      [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],
                                     [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],
                                            [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'],
                                             [1, 2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                      'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                      'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                      'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                      'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                                     inplace=True)

    car_price = model.predict(input_data_model)

    st.markdown('Car Price is going to be ' + str(int(car_price[0])))

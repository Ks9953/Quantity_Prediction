import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

# Define the file path relative to the script
file_path = 'Coal_Data.xlsx'

# Check if the file exists
if os.path.exists(file_path):
    # Load data
    data = pd.read_excel(file_path)
    
    # Strip spaces from column names
    data.columns = data.columns.str.strip()

    # Check if 'Month' column exists
    if 'Month' in data.columns:
        data['Month'] = data['Month'].str.strip()
        st.write("Data loaded successfully.")
    else:
        st.error("The dataset does not contain a 'Month' column.")
        st.stop()

    # Encoding categorical features
    le_bidder = LabelEncoder()
    le_month = LabelEncoder()
    data['Bidder Name Encoded'] = le_bidder.fit_transform(data['Bidder Name'])
    data['Month Encoded'] = le_month.fit_transform(data['Month'])

    # Selecting features and target
    X = data[['Bidder Name Encoded', 'Month Encoded']]
    y = data['Allocated Qty']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction function
    def predict_quantity(bidder_name, month):
        # Encode the bidder name and month
        if bidder_name in le_bidder.classes_ and month in le_month.classes_:
            bidder_name_encoded = le_bidder.transform([bidder_name])[0]
            month_encoded = le_month.transform([month])[0]
        else:
            st.error(f"Input not recognized. Please check your Bidder Name or Month.")
            return None

        # Prepare the input for prediction
        input_data = np.array([[bidder_name_encoded, month_encoded]])
        prediction = model.predict(input_data)
        
        return prediction[0]

    # User input through Streamlit
    st.title("Coal Bidder Quantity Prediction")

    bidder_name_input = st.selectbox("Select Bidder Name", data['Bidder Name'].unique())
    month_input = st.selectbox("Select Month", data['Month'].unique())

    if st.button("Predict Quantity"):
        predicted_quantity = predict_quantity(bidder_name_input, month_input)
        
        if predicted_quantity is not None:
            st.success(f"Predicted Allocated Quantity for {bidder_name_input} in {month_input}: {predicted_quantity:.2f}")
else:
    st.error(f"File '{file_path}' not found. Please ensure the file is in the correct directory.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Define file path relative to the script
file_path = 'Coal_Data.xlsx'

# Load and preprocess data
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip()  # Strip spaces from column names
    if 'Month' in data.columns:
        data['Month'] = data['Month'].str.strip()
        return data
    else:
        st.error("The dataset does not contain a 'Month' column.")
        st.stop()

# Encode features and train the model
@st.cache_resource
def train_model(data):
    # Encode categorical features
    le_mine = LabelEncoder()
    le_grade = LabelEncoder()
    le_bidder = LabelEncoder()
    le_month = LabelEncoder()

    data['Mine Name Encoded'] = le_mine.fit_transform(data['Mine Name'])
    data['Grade Encoded'] = le_grade.fit_transform(data['Grade'])
    data['Bidder Name Encoded'] = le_bidder.fit_transform(data['Bidder Name'])
    data['Month Encoded'] = le_month.fit_transform(data['Month'])

    # Selecting features and target
    features = ['Mine Name Encoded', 'Grade Encoded', 'Bidder Name Encoded', 'Quantity Offered', 'Month Encoded']
    X = data[features]
    y = data['Allocated Qty']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, le_mine, le_grade, le_bidder, le_month

# Prediction function
def predict_quantity(model, le_mine, le_grade, le_bidder, le_month, mine_name, grade, bidder_name, quantity_offered, month):
    # Encode inputs
    if all(x in le_mine.classes_ for x in [mine_name]) and \
       all(x in le_grade.classes_ for x in [grade]) and \
       all(x in le_bidder.classes_ for x in [bidder_name]) and \
       month in le_month.classes_:
        
        mine_encoded = le_mine.transform([mine_name])[0]
        grade_encoded = le_grade.transform([grade])[0]
        bidder_encoded = le_bidder.transform([bidder_name])[0]
        month_encoded = le_month.transform([month])[0]
        
        # Prepare input data for prediction
        input_data = np.array([[mine_encoded, grade_encoded, bidder_encoded, quantity_offered, month_encoded]])
        prediction = model.predict(input_data)
        
        return prediction[0]
    else:
        st.error("Input not recognized. Please check your inputs.")
        return None

# Streamlit App
def main():
    st.title("Coal Bidder Quantity Prediction")

    # Load data and train model
    data = load_data(file_path)
    model, le_mine, le_grade, le_bidder, le_month = train_model(data)

    # User inputs
    mine_name_input = st.selectbox("Select Mine Name", data['Mine Name'].unique())
    grade_input = st.selectbox("Select Grade", data['Grade'].unique())
    bidder_name_input = st.selectbox("Select Bidder Name", data['Bidder Name'].unique())
    quantity_offered_input = st.number_input("Quantity Offered", min_value=0)
    month_input = st.selectbox("Select Month", data['Month'].unique())

    if st.button("Predict Quantity"):
        predicted_quantity = predict_quantity(model, le_mine, le_grade, le_bidder, le_month,
                                              mine_name_input, grade_input, bidder_name_input,
                                              quantity_offered_input, month_input)
        
        if predicted_quantity is not None:
            st.success(f"Predicted Allocated Quantity: {predicted_quantity:.2f}")

if __name__ == "__main__":
    main()

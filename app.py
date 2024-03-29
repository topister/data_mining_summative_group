import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# # Load the saved model
model = joblib.load("compressed_bank_account_prediction_model.pkl")


# Load the encoder
encoder = joblib.load("encoder.pkl")

# Define the Streamlit app
def main():
    # Set the title of the app
    st.title("Bank Account Likelihood Predictor")

    # Add a brief description
    st.write("This app predicts the likelihood of a person having a bank account.")

    # Add input fields for user to enter data
    country = st.selectbox("Select Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
    year = st.number_input("Enter Year", value=2018)
    location_type = st.selectbox("Select Location Type", ["Rural", "Urban"])
    cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
    household_size = st.number_input("Enter Household Size", value=0)
    age_of_respondent = st.number_input("Enter Age of Respondent", value=0)
    gender_of_respondent = st.selectbox("Select Gender", ["Male", "Female"])
    relationship_with_head = st.selectbox("Relationship with Head", ["Head of Household", "Other relative", "Other non-relatives", "Parent", "Spouse"])
    marital_status = st.selectbox("Marital Status", ["Married/Living together", "Single/Never Married", "Widowed", "Divorced/Seperated", "Dont know"])
    education_level = st.selectbox("Education Level", ["No formal education", "Primary education", "Secondary education", "Vocational/Specialised training", "Tertiary education", "Other/Dont know/RTA"])
    job_type = st.selectbox("Job Type", ["Farming and Fishing", "Formally employed Government", "Formally employed Private", "Government Dependent", "Informally employed", "No Income", "Other Income", "Remittance Dependent", "Self employed"])

    # Once user provides input, make prediction
    if st.button("Predict"):
        # Create a DataFrame with user input
        user_data = pd.DataFrame({
            "country": [country],
            "year": [year],
            "location_type": [location_type],
            "cellphone_access": [cellphone_access],
            "household_size": [household_size],
            "age_of_respondent": [age_of_respondent],
            "gender_of_respondent": [gender_of_respondent],
            "relationship_with_head": [relationship_with_head],
            "marital_status": [marital_status],
            "education_level": [education_level],
            "job_type": [job_type]
        })

        

        # Data Preprocessing for test data (similar to what we did for training data)
        user_data.dropna(inplace=True)  # Drop rows with missing values for simplicity
        
        # Apply one-hot encoding to categorical features in user_data
        encoded_user_data = encoder.transform(user_data[['country', 'location_type', 'cellphone_access', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']])

        # Convert the encoded data to a DataFrame
        encoded_df = pd.DataFrame(encoded_user_data.toarray(), columns=encoder.get_feature_names_out(['country', 'location_type', 'cellphone_access', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']))

        # Drop the original categorical features from user_data
        user_data.drop(['country', 'location_type', 'cellphone_access', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type'], axis=1, inplace=True)

        # Concatenate the encoded data with the original user_data
        user_data = pd.concat([user_data.reset_index(drop=True), encoded_df], axis=1)


        # Make prediction
        prediction = model.predict(user_data)

        # Make predictions on test data
        # predictions = model.predict(test_data)
        # st.write(prediction)

        # # Show predictions
        # print(prediction)

        if prediction == ['Yes']:
            st.success("Likelihood of having a bank account: High")
        else:
            st.success("Likelihood of having a bank account: Low")
        

       
        # Debugging output
        # st.write("Encoded Data:")
        # st.write(encoded_df)
        # st.write("Feature Names:")
        # st.write(encoded_df.columns)


# Run the app
if __name__ == "__main__":
    main()

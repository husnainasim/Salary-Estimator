import streamlit as st 
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model=tf.keras.models.load_model('regression_model.h5')

## loading the encoder and scaler 
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)
    
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scalar=pickle.load(file)
    

# Streamlit app
st.title('Estimated Salary Prediction')

# User Input 
geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender(0:Female & 1:Male)',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance($)')
credit_score=st.number_input('Credit Score')
exited=st.selectbox('Exited(0=No;1=Yes)',[0,1])
# estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


# Prepare the input data --> This means that we need a dictionary 
#Why do we use ([gender])[0] here?
# 'Gender': [label_encoder_gender.transform([gender])[0]]
# Understanding transform()
# label_encoder_gender is an instance of LabelEncoder.
# It is used to convert categorical labels (like "Male", "Female") into numerical values (0, 1).
# Why [gender]?
# label_encoder_gender.transform() requires a list of values (even if we have just one value).
# So we wrap gender in a list: label_encoder_gender.transform([gender]).
# Why [0]?
# label_encoder_gender.transform([gender]) returns an array with a single-element.
# Example:

# label_encoder_gender.transform(["Male"])  # Output: array([1])
# Since we only have one value, we extract it using [0] to get 1 instead of [1].
# ✅ Correct:


# label_encoder_gender.transform(["Male"])[0]  # Returns 1
# ❌ Incorrect:


# label_encoder_gender.transform("Male")  # ❌ Will cause an error
# Because "Male" is not a list, LabelEncoder expects a list-like structure.
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[exited]
    
})

#One hot encoded 'Geography'

geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#Combine one hot encoded columns with the input data 
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled=scalar.transform(input_data)


# st.write("User Inputs:")
# st.write(f"Geography: {geography}")
# st.write(f"Gender: {gender}")
# st.write(f"Age: {age}")
# st.write(f"Balance: {balance}")
# st.write(f"Credit Score: {credit_score}")
# st.write(f"Exited: {exited}")
# st.write(f"Tenure: {tenure}")
# st.write(f"Number of Products: {num_of_products}")
# st.write(f"Has Credit Card: {has_cr_card}")
# st.write(f"Is Active Member: {is_active_member}")

# st.write("Encoded Values:")
# st.write(f"Encoded Gender: {label_encoder_gender.transform([gender])[0]}")
# st.write(f"One-Hot Encoded Geography: {geo_encoded_df}")

# st.write("Final Input Data Before Scaling:")
# st.write(input_data)

# st.write("Scaled Input Data:")
# st.write(input_data_scaled)

# st.write("Feature Order:", list(input_data.columns))

#!predict the churn 
prediction=model.predict(input_data_scaled)
predicted_salary=prediction[0][0]

st.write("Model Raw Output (Prediction):")
st.write(prediction[0][0])


st.write(f'Predicted Estimated Salary: ${predicted_salary:.2f}')


import streamlit as st
import joblib
import numpy as np

def load_model():
    model_path = 'svm_model.pkl'
    svm_model = joblib.load(model_path)  
    return svm_model

def user_input_features():
    st.sidebar.header('User Input Features')
    
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=25)
    estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0, max_value=1000000, value=50000)
    
    gender = 0 if gender == 'Male' else 1
    
    input_data = np.array([[gender, age, estimated_salary]])
    
    return input_data

def display_prediction(prediction):
    st.write(f'**Prediction**: {prediction[0]}') 

def main():
    st.title('SVM Model Prediction App')
    st.write("""
    This app allows you to input data and get predictions using the pre-trained Support Vector Machine (SVM) model.
    """)
    
    model = load_model()
    
    input_data = user_input_features()
    
    if st.button('Predict'):
        prediction = model.predict(input_data)
        display_prediction(prediction)
    
    st.write("""
    ## Instructions:
    - Use the sidebar to input features for prediction.
    - Click 'Predict' to see the model's output.
    """)

if __name__ == '__main__':
    main()  

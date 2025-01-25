import streamlit as st
import joblib
import numpy as np

# 1. Load the pre-trained SVM model
def load_model():
    # Load the model from the .pkl file
    model_path = 'svm_model.pkl'  # Use the correct path
    svm_model = joblib.load(model_path)  # Make sure this .pkl file is in the same directory
    return svm_model

# 2. Frontend - User input form
def user_input_features():
    st.sidebar.header('User Input Features')
    
    # Create input fields for gender, age, and estimated salary
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=25)
    estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0, max_value=1000000, value=50000)
    
    # Convert gender to numerical values if needed for the model (e.g., Male = 0, Female = 1)
    gender = 0 if gender == 'Male' else 1
    
    # Create a NumPy array from user input to match the expected input format for the SVM model
    input_data = np.array([[gender, age, estimated_salary]])
    
    return input_data

# 3. Display prediction result
def display_prediction(prediction):
    st.write(f'**Prediction**: {prediction[0]}')  # Assuming the model returns a single prediction

# 4. Main function for the app
def main():
    st.title('SVM Model Prediction App')
    st.write('''
    This app allows you to input data and get predictions using the pre-trained Support Vector Machine (SVM) model.
    ''')
    
    # Load the SVM model
    model = load_model()
    
    # Get user input
    input_data = user_input_features()
    
    # Predict when the user clicks 'Predict'
    if st.button('Predict'):
        prediction = model.predict(input_data)
        display_prediction(prediction)
    
    st.write('''
    ## Instructions:
    - Use the sidebar to input features for prediction.
    - Click 'Predict' to see the model's output.
    ''')

if __name__ == '__main__':
    main()
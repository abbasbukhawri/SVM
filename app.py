import streamlit as st
import joblib
import numpy as np
import os

# 1. Load the pre-trained SVM model with error handling
def load_model():
    model_path = 'svm_model.pkl'  # Adjust this if the path is different
    if os.path.exists(model_path):
        try:
            # Load the model from the .pkl file
            svm_model = joblib.load(model_path)
            return svm_model
        except Exception as e:
            st.error(f"An error occurred while loading the model: {e}")
            return None
    else:
        st.error(f"Model file not found: {model_path}")
        return None

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
    try:
        st.write(f'**Prediction**: {prediction[0]}')  # Assuming the model returns a single prediction
    except Exception as e:
        st.error(f"An error occurred while displaying the prediction: {e}")

# 4. Main function for the app
def main():
    st.title('SVM Model Prediction App')
    st.write('''
    This app allows you to input data and get predictions using the pre-trained Support Vector Machine (SVM) model.
    ''')
    
    # Load the SVM model
    model = load_model()
    
    if model is not None:
        # Get user input
        input_data = user_input_features()
        
        # Predict when the user clicks 'Predict'
        if st.button('Predict'):
            try:
                prediction = model.predict(input_data)
                display_prediction(prediction)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.write('Model loading failed. Please check the model file path or the model itself.')
    
    st.write('''
    ## Instructions:
    - Use the sidebar to input features for prediction.
    - Click 'Predict' to see the model's output.
    ''')

if __name__ == '__main__':
    main()

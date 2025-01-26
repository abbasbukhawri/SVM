import joblib
import streamlit as st
import numpy as np

# 1. Load the pre-trained SVM model with safe loading
def load_model():
    model_path = 'svm_model.pkl'  # Adjust the path if needed

    try:
        # Load the model, ignoring errors due to missing modules
        svm_model = joblib.load(model_path, mmap_mode=None)  # Safe loading (ignoring mmap_mode)
        return svm_model
    except ModuleNotFoundError as e:
        st.error(f"Model could not be loaded due to a missing module: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# 2. User input and prediction remains unchanged
def user_input_features():
    st.sidebar.header('User Input Features')
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=25)
    estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0, max_value=1000000, value=50000)
    
    # Convert gender to numerical values if needed for the model (e.g., Male = 0, Female = 1)
    gender = 0 if gender == 'Male' else 1
    
    # Create a NumPy array from user input
    input_data = np.array([[gender, age, estimated_salary]])
    return input_data

def display_prediction(prediction):
    if prediction is not None:
        st.write(f'**Prediction**: {prediction[0]}')

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
            prediction = model.predict(input_data)
            display_prediction(prediction)
    else:
        st.error("The model could not be loaded. Please check the logs for more details.")
    
    st.write('''
    ## Instructions:
    - Use the sidebar to input features for prediction.
    - Click 'Predict' to see the model's output.
    ''')

if __name__ == '__main__':
    main()

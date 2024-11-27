import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Set custom theme using st.set_page_config
st.set_page_config(
    page_title="Ulcerative Colitis Prediction",
    page_icon="🔬",  # Icon in browser tab
    layout="wide",  # Wide layout
    initial_sidebar_state="collapsed",  # Collapse sidebar by default
)

# Load the saved model
model = joblib.load('xgb_ulcerative_colitis_model.pkl')

# Initialize LabelEncoder
encoder = LabelEncoder()
encoder.fit(["Low", "Moderate", "High"])  # Match with training data categories

# Classification key for the prediction
classification_key = {
    0: "No Ulcerative Colitis",
    1: "Ulcerative Colitis Detected"
}

def preprocess_input(input_data, encoder):
    """
    Preprocess input data to match the training data structure.
    Args:
        input_data (dict): Input data with keys as feature names.
        encoder (LabelEncoder): Encoder for categorical variables.

    Returns:
        pd.DataFrame: Preprocessed input data.
    """
    # Convert input dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in ['Dietary_Fiber_Intake', 'Physical_Activity_Level']:
        input_df[col] = encoder.transform(input_df[col])

    return input_df

def predict_ulcerative_colitis(model, input_data, encoder, classification_key):
    """
    Predict Ulcerative Colitis given the input features.

    Args:
        model: Trained XGBoost model.
        input_data (dict): Input data with feature names as keys and their values.
        encoder (LabelEncoder): Encoder for categorical variables.
        classification_key (dict): Mapping of prediction output to human-readable classes.

    Returns:
        str: Human-readable prediction result.
    """
    # Preprocess the input
    processed_input = preprocess_input(input_data, encoder)
    
    # Make prediction
    prediction = model.predict(processed_input)

    # Return human-readable classification
    return classification_key[int(prediction[0])]

# Function to display the landing page
def landing_page():
    # Set background color and animations using CSS
    st.markdown("""
        <style>
            .reportview-container {
                background-color: #e0f7fa;  /* Light cyan background */
            }
            .sidebar .sidebar-content {
                background-color: #00796b;  /* Teal background for sidebar */
            }
            h1, h2, h3, p {
                color: #004d40;  /* Dark teal text color */
            }
            /* Icon animations */
            .icon {
                font-size: 50px;
                color: #00796b;
                transition: transform 0.5s ease;
            }
            .icon:hover {
                transform: rotate(360deg);  /* Rotate the icon on hover */
            }
            .icon-fade {
                animation: fadeIn 1.5s ease-in-out;
            }
            @keyframes fadeIn {
                0% { opacity: 0; }
                100% { opacity: 1; }
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Ulcerative Colitis Prediction")
    st.markdown("""
        <h2 style="text-align: center;">Welcome to the Ulcerative Colitis Prediction App</h2>
        <p style="text-align: center;">
            This app uses machine learning to predict the likelihood of Ulcerative Colitis based on key health and lifestyle factors.
        </p>
        <hr>
    """, unsafe_allow_html=True)

    # Add Image
    st.image('https://d2jx2rerrg6sh3.cloudfront.net/images/news/ImageForNews_763967_16995014253356857.jpg', use_container_width=True)

    # About the Project
    st.subheader("About the Project")
    st.markdown("""
        Ulcerative Colitis is a chronic condition that affects the large intestine and rectum, causing inflammation and ulcers.
        Predicting the likelihood of Ulcerative Colitis early can help in preventive measures and improve the management of the condition.
    
        This app uses a machine learning model trained on various factors such as:
        - Age
        - Family History
        - BMI (Body Mass Index)
        - Smoking History
        - Dietary Fiber Intake
        - Physical Activity Level
        - Stress Level
        - Medication Use
    
        By entering these details, the app will predict whether there is a risk of developing Ulcerative Colitis or not.
    """)

    # Add animated icons
    st.markdown("""
        <div style="display: flex; justify-content: center; gap: 20px;">
            <i class="fas fa-users icon icon-fade"></i>
            <i class="fas fa-heartbeat icon icon-fade"></i>
            <i class="fas fa-brain icon icon-fade"></i>
        </div>
        <div style="text-align: center;">
            <p>Early detection of health conditions | Improve patient care | Powered by Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)

    # Button to navigate to the prediction page
    if st.button("Go to Prediction Page"):
        st.session_state.page = "prediction"

# Function to display the prediction page
def prediction_page():
    # Set background color and animations using CSS
    st.markdown("""
        <style>
            .reportview-container {
                background-color: #e0f7fa;  /* Light cyan background */
            }
            .sidebar .sidebar-content {
                background-color: #00796b;  /* Teal background for sidebar */
            }
            h1, h2, h3, p {
                color: #004d40;  /* Dark teal text color */
            }
            /* Icon animations */
            .icon {
                font-size: 50px;
                color: #00796b;
                transition: transform 0.5s ease;
            }
            .icon:hover {
                transform: rotate(360deg);  /* Rotate the icon on hover */
            }
            .icon-fade {
                animation: fadeIn 1.5s ease-in-out;
            }
            @keyframes fadeIn {
                0% { opacity: 0; }
                100% { opacity: 1; }
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Ulcerative Colitis Prediction - Enter your Information")

    # Input form for prediction
    age = st.number_input("Age", min_value=1, max_value=120, value=69)
    family_history = st.selectbox("Family History", [0, 1])  # 0 = No, 1 = Yes
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.69)
    smoking_history = st.selectbox("Smoking History", [0, 1])  # 0 = No, 1 = Yes
    dietary_fiber_intake = st.selectbox("Dietary Fiber Intake", ["Low", "Moderate", "High"])
    physical_activity_level = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    stress_level = st.slider("Stress Level", min_value=0, max_value=10, value=7)
    medication_use = st.selectbox("Medication Use", [0, 1])  # 0 = No, 1 = Yes

    # Prepare the input data
    input_data = {
        "Age": age,
        "Family_History": family_history,
        "BMI": bmi,
        "Smoking_History": smoking_history,
        "Dietary_Fiber_Intake": dietary_fiber_intake,
        "Physical_Activity_Level": physical_activity_level,
        "Stress_Level": stress_level,
        "Medication_Use": medication_use
    }

    # Predict and display result
    if st.button("Predict"):
        prediction = predict_ulcerative_colitis(model, input_data, encoder, classification_key)
        st.write(f"Prediction: {prediction}")
    
    # Button to go back to the landing page
    if st.button("Back to Landing Page"):
        st.session_state.page = "landing"

# Main app logic
if "page" not in st.session_state:
    st.session_state.page = "landing"  # Default to landing page

if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "prediction":
    prediction_page()

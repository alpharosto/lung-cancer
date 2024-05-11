import streamlit as st
import numpy as np
import pickle

# Load the pickled model
with open('lung_cancer_predictor_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def main():
    st.write("""# Lung Cancer Detection Web App""")

    st.write("""
    ## About
    
    Lung cancer is a type of cancer that begins in the lungs and most often occurs in people who smoke. Two major types of lung cancer are non-small cell lung cancer and small cell lung cancer. Causes of lung cancer include smoking, second-hand smoke, exposure to certain toxins and family history. Symptoms include a cough (often with blood), chest pain, wheezing and weight loss. These symptoms often don't appear until the cancer is advanced. Treatments vary but may include surgery, chemotherapy, radiation therapy, targeted drug therapy and immunotherapy.
    
    **This Streamlit App utilizes a Machine Learning model in order to detect lung cancer in patients based on the following criteria: age, gender, blood pressure, smoke, coughing, allergies, fatigue etc.** 
    
    The model is loaded directly into the app.
    
    
    """)

    st.sidebar.header('User Input Features')

    gender = st.sidebar.number_input("GENDER: Enter 1 for Male and 0 for Female", min_value=0, max_value=1)
    age = st.sidebar.slider("AGE: Enter your Age", min_value=1, max_value=100)
    smoking = st.sidebar.number_input("SMOKING: Enter 1 if you smoke or 0 if you don't smoke", min_value=0, max_value=1)
    yellow_finger = st.sidebar.number_input("YELLOW FINGERS: Enter 1 if you have yellow fingers or 0 if you don't", min_value=0, max_value=1)
    anxiety = st.sidebar.number_input("ANXIETY: Enter 1 if you have anxiety and 0 if you don't", min_value=0, max_value=1)
    peer = st.sidebar.number_input("PEER PRESSURE: Enter 1 if you feel you suffer from peer pressure or 0 if you don't", min_value=0, max_value=1)
    chronic = st.sidebar.number_input("CHRONIC DISEASE: Enter 1 if you suffer from a chronic disease or O if you don't", min_value=0, max_value=1)
    fatigue = st.sidebar.number_input("FATIGUE: Enter 1 if you have fatigue or 0 if you don't", min_value=0, max_value=1)
    allergy = st.sidebar.number_input("ALLERGY: Enter 1 if you have some sort of allergy or 0 if you don't", min_value=0, max_value=1)
    wheezing = st.sidebar.number_input("WHEEZING: Enter 1 if you wheeze or 0 if you don't", min_value=0, max_value=1)
    alcohol =  st.sidebar.number_input("ALCOHOL CONSUMPTION: Enter 1 if you consume alcohol or 0 if you don't", min_value=0, max_value=1)
    coughing = st.sidebar.number_input("COUGHING: Enter 1 if you cough a lot or 0 if you don't", min_value=0, max_value=1)
    breath = st.sidebar.number_input("SHORTNESS OF BREATH: Enter 1 if you suffer from shortness of breath or 0 if you don't", min_value=0, max_value=1)
    swallow =  st.sidebar.number_input("SWALLOWING DIFFICULTY: Enter 1 if you have difficulty swallowing or 0 if you don't", min_value=0, max_value=1)
    chest =  st.sidebar.number_input("CHEST PAIN: Enter 1 if you have chest pain or 0 if you don't", min_value=0, max_value=1)

    st.write("Model Loaded Successfully:", isinstance(model, type))

    if st.button('Detection Result'):
        # Create a DataFrame from user input
        user_data = np.array([gender, age, smoking, yellow_finger, anxiety, peer, chronic, fatigue, allergy, wheezing, alcohol, coughing, breath, swallow, chest]).reshape(1, -1)

        # Predict
        result = model.predict(user_data)
        st.write(result[0])  # Since result is likely an array, just output the first element

if __name__ == "__main__":
    main()

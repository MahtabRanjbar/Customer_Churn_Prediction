import streamlit as st
import requests

API_URL = 'http://localhost:8000'

def retrain_model():
    st.header("🔄 Retrain Model")
    st.markdown("Upload a new CSV file with training data to retrain the model.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if st.button("Retrain Model"):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post(f"{API_URL}/retrain", files={"file": uploaded_file})
                if response.status_code == 200:
                    result = response.json()
                    st.success("Model retrained successfully!")
                    st.write("Accuracy:", result.get('accuracy'))
                    st.write("Classification Report:", result.get('classification_report'))
                else:
                    st.error("Retraining failed. Check the file format and data structure.")
            except Exception as e:
                st.error(f"Error: {e}")

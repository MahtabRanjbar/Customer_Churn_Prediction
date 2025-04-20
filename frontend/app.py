
import streamlit as st
from utils import load_data

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Customer Churn Dashboard")
    st.markdown("""
    Welcome to the **Customer Churn Dashboard**. This interactive tool allows you to explore key insights from the Telco Customer Churn data.
    Use the sidebar to filter data and switch between tabs for an in-depth look at visualizations, insights, predictions, and more.
    """)

if __name__ == "__main__":
    main()
    
    # Load and filter data once
    data = load_data()
    filtered_data = filter_data(data)


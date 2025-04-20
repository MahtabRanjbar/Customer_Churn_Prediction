import streamlit as st
import requests
import os

# Try to read from the environment, defaulting to the Compose service name:
API_URL = os.getenv("API_URL", "http://backend:8000")

def prediction_form():
    st.header("🔮 Customer Churn Prediction")
    st.markdown(
        "Fill in the customer’s profile below. "
        "Hover over any input for guidance on how to choose."
    )
    with st.form("prediction_form"):
        # --- Numeric Inputs ---
        tenure = st.number_input(
            "⏳ Tenure (months)",
            min_value=0, max_value=100, value=12,
            help="How many months this customer has been with the company."
        )
        monthly_charges = st.number_input(
            "💰 Monthly Charges ($)",
            min_value=0.0, value=70.0, format="%.2f",
            help="What the customer pays every month, on average."
        )

        # --- Core Categorical Inputs ---
        contract = st.selectbox(
            "📄 Contract Type",
            ["Month-to-month", "One year", "Two year"],
            help=(
                "Month-to-month contracts often see higher churn.\n"
                "One- and two-year contracts generally retain customers longer."
            )
        )
        gender = st.selectbox(
            "👤 Gender",
            ["Female", "Male"],
            help="Select the customer’s reported gender."
        )
        senior_citizen = st.selectbox(
            "🧓 Senior Citizen",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether the customer is a senior citizen (1 = Yes)."
        )
        partner = st.selectbox(
            "💑 Has Partner?",
            ["Yes", "No"],
            help="Whether the customer has a partner."
        )
        dependents = st.selectbox(
            "👨‍👩‍👧 Has Dependents?",
            ["Yes", "No"],
            help="Whether the customer has dependents."
        )

        # --- Network & Billing Inputs ---
        internet_service = st.selectbox(
            "🌐 Internet Service",
            ["DSL", "Fiber optic", "No"],
            help=(
                "Type of internet service the customer uses. "
                "Customers with Fiber optic historically churn more."
            )
        )
        paperless_billing = st.selectbox(
            "🧾 Paperless Billing?",
            ["Yes", "No"],
            help="Whether the customer uses paperless billing."
        )
        payment_method = st.selectbox(
            "💳 Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            help=(
                "How the customer pays their bill. "
                "Electronic check users are often at higher churn risk."
            )
        )

        # --- Services Multiselect ---
        services_list = st.multiselect(
            "🛎️ Optional Services",
            ["Multiple Lines", "Online Security", "Online Backup",
             "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"],
            help=(
                "Choose all optional services the customer subscribes to. "
                "More services generally indicate lower churn risk."
            )
        )

        submitted = st.form_submit_button("Predict Churn")
        if submitted:
            payload = {
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "contract": contract,
                "gender": gender,
                "senior_citizen": senior_citizen,
                "partner": partner,
                "dependents": dependents,
                "internet_service": internet_service,
                "paperless_billing": paperless_billing,
                "payment_method": payment_method,
                "services_list": services_list
            }
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.success(
                        f"**Prediction:** {result['prediction']}  \n"
                        f"**Churn Probability:** {result['probability']:.2%}"
                    )
                else:
                    st.error(f"Prediction failed ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")

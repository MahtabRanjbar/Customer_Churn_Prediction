import streamlit as st

def filter_data(df):
    st.sidebar.markdown(
        """
        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 10px;">
            <h2 style="color: #111111; font-family: 'Segoe UI', sans-serif; text-align: center;">🔎 Filter Options</h2>
            <p style="font-size: 14px; text-align: center; color: #333333;">Use the options below to refine the data view</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    contracts = sorted(df['contract'].unique())
    selected_contracts = st.sidebar.multiselect("📄 Contract Type", contracts, default=contracts, key="contract_multiselect")
    df = df[df['contract'].isin(selected_contracts)]
    
    genders = sorted(df['gender'].unique())
    selected_genders = st.sidebar.multiselect("👤 Gender", genders, default=genders, key="gender_multiselect")
    df = df[df['gender'].isin(selected_genders)]
    
    payment_methods = sorted(df['payment_method'].unique())
    selected_payments = st.sidebar.multiselect("💳 Payment Method", payment_methods, default=payment_methods, key="payment_multiselect")
    df = df[df['payment_method'].isin(selected_payments)]
    
    internet_services = sorted(df['internet_service'].unique())
    selected_internet = st.sidebar.multiselect("🌐 Internet Service", internet_services, default=internet_services, key="internet_multiselect")
    df = df[df['internet_service'].isin(selected_internet)]
    
    partner_options = sorted(df['partner'].unique())
    selected_partner = st.sidebar.multiselect("💑 Partner", partner_options, default=partner_options, key="partner_multiselect")
    df = df[df['partner'].isin(selected_partner)]
    
    dependents_options = sorted(df['dependents'].unique())
    selected_dependents = st.sidebar.multiselect("👨‍👩‍👧 Dependents", dependents_options, default=dependents_options, key="dependents_multiselect")
    df = df[df['dependents'].isin(selected_dependents)]
    
    min_tenure, max_tenure = int(df['tenure'].min()), int(df['tenure'].max())
    tenure_range = st.sidebar.slider("⏳ Tenure Range (months)", min_value=min_tenure, max_value=max_tenure, value=(min_tenure, max_tenure), key="tenure_slider")
    df = df[(df['tenure'] >= tenure_range[0]) & (df['tenure'] <= tenure_range[1])]
    
    return df

# frontend/components/metrics.py

import streamlit as st

def display_summary_metrics(df):
    total_customers = df.shape[0]
    churn_rate = (df['churn'].str.lower() == 'yes').mean() * 100
    avg_monthly_charges = df['monthly_charges'].mean()
    avg_tenure = df['tenure'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", total_customers)
    col2.metric("📉 Churn Rate (%)", f"{churn_rate:.2f}%")
    col3.metric("💰 Avg. Monthly Charges", f"${avg_monthly_charges:.2f}")
    col4.metric("⏳ Avg. Tenure (months)", f"{avg_tenure:.1f}")

def download_filtered_data(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

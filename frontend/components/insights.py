# frontend/components/insights.py

import streamlit as st
import plotly.express as px

def compute_insights(df):
    insights = []
    overall_churn_rate = (df['churn'].str.lower() == 'yes').mean() * 100
    insights.append(f"**Overall Churn Rate:** {overall_churn_rate:.2f}% of customers have churned.")
    
    contract_churn = df.groupby('contract')['churn'].apply(lambda x: (x.str.lower()=='yes').mean() * 100).reset_index()
    for _, row in contract_churn.iterrows():
        insights.append(f"**Churn Rate for {row['contract']} Contracts:** {row['churn']:.2f}%")
    
    internet_churn = df.groupby('internet_service')['churn'].apply(lambda x: (x.str.lower()=='yes').mean() * 100).reset_index()
    for _, row in internet_churn.iterrows():
        insights.append(f"**Churn Rate for {row['internet_service']} Service:** {row['churn']:.2f}%")
    
    avg_charges = df.groupby('churn')['monthly_charges'].mean().reset_index()
    for _, row in avg_charges.iterrows():
        insights.append(f"**Avg. Monthly Charge for churn = '{row['churn']}':** ${row['monthly_charges']:.2f}")
    
    avg_tenure = df.groupby('churn')['tenure'].mean().reset_index()
    for _, row in avg_tenure.iterrows():
        insights.append(f"**Avg. Tenure for churn = '{row['churn']}':** {row['tenure']:.1f} months")
    
    return insights

def insights_tab(df):
    st.header("📊 Insights")
    
    col1, col2, col3 = st.columns(3)
    overall_churn = (df['churn'].str.lower() == 'yes').mean() * 100
    avg_charge_churned = df[df['churn'].str.lower()=='yes']['monthly_charges'].mean()
    avg_charge_non_churned = df[df['churn'].str.lower()=='no']['monthly_charges'].mean()
    col1.metric("📉 Overall Churn Rate", f"{overall_churn:.2f}%")
    col2.metric("💰 Avg. Charge (Churned)", f"${avg_charge_churned:.2f}")
    col3.metric("💵 Avg. Charge (Non-Churned)", f"${avg_charge_non_churned:.2f}")
    
    st.markdown("## Segment Analysis")
    seg_var = st.selectbox("Select a segmentation variable", ["contract", "internet_service", "gender", "payment_method"], key="segmentation_select")
    seg_churn = df.groupby(seg_var)['churn'].apply(lambda x: (x.str.lower()=='yes').mean()*100).reset_index(name='Churn Rate (%)')
    fig_seg = px.bar(
        seg_churn,
        x=seg_var,
        y='Churn Rate (%)',
        text='Churn Rate (%)',
        template="plotly_white",
        color=seg_var
    )
    fig_seg.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_seg.update_layout(
        yaxis=dict(title="Churn Rate (%)"),
        xaxis=dict(title=seg_var.capitalize()),
        title=f"Churn Rate by {seg_var.capitalize()}",
        title_font=dict(size=18, family="Segoe UI", color="#111111")
    )
    st.plotly_chart(fig_seg, use_container_width=True)
    
    st.markdown("## Correlation Analysis")
    numeric_features = df[['tenure', 'monthly_charges', 'total_charges']]
    corr = numeric_features.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        template="plotly_white",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("## Additional Analysis")
    st.markdown("""
    - **Contract Impact:** Month-to-month contracts show significantly higher churn rates compared to one-year or two-year contracts.
    - **Internet Service:** Fiber optic users tend to churn at a higher rate than DSL users or those with no internet service.
    - **Pricing Sensitivity:** Customers paying higher monthly charges are more likely to churn.
    - **Customer Loyalty:** Longer tenure is generally associated with lower churn rates.
    - **Demographic & Payment Insights:** Gender and payment method differences add further layers to the churn behavior.
    """)
    
    with st.expander("Deep Dive Analysis"):
        st.markdown("""
        **Deep Dive Insights:**
        - **Segmentation:** Detailed segmentation by contract type, internet service, and payment method reveals key drivers of churn.
        - **Correlation Patterns:** The correlation heatmap shows that while tenure and monthly charges have an inverse relationship, this trend varies across segments.
        - **Actionable Strategies:** These insights suggest targeted retention strategies—such as revising contract terms or pricing for high-risk segments—could reduce churn.
        - **Advanced Analysis:** Consider employing predictive modeling techniques (e.g., logistic regression or decision trees) to further quantify the impact of these factors.
        """)
        



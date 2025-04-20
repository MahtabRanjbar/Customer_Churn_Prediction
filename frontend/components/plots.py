import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans

def plot_monthly_charges_histogram(df):
    fig = px.histogram(
        df,
        x='monthly_charges',
        nbins=30,
        title="Monthly Charges Distribution",
        color='churn',
        marginal="box",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        hover_data={'tenure': True, 'total_charges': True}
    )
    fig.update_layout(
        xaxis_title="Monthly Charges ($)",
        yaxis_title="Number of Customers",
        title_font=dict(size=18, family="Segoe UI", color="#111111"),
        legend_title="Churn Status"
    )
    fig.update_traces(
        hovertemplate="<b>Monthly Charges:</b> %{x:$,.2f}<br><b>Count:</b> %{y}<br><b>Tenure:</b> %{customdata[0]} months<br><b>Total Charges:</b> $%{customdata[1]:,.2f}<extra></extra>"
    )
    return fig

def plot_tenure_vs_charges(df):
    fig = px.scatter(
        df,
        x='tenure',
        y='monthly_charges',
        color='churn',
        trendline="ols",
        title="Relationship Between Tenure and Monthly Charges",
        template="plotly_white",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Plotly,
        hover_data={'gender': True, 'contract': True}
    )
    fig.update_layout(
        xaxis_title="Tenure (Months)",
        yaxis_title="Monthly Charges ($)",
        title_font=dict(size=18, family="Segoe UI", color="#111111"),
        legend_title="Churn Status",
        hovermode="closest"
    )
    fig.update_traces(
        marker=dict(size=10, symbol="circle"),
        hovertemplate="<b>Tenure:</b> %{x} months<br><b>Monthly Charges:</b> $%{y:.2f}<br><b>Gender:</b> %{customdata[0]}<br><b>Contract:</b> %{customdata[1]}<extra></extra>"
    )
    return fig

def plot_churn_distribution(df):
    churn_counts = df['churn'].value_counts().reset_index()
    churn_counts.columns = ['churn', 'count']
    fig = px.pie(
        churn_counts,
        names='churn',
        values='count',
        title="Churn Distribution",
        hole=0.4,
        template="plotly_white"
    )
    fig.update_traces(textinfo='percent+label')
    return fig

def plot_tenure_boxplot(df):
    fig = px.box(
        df,
        x='churn',
        y='tenure',
        title="Tenure Distribution by Churn",
        template="plotly_white"
    )
    fig.update_layout(
        xaxis_title="Churn",
        yaxis_title="Tenure (Months)",
        title_font=dict(size=18, family="Segoe UI", color="#111111")
    )
    return fig

def plot_payment_method_breakdown(df):
    pm_data = df.groupby(['payment_method', 'churn']).size().reset_index(name='count')
    fig = px.bar(
        pm_data,
        x='payment_method',
        y='count',
        color='churn',
        barmode='group',
        title="Payment Method Breakdown by Churn",
        template="plotly_white"
    )
    fig.update_layout(xaxis_title="Payment Method", yaxis_title="Count")
    return fig

def plot_contract_churn(df):
    contract_data = df.groupby(['contract', 'churn']).size().reset_index(name='count')
    fig = px.bar(
        contract_data,
        x='contract',
        y='count',
        color='churn',
        barmode='group',
        title="Churn by Contract Type",
        template="plotly_white"
    )
    fig.update_layout(xaxis_title="Contract Type", yaxis_title="Count")
    return fig

def plot_sunburst_chart(df):
    fig = px.sunburst(
        df,
        path=['contract', 'payment_method', 'churn'],
        title="Customer Segmentation by Contract, Payment Method, and Churn",
        template="plotly_white"
    )
    return fig

def plot_internet_service_donut(df):
    internet_data = df.groupby(['internet_service', 'churn']).size().reset_index(name='count')
    fig = px.pie(
        internet_data,
        names='internet_service',
        values='count',
        title="Internet Service Usage",
        hole=0.5,
        template="plotly_white"
    )
    return fig

def expander_explanation(plot_name):
    explanations = {
        "churn_distribution": "This donut chart displays the overall proportion of customers who have churned versus those who have not. It provides a quick visual summary of the churn status.",
        "monthly_charges": "This histogram, with a marginal box plot, illustrates the distribution of monthly charges among customers. The box shows medians, quartiles, and potential outliers.",
        "tenure_vs_charges": "This scatter plot depicts the relationship between customer tenure and monthly charges. The OLS trendline reveals the overall trend and highlights clusters or outliers.",
        "tenure_boxplot": "This box plot compares tenure distributions for churned vs. non-churned customers, highlighting differences in customer loyalty.",
        "payment_method": "This bar chart shows how payment methods correlate with churn, indicating which methods may be linked to higher churn rates.",
        "contract_churn": "This visualization breaks down churn by contract type, showing that month-to-month contracts often have higher churn rates.",
        "sunburst": "The sunburst chart provides a multi-level view of customer segmentation by contract type, payment method, and churn status.",
        "internet_service": "This donut chart represents the usage of different internet service types and their relation to churn."
    }
    return explanations.get(plot_name, "No explanation available.")

def plot_null_values_heatmap(df):
    if df.isnull().sum().sum() == 0:
        return None
    missing = df.isnull().astype(int)
    fig = px.imshow(missing, color_continuous_scale='Blues', title="Missing Values Heatmap", template="plotly_white")
    return fig

def plot_senior_vs_churn(df):
    df['SeniorCitizenCat'] = df['senior_citizen'].apply(lambda x: 'Yes' if x == 1 else 'No')
    senior_counts = df.groupby(['SeniorCitizenCat', 'churn']).size().reset_index(name='count')
    fig = px.bar(senior_counts, x='SeniorCitizenCat', y='count', color='churn',
                 title="Senior Citizen vs. Churn", barmode='group', template="plotly_white")
    fig.update_layout(xaxis_title="Senior Citizen", yaxis_title="Count")
    return fig

def plot_partner_dependents_vs_churn(df):
    df['PartnerDependents'] = df['partner'] + " & " + df['dependents']
    pd_counts = df.groupby(['PartnerDependents', 'churn']).size().reset_index(name='count')
    fig = px.bar(pd_counts, x='PartnerDependents', y='count', color='churn',
                 title="Partner & Dependents vs. Churn", barmode='group', template="plotly_white")
    fig.update_layout(xaxis_title="Partner & Dependents", yaxis_title="Count")
    return fig

def plot_gender_vs_churn(df):
    gender_counts = df.groupby(['gender', 'churn']).size().reset_index(name='count')
    fig = px.bar(gender_counts, x='gender', y='count', color='churn',
                 title="Gender vs. Churn", barmode='group', template="plotly_white")
    fig.update_layout(xaxis_title="Gender", yaxis_title="Count")
    return fig

def plot_service_usage_vs_churn(df):
    # Updated service columns to match the renamed lowercase columns.
    service_cols = [
        "multiple_lines", "online_security", "online_backup",
        "device_protection", "tech_support", "streaming_tv", "streaming_movies"
    ]
    # Ensure only columns present in the DataFrame are used.
    available_cols = [col for col in service_cols if col in df.columns]
    if not available_cols:
        return None

    melted = df.melt(id_vars=['customer_id', 'churn'], value_vars=available_cols, 
                     var_name="Service", value_name="Usage")
    # Convert values to binary: assume 'Yes' means 1 and anything else is 0.
    melted['Usage'] = melted['Usage'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    usage = melted.groupby(['Service', 'churn'])['Usage'].sum().reset_index()
    fig = px.bar(
        usage,
        x='Service',
        y='Usage',
        color='churn',
        barmode='group',
        title="Service Usage vs. Churn",
        template="plotly_white"
    )
    fig.update_layout(xaxis_title="Service", yaxis_title="Count of 'Yes'")
    return fig


def plot_paperlessbilling_vs_churn(df):
    if 'paperless_billing' in df.columns:
        pb_counts = df.groupby(['paperless_billing', 'churn']).size().reset_index(name='count')
        fig = px.bar(pb_counts, x='paperless_billing', y='count', color='churn',
                     title="Paperless Billing vs. Churn", barmode='group', template="plotly_white")
        fig.update_layout(xaxis_title="Paperless Billing", yaxis_title="Count")
        return fig
    else:
        return None

def plot_totalcharges_boxplot(df):
    if 'total_charges' in df.columns:
        fig = px.box(df, x='churn', y='total_charges', title="Total Charges by Churn", template="plotly_white")
        fig.update_layout(xaxis_title="Churn", yaxis_title="Total Charges ($)")
        return fig
    else:
        return None

def plot_scatter_matrix(df):
    # Choose numeric columns to include. Adjust if needed.
    numeric_columns = ['tenure', 'monthly_charges']
    if 'total_charges' in df.columns:
        numeric_columns.append('total_charges')
    
    fig = px.scatter_matrix(
        df,
        dimensions=numeric_columns,
        color='churn',
        template="plotly_white",
        title="Scatter Matrix for Numerical Features",
        height=800,
        width=800,
        hover_data=df.columns
    )
    # Hide diagonal to avoid overlapping markers and reduce clutter.
    fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.7))
    
    # Adjust layout for improved spacing.
    fig.update_layout(
        title_font=dict(size=18, family="Segoe UI", color="#111111"),
        margin=dict(l=50, r=50, b=50, t=50),
        plot_bgcolor="white"
    )
    return fig

def plot_kmeans_clusters(df):
    numeric_columns = ['tenure', 'monthly_charges']
    if 'total_charges' in df.columns:
        numeric_columns.append('total_charges')
    data_numeric = df[numeric_columns].dropna()
    if data_numeric.empty:
        return None
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_numeric)
    data_numeric['Cluster'] = clusters.astype(str)
    data_numeric['churn'] = df.loc[data_numeric.index, 'churn']
    fig = px.scatter(
        data_numeric,
        x='tenure',
        y='monthly_charges',
        color='Cluster',
        symbol='churn',
        title="KMeans Clusters of Customers",
        template="plotly_white"
    )
    fig.update_layout(xaxis_title="Tenure (Months)", yaxis_title="Monthly Charges ($)")
    return fig

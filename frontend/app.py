import streamlit as st
import pandas as pd
import plotly.express as px
from faker import Faker
import random
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data():
    # Replace this with your CSV path or data loading logic
    data = pd.read_csv("frontend/fake_stock_data.csv")  # or use pd.read_excel if needed
    return data

# Function to create fake data
def create_fake_data(n=100):
    fake = Faker()
    data = []
    for _ in range(n):
        data.append({
            'Name': fake.name(),
            'Email': fake.email(),
            'Phone': fake.phone_number(),
            'Country': fake.country(),
            'Status': random.choice(['New', 'Processing', 'Success', 'Failed']),
            'Revenue ($)': round(random.uniform(100, 10000), 2)
        })
    return pd.DataFrame(data)

def show_dashboard():
    st.title("📊 CRM Dashboard - Customer Management")
    df = create_fake_data(150)

    # Overview metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("🔢 Total Customers", len(df))
    col2.metric("💰 Total Revenue", f"${df['Revenue ($)'].sum():,.2f}")
    col3.metric("📈 Successful Customers", df[df['Status'] == 'Success'].shape[0])

    st.markdown("---")

    # Status distribution chart
    status_counts = df['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig_status = px.pie(status_counts, names='Status', values='Count',
                        title='Customer Status Distribution', hole=0.4)
    st.plotly_chart(fig_status, use_container_width=True)

    # Top countries chart
    country_counts = df['Country'].value_counts().nlargest(10).reset_index()
    country_counts.columns = ['Country', 'Count']
    fig_country = px.bar(country_counts, x='Country', y='Count',
                         title='Top 10 Countries with Most Customers',
                         labels={'Country': 'Country', 'Count': 'Count'})
    st.plotly_chart(fig_country, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Customer List")
    st.dataframe(df)

def correlation_heatmap():
    st.title("Correlation Analysis: Features vs Close Price")
    df = load_data()

    with st.sidebar:
        # Sidebar or selectbox to choose Ticker Symbol (Mode)
        ticker_options = df['Ticker_symbol'].unique()
        choose_ticker = st.selectbox("Choose Bank:",ticker_options)
        # Filter the DataFrame based on the selected mode
    filtered_df = df[df['Ticker_symbol'] == choose_ticker]
    # Only numerical features
    numerics = filtered_df.select_dtypes(include='number')
    correlation = numerics.corr()['close'].drop('close')
    st.title("Correlation with Close Price (Plotly)")

    # Plot bar chart
    fig = px.bar(
        correlation.sort_values(ascending=True),
        orientation='h',
        labels={'value': 'Correlation'},
        title='Feature Correlation with Close Price'
    )
    st.plotly_chart(fig)

    # Optional: correlation heatmap
    st.subheader("Full Correlation Heatmap")
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=numerics.corr().values,
            x=numerics.columns,
            y=numerics.columns,
            colorscale='Viridis'
        )
    )
    heatmap_fig.update_layout(height=600)
    st.plotly_chart(heatmap_fig)


# Main interface
def main():
    st.set_page_config(page_title="CRM Dashboard", layout="wide")
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Dashboard"

    with st.sidebar:
        st.header("🔧 Menu")
        if st.button("🏠 Dashboard"):
            st.session_state.selected_page = "Dashboard"
        if st.button("📁 Prediction"):
            st.session_state.selected_page = "Prediction"
        if st.button("💬 Support"):
            st.session_state.selected_page = "Support"
        if st.button("💬 Factor"):
            st.session_state.selected_page = "Factor"

    page = st.session_state.selected_page

    if page == "Dashboard":
        show_dashboard()
        correlation_heatmap()
    elif page == "Prediction":
        from pages import price_inference
        st.title("📁 Model Prediction")
        price_inference.inference()
    elif page == "Support":
        st.title("💬 Support")
        st.write("Need help? Email us at `support@example.com`.")
    elif page == "Factor":
        st.title("💬 Correlation map")
        correlation_heatmap()

if __name__ == "__main__":
    main()

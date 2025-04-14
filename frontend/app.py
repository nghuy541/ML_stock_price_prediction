import streamlit as st
import pandas as pd
import plotly.express as px
from faker import Faker
import random
import plotly.express as px
import plotly.graph_objects as go
# import google.generativeai as genai
from google import genai


@st.cache_data
def load_data():
    # Replace this with your CSV path or data loading logic
    data = pd.read_csv("frontend/full_data.csv")  # or use pd.read_excel if needed
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

def ai_investment_analysis(ticker, correlation, model_type, api_key):
    prompt = f"""
    Tôi là một nhà đầu tư đang phân tích cổ phiếu của ngân hàng {ticker}.
    Bạn có thể phân tích mối tương quan giữa các chỉ số tài chính và giá khi đóng phiên dưới đây
    {correlation.to_string()}
    Và kết luận ngắn gọn các yêu tố sau
    1. Hiệu quả hoạt động và sử dụng vốn
    2. Định giá và kỳ vọng thị trường
    3. Chất lượng tài chính và tiềm năng phát triển
    
    """
    # genai.configure(api_key=api_key)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text

def correlation_heatmap():
    st.title("📊 Phân tích tương quan các chỉ số tài chính với giá đóng phiên")
    df = load_data()
        
    with st.expander("Cài đặt"):
        # Sidebar or selectbox to choose Ticker Symbol (Mode)
        ticker_options = df['Ticker_symbol'].unique()
        choose_ticker = st.selectbox("Chọn tên ngân hàng:", ticker_options)
        choose_model = st.selectbox("Lựa chọn mô hình", ['gemini','llama3'])
        api_key = st.text_input("Nhập api key của mô hình:",type='password')
        analyzed_button = st.button("Phân tích")

    filtered_df = df[df['Ticker_symbol'] == choose_ticker]
    numerics = filtered_df.select_dtypes(include='number')
    correlation = numerics.corr()

    # Biểu đồ cột (correlation với close)
    st.subheader(f"🔍 Tương quan từng chỉ số với giá đóng phiên của ngân hàng {choose_ticker}")
    correlation_with_close = correlation['close'].drop('close').sort_values()
    fig_bar = px.bar(
        correlation_with_close,
        orientation='h',
        labels={'index': 'Chỉ số tài chính', 'value': 'Hệ số tương quan'},
        color=correlation_with_close.values,
        color_continuous_scale='Tealgrn',
        title='Hệ số tương quan giữa các chỉ số tài chính và giá đóng phiên'
    )
    fig_bar.update_layout(
        height=500,
        xaxis_title="Hệ số tương quan",
        yaxis_title="",
        title_x=0.5
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Biểu đồ nhiệt
    with st.expander("🌡️ Biểu đồ nhiệt toàn bộ tương quan"):
        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.columns,
                colorscale='YlGnBu',
                zmin=-1,
                zmax=1,
                colorbar=dict(title='Hệ số tương quan'),
                hoverongaps=False
            )
        )
        heatmap_fig.update_layout(
            height=700,
            title='Biểu đồ nhiệt thể hiện mối tương quan giữa các chỉ số tài chính',
            title_x=0.5,
            xaxis=dict(tickangle=-45),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
    # Phân tích AI
    if analyzed_button:
        st.subheader("💡 Phân tích đầu tư từ AI")
        with st.spinner("🤖 Đang phân tích với mô hình AI..."):
            ai_result = ai_investment_analysis(choose_ticker, correlation_with_close, choose_model, api_key)
            st.success("✅ Phân tích hoàn tất!")
            st.markdown(ai_result)
    else:
        st.warning("🔐 Vui lòng nhập API Key để kích hoạt phân tích AI.")

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
        if st.button("💬 Các yếu tố ảnh hưởng đến giá"):
            st.session_state.selected_page = "Factor"
    page = st.session_state.selected_page
    if page == "Dashboard":
        show_dashboard()
        correlation_heatmap()
    elif page == "Prediction":
        from pages import price_inference
        st.title("📁 Dự đoán xu hướng")
        price_inference.inference()
    elif page == "Hỗ trợ":
        st.title("💬 Support")
        st.write("Need help? Email us at `support@example.com`.")
    elif page == "Factor":
        correlation_heatmap()

if __name__ == "__main__":
    main()

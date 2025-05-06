import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import load_model
import plotly.graph_objects as go
from datetime import timedelta
import plotly.express as px
import numpy as np
import time
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from google import genai

def visualize_top5_chart(top5_df,metric):
    # Vẽ biểu đồ cột
    fig = px.bar(
        top5_df,
        x="symbol",
        y=metric,
        title=f"📊 Top 5 stocks by {metric}",
        text=metric,
        color="symbol",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(yaxis_title="Growth (%)", xaxis_title="Stock symbol", uniformtext_minsize=8, uniformtext_mode='hide')

    # Hiển thị trong Streamlit
    st.plotly_chart(fig, use_container_width=True)

def ai_investment_analysis(df, api_key):
    prompt = f"""
    Tôi là một người mới bắt đầu tìm hiểu cổ phiếu.
    Bạn là chuyên gia có 10 năm kinh nghiệm trong lĩnh vực đầu tư cổ phiếu. 
    Tôi sẽ cung cấp cho bạn top 5 cổ phiếu có tiềm năng tăng trưởng để bạn phân tích sơ bộ
    {df.to_string()}
    Yêu cầu:
    + Từ phân tích của bạn, nếu thiếu thông tin đừng đưa ra kết luận vội, bạn hãy kêu người dùng bổ sung thêm tài liệu
    (ví dụ tin tức, báo cáo tài chính,)
    + Phân tích ngắn gọn
    """
    # genai.configure(api_key=api_key)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text

def plot_rank_stock(df):
    # Plot using Plotly
    fig = go.Figure()
    # Add each column as a trace
    for period in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[period],
            name=period
        ))
    # Update layout
    fig.update_layout(
        title="Percentage growth by period",
        xaxis_title="Symbol",
        yaxis_title="Growth(%)",
        barmode="group"
    )
    # Show in Streamlit
    st.plotly_chart(fig)

import plotly.graph_objects as go

def rank_stock(df):
    with st.expander(f"🏆 Top stocks with high growth percentage"):
        plot_rank_stock(df)
        # Cho người dùng chọn số lượng top cổ phiếu
        top_n = st.selectbox("🔢 How many top stocks to display?", [3, 5, 10, 15, 20], index=1)
        # Select metric to rank
        metric = st.selectbox("📅 Select rating time:", ["7 days", "1 month", "1 quarter"])
        st.subheader(f"Ranking in {metric} (Top {top_n})")

        # Sort and get top N
        topN_df = df.sort_values(by=metric, ascending=False).head(top_n).reset_index()
        topN_df.index += 1  # Start ranking from 1

        api_key = 'AIzaSyDKqLGLKVWtgqEC0AsNhjnFWQ6CoL8kvHs'
        display_df = topN_df[["symbol", metric]]

        # Hiển thị biểu đồ và bảng
        visualize_top5_chart(display_df, metric)  # bạn có thể rename hàm thành `visualize_topN_chart`
        st.write("Do you want?")

        help_button = st.button("🧠 Investing with AI", help="Click to get suggestions and automatic comments from AI.")
        # chart_button = st.button("📊 View stock ranking chart ", help="Click to view analysis chart")
        # if chart_button:
        #     st.success("✅ Completed analysis chart")
        #     plot_rank_stock(df)

        if help_button:
            with st.spinner("🤖 Analyzing..."):
                answer = ai_investment_analysis(display_df, api_key)
            st.success("✅ AI has finished analyzing:")
            st.write(answer)
            investment_goal = st.selectbox(
                "🎯 Choose your investment goals:",
                [
                    "⚡ Surfing (Short Term)",
                    "📆 Medium term (growth in a few months)",
                    "🛡️ Long term (stable, low risk)",
                    "🚀 Find stocks with strong growth",
                    "💰 Prefer regular dividends"
                ],
                help="Choose to help AI make suggestions that fit your strategy."
            )

       

def run():
    st.title("Ranking of top growth stock")
    file_path = "frontend/stock_price_with_growth.csv"
    df = pd.read_csv("frontend/stock_price.csv")
    df = convert_time(df)
    symbols = df['symbol'].unique()
    model = load_model("frontend/models/lstm_model_ver1.keras")

    def thread_worker(symbol):
        return process_symbol(symbol, df, model)

    # with st.expander(f"🏆Stock growth trend analysis"):
    #     choose_stock = st.selectbox("Choose stocks:",symbols)
    #     analyze_trend = st.button("Trend analysis")
    #     if analyze_trend:
    #         process_single_symbol(choose_stock, df, model)


    if os.path.exists(file_path):
        print(f"The file '{file_path}' exists.")
        df = pd.read_csv("frontend/stock_price_with_growth.csv")
        # Remove percentage signs and convert columns to numeric
        for col in ["7 days", "1 month", "1 quarter"]:
            df[col] = df[col].str.replace('%', '').astype(float)
        # Group by symbol and calculate the mean for the 3 columns
        mean_df = df.groupby("symbol")[["7 days", "1 month", "1 quarter"]].mean().round(2)
        rank_stock(mean_df)

def convert_time(data):
    data['time'] = pd.to_datetime(data['time'])
    return data.sort_values(by=['symbol', 'time'])

def process_single_symbol(symbol, df, model):
    df_symbol = df[df['symbol'] == symbol].copy()
    df_symbol = add_features(df_symbol)

    label_encoder = LabelEncoder()
    df_symbol['index_encoded'] = label_encoder.fit_transform(df_symbol['symbol'])

    features = ['open', 'high', 'low', 'volume', 'index_encoded',
                'lag_1', 'lag_2', 'rolling_mean_3', 'rolling_std_3', 'momentum_1']
    target = 'close'

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(df_symbol[features])
    y_scaled = scaler_y.fit_transform(df_symbol[[target]])
    preds = forecast_next_days(model, X_scaled, scaler_x, scaler_y, n_days=252)
    # preds = batch_forecast_next_days(model, X_scaled, scaler_x, scaler_y, n_days=252)
    last_close = df_symbol['close'].iloc[-1]
    last_date = df_symbol['time'].iloc[-1]
    growth_data = growth_analysis(last_close, preds)

def add_features(data):
    data['lag_1'] = data['close'].shift(1)
    data['lag_2'] = data['close'].shift(2)
    data['rolling_mean_3'] = data['close'].rolling(window=3).mean()
    data['rolling_std_3'] = data['close'].rolling(window=3).std()
    data['momentum_1'] = data['close'].diff()
    data.fillna(method='bfill', inplace=True)
    data.fillna(0, inplace=True)
    return data    

def growth_analysis(start_price, preds):
    growth_data = []
    days = {
        "7 days": 7,
        "1 month": 21,
        "1 quarter": 63,
        "1 year": 252
    }
    st.header("🔍 Forecast growth analysis")
    for label, day in days.items():
        if day < len(preds):
            future_price = preds[day]
            growth_pct = (future_price - start_price) / start_price * 100
            growth_data.append({'label': label, 'growth_pct': f"{growth_pct:.2f}%"})
            st.metric(label=f"Growth later {label}", value=f"{growth_pct:.2f}%")
    return growth_data

def forecast_next_days(model, data, scaler_x, scaler_y, n_days):
    last_sequence = data[-60:]  # (60, num_features)
    num_features = last_sequence.shape[1]
    predictions = []

    for _ in range(n_days):
        input_seq = np.expand_dims(last_sequence, axis=0)  # (1, 60, num_features)
        pred_scaled = model.predict(input_seq)  # (1, 1)
        pred_scaled = pred_scaled.reshape(-1, 1)

        pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred)

        # Cập nhật: tạo 1 step mới lặp lại pred_scaled thành 1 dòng có num_features
        new_step = np.tile(pred_scaled[0], (1, num_features))  # (1, num_features)
        last_sequence = np.append(last_sequence[1:], new_step, axis=0)
    return predictions

def process_symbol(symbol, df, model):
    df_symbol = df[df['symbol'] == symbol].copy()
    df_symbol = add_features(df_symbol)

    label_encoder = LabelEncoder()
    df_symbol['index_encoded'] = label_encoder.fit_transform(df_symbol['symbol'])

    features = ['open', 'high', 'low', 'volume', 'index_encoded',
                'lag_1', 'lag_2', 'rolling_mean_3', 'rolling_std_3', 'momentum_1']
    target = 'close'

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(df_symbol[features])
    y_scaled = scaler_y.fit_transform(df_symbol[[target]])
    preds = forecast_next_days(model, X_scaled, scaler_x, scaler_y, n_days=252)
    # preds = batch_forecast_next_days(model, X_scaled, scaler_x, scaler_y, n_days=252)
    last_close = df_symbol['close'].iloc[-1]
    last_date = df_symbol['time'].iloc[-1]
    growth_data = growth_analysis(last_close, preds)

    # Add growth data as new columns to the DataFrame
    for data in growth_data:
        df_symbol[data['label']] = data['growth_pct']
    
    return df_symbol
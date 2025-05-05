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

# Load model và scaler 1 lần ngoài vòng lặp
model = load_model("frontend/models/lstm_model_ver1.keras")

# ========== Helper ==========
def convert_time(data):
    data['time'] = pd.to_datetime(data['time'])
    return data.sort_values(by=['symbol', 'time'])

def add_features(data):
    data['lag_1'] = data['close'].shift(1)
    data['lag_2'] = data['close'].shift(2)
    data['rolling_mean_3'] = data['close'].rolling(window=3).mean()
    data['rolling_std_3'] = data['close'].rolling(window=3).std()
    data['momentum_1'] = data['close'].diff()
    data.fillna(method='bfill', inplace=True)
    data.fillna(0, inplace=True)
    return data

def create_sequences(data, time_steps=60):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
    return np.array(X)


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

def batch_forecast_next_days(model, X_batch, scalers_y, n_days):
    """
    Dự đoán batch cho nhiều chuỗi (symbols).
    Args:
        model: mô hình LSTM đã load
        X_batch: np.array shape (batch_size, 60, num_features)
        scalers_y: list các scaler_y tương ứng với từng symbol
        n_days: số ngày cần dự đoán

    Returns:
        all_preds: list gồm batch_size phần tử, mỗi phần tử là list 252 giá dự đoán
    """
    batch_size, time_steps, num_features = X_batch.shape
    all_preds = [[] for _ in range(batch_size)]

    for _ in range(n_days):
        pred_scaled = model.predict(X_batch, verbose=0)  # shape (batch_size, 1)

        for i in range(batch_size):
            pred_inv = scalers_y[i].inverse_transform(pred_scaled[i].reshape(1, -1))[0][0]
            all_preds[i].append(pred_inv)

            new_step = np.tile(pred_scaled[i], (1, num_features))  # shape (1, num_features)
            X_batch[i] = np.append(X_batch[i][1:], new_step, axis=0)
    return all_preds

def plot_predictions(preds, last_date):
    future_dates = [last_date + timedelta(days=i+1) for i in range(len(preds))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines+markers', name="Giá dự đoán"))
    fig.update_layout(title="Biểu đồ giá cổ phiếu dự đoán (1 năm tới)",xaxis_title="Thời gian", yaxis_title="Giá cổ phiếu")
    return fig

def visualize_top5_chart(top5_df,metric):
    # Vẽ biểu đồ cột
    fig = px.bar(
        top5_df,
        x="symbol",
        y=metric,
        title=f"📊 Top 5 cổ phiếu theo {metric}",
        text=metric,
        color="symbol",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(yaxis_title="Tăng trưởng (%)", xaxis_title="Mã cổ phiếu", uniformtext_minsize=8, uniformtext_mode='hide')

    # Hiển thị trong Streamlit
    st.plotly_chart(fig, use_container_width=True)

def rank_stock(df):
    with st.expander(f"🏆 Top 5 cổ phiếu có phần trăm tăng trưởng cao"):
        # Select metric to rank
        metric = st.selectbox("Lựa chọn thời gian xếp hạng:", ["7 ngày", "1 tháng", "1 quý"])
        st.subheader(f"Xếp hạng trong thời gian {metric}")
        # Sort and get top 5
        top5 = df.sort_values(by=metric, ascending=False).head(5).reset_index()
        api_key = 'AIzaSyDKqLGLKVWtgqEC0AsNhjnFWQ6CoL8kvHs'
        top5.index += 1  # Start ranking from 1
        # Display table
        top5_df = top5[["symbol", metric]]
        # st.table(top5_df.style.format({metric: "{:.2f}%"}))
        visualize_top5_chart(top5_df,metric)
        st.write("Bạn có muốn?")
        help_button = st.button("🧠 Nhờ AI trợ giúp đầu tư", help="Click để nhận gợi ý và nhận định tự động từ AI.")
        chart_button = st.button("📊 Xem biểu đồ xếp hạng cổ phiếu ", help="Click để xem biểu đồ phân tích")
        if help_button:
            with st.spinner("🤖 Đang phân tích..."):
                answer = ai_investment_analysis(top5_df, api_key)
            st.success("✅ AI đã phân tích xong:")
            st.write(answer)
            investment_goal = st.selectbox(
                    "🎯 Chọn mục tiêu đầu tư của bạn:",
                    [
                        "⚡ Lướt sóng (Ngắn hạn)",
                        "📆 Trung hạn (tăng trưởng vài tháng)",
                        "🛡️ Dài hạn (ổn định, ít rủi ro)",
                        "🚀 Tìm cổ phiếu tăng mạnh",
                        "💰 Ưu tiên cổ tức đều đặn"
                    ],
                    help="Lựa chọn giúp AI đưa ra gợi ý phù hợp với chiến lược của bạn."
                )
        if chart_button:
            st.success("✅ Biểu đồ phân tích đã hoàn thành")
            plot_rank_stock(df)


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

def growth_analysis(start_price, preds):
    growth_data = []
    days = {
        "7 ngày": 7,
        "1 tháng": 21,
        "1 quý": 63,
        "1 năm": 252
    }
    st.header("🔍 Phân tích tăng trưởng dự đoán")
    for label, day in days.items():
        if day < len(preds):
            future_price = preds[day]
            growth_pct = (future_price - start_price) / start_price * 100
            growth_data.append({'label': label, 'growth_pct': f"{growth_pct:.2f}%"})
            st.metric(label=f"Tăng trưởng sau {label}", value=f"{growth_pct:.2f}%")
    return growth_data

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
    # for result in results:
    st.plotly_chart(plot_predictions(preds, last_date), use_container_width=True)
            # growth_analysis(result['last_close'], result['preds'])

    # Add growth data as new columns to the DataFrame
    # for data in growth_data:
    #     df_symbol[data['label']] = data['growth_pct']
    
    # return df_symbol

    # return {
    #     'symbol': symbol,
    #     'preds': preds,
    #     'last_close': last_close,
    #     'last_date': last_date
    # }
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
        title="Trung bình phần trăm tăng trưởng theo mã cổ phiếu",
        xaxis_title="Mã cổ phiếu",
        yaxis_title="Tăng trưởng(%)",
        barmode="group"
    )
    # Show in Streamlit
    st.plotly_chart(fig)

def wrapper(symbol_df_tuple):
    symbol, df = symbol_df_tuple
    return process_symbol(symbol, df)


def run_bk():
    st.title("Tổng quan phần trăm tăng trưởng của từng mã cổ phiếu")
    file_path = "frontend/stock_price_with_growth.csv"
    df = pd.read_csv("frontend/stock_price.csv")
    df = convert_time(df)
    symbols = df['symbol'].unique()
    model = load_model("frontend/models/lstm_model_ver1.keras")

    def thread_worker(symbol):
        return process_symbol(symbol, df, model)

    # choose_stock = st.selectbox("Chọn cổ phiếu:",symbols)

    if os.path.exists(file_path):
        print(f"The file '{file_path}' exists.")
        df = pd.read_csv("frontend/stock_price_with_growth.csv")
        # Remove percentage signs and convert columns to numeric
        for col in ["7 ngày", "1 tháng", "1 quý"]:
            df[col] = df[col].str.replace('%', '').astype(float)
        # Group by symbol and calculate the mean for the 3 columns
        mean_df = df.groupby("symbol")[["7 ngày", "1 tháng", "1 quý"]].mean().round(2)
        rank_stock(mean_df)
    else:
        # start_time = time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(thread_worker, symbols))
        # Combine the results back into a single DataFrame
        updated_df = pd.concat(results)
        # Save the updated DataFrame to a new CSV file
        updated_df.to_csv("frontend/stock_price_with_growth.csv", index=False)
    
def run():
    st.title("Tổng quan phần trăm tăng trưởng của từng mã cổ phiếu")
    file_path = "frontend/stock_price_with_growth.csv"
    df = pd.read_csv("frontend/stock_price.csv")
    df = convert_time(df)
    symbols = df['symbol'].unique()
    model = load_model("frontend/models/lstm_model_ver1.keras")

    def thread_worker(symbol):
        return process_symbol(symbol, df, model)

    with st.expander(f"🏆Phân tích xu hướng tăng trưởng của cổ phiếu"):
        choose_stock = st.selectbox("Chọn cổ phiếu:",symbols)
        analyze_trend = st.button("Phân tích xu hướng")
        if analyze_trend:
            process_single_symbol(choose_stock, df, model)

    if os.path.exists(file_path):
        print(f"The file '{file_path}' exists.")
        df = pd.read_csv("frontend/stock_price_with_growth.csv")
        # Remove percentage signs and convert columns to numeric
        for col in ["7 ngày", "1 tháng", "1 quý"]:
            df[col] = df[col].str.replace('%', '').astype(float)
        # Group by symbol and calculate the mean for the 3 columns
        mean_df = df.groupby("symbol")[["7 ngày", "1 tháng", "1 quý"]].mean().round(2)
        rank_stock(mean_df)
    else:
        # start_time = time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(thread_worker, symbols))
        # Combine the results back into a single DataFrame
        updated_df = pd.concat(results)
        # Save the updated DataFrame to a new CSV file
        updated_df.to_csv("frontend/stock_price_with_growth.csv", index=False)
    # for result in results:
    #     st.subheader(f"📈 Dự đoán cho mã {result['symbol']}")
    #     if result['symbol'] == 'ACB':
    #         st.plotly_chart(plot_predictions(result['preds'], result['last_date'])) 
        # growth_analysis(result['last_close'], result['preds'])
    

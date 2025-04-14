from keras.models import Sequential
from keras.layers import Dense, LSTM
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder,LabelEncoder
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# Sequence creator
def create_sequences(X, y, time_steps=3):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

def convert_time(data):
    data = data.rename(columns={'index': 'symbol'})
    # Chuyển cột 'date' sang định dạng datetime
    data['time'] = pd.to_datetime(data['time'], format='%m/%d/%Y %H:%M')
    # Sort to ensure time/ticker order
    data = data.sort_values(by=['symbol','time'])
    return data

def label_encoder(data):
    # Label Encoding
    le = LabelEncoder()
    data['index_encoded'] = le.fit_transform(data['symbol'])
    return data

def add_features(data):
    data['lag_1'] = data['close'].shift(1)  # Close hôm trước
    data['lag_2'] = data['close'].shift(2)  # Close 2 hôm trước
    # Fill lag features with next valid value (backfill)
    data['lag_1'] = data['lag_1'].fillna(method='bfill')
    data['lag_2'] = data['lag_2'].fillna(method='bfill')
    # Giá trị trung bình trong một cửa sổ thời gian nhất định (5 ngày, 10 ngày,...)
    data['rolling_mean_3'] = data['close'].rolling(window=3).mean()
    data['rolling_std_3'] = data['close'].rolling(window=3).std()
    # Fill rolling mean using expanding mean (as a smoother trend)
    data['rolling_mean_3'] = data['rolling_mean_3'].fillna(data['close'].expanding().mean())
    # Fill rolling std with 0 (no variation known at start)
    data['rolling_std_3'] = data['rolling_std_3'].fillna(0)
    # Chênh lệch giá hôm nay so với hôm trước (tốc độ thay đổi)
    data['momentum_1'] = data['close'] - data['close'].shift(1)
    # Fill momentum and return with 0 (neutral change)
    data['momentum_1'] = data['momentum_1'].fillna(0)
    return data

def predict(scaler_y,model,input_data):
    # Make prediction
    predictions = model.predict(input_data)[:, 0, 0].reshape(-1, 1) 
    y_pred_actual = scaler_y.inverse_transform(predictions)
    return y_pred_actual

def visualize_chart(test_df,actual_prices,predicted_prices):
    # Flatten y_test and predictions (y_pred_actual)
    actual_prices = actual_prices.flatten()
    predicted_prices = predicted_prices.flatten()
    # Create the figure
    fig = go.Figure()
    # Add actual price line
    fig.add_trace(go.Scatter(
        y=actual_prices,
        mode='lines',
        name='Giá thực tế',
        line=dict(color='blue')
    ))
    # Add predicted price line
    fig.add_trace(go.Scatter(
        y=predicted_prices,
        mode='lines',
        name='Giá dự đoán',
        line=dict(color='orange')
    ))
    # Customize layout
    fig.update_layout(
        title='Biểu đồ kết quả dự đoán giá của mô hình và giá thực tế',
        xaxis_title='Thời gian',
        yaxis_title='Giá',
        legend=dict(x=0, y=1),
        template='plotly_white',
        height=500,
        width=900)
    # Optional: Add grid
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig
def price_growth_analysis(current_price,predicted_price):
    st.title("📈 Kết luận")
    # Lấy phần tử cuối cùng
    current_price = float(current_price[-1])
    predicted_price = float(predicted_price[-1])
    if current_price > 0 and predicted_price > 0:
        growth_percent = ((predicted_price - current_price) / current_price) * 100
        st.metric(
            label="🔢 Tỷ lệ tăng trưởng dự đoán",
            value=f"{growth_percent:.2f}%",
            delta=f"{predicted_price - current_price:,.0f} VNĐ"
        )
        if growth_percent > 0:
            st.success(f"✅ Giá dự đoán cao hơn giá hiện tại khoảng {growth_percent:.2f}%.")
            st.info("💡 Nhà đầu tư có thể kỳ vọng lợi nhuận ngắn hạn nếu mua vào sớm.")
        elif growth_percent < 0:
            st.warning(f"⚠️ Giá dự đoán thấp hơn giá hiện tại khoảng {abs(growth_percent):.2f}%.")
            st.info("💡 Cân nhắc rủi ro trước khi đầu tư.")
        else:
            st.write("🔍 Giá dự đoán bằng với giá hiện tại. Không có thay đổi.")
def growth_percent_cal(df):
    # ✅ 1. Tăng trưởng phần trăm theo ngày (daily growth rate)
    df['daily_growth_pct_GT'] = df['actual_price'].pct_change() * 100
    df['daily_growth_pct_pred'] = df['predicted_price'].pct_change() * 100

    # ✅ 2. Tăng trưởng lũy kế (cumulative growth) từ thời điểm đầu
    df['cumulative_growth_GT'] = ((df['actual_price'] / df['actual_price'].iloc[0]) - 1) * 100
    df['cumulative_growth_pred'] = ((df['predicted_price'] / df['predicted_price'].iloc[0]) - 1) * 100

    # ✅ 3. Tăng trưởng trong 1 khoảng thời gian (ví dụ: 7 ngày)
    df['growth_7d_GT'] = df['actual_price'].pct_change(periods=7) * 100
    df['growth_7d_pred'] = df['predicted_price'].pct_change(periods=7) * 100
    return df


def inference():
    st.title("Dự đoán tăng trưởng của cổ phiếu")
    features = ['open', 'high', 'low', 'volume', 'index_encoded','lag_1','lag_2',
    'rolling_mean_3','rolling_std_3','momentum_1']
    target_col = 'close'
    time_steps = 1
    test_df1 = pd.read_csv("frontend/stock_price.csv")
    test_df1 = convert_time(test_df1)
    test_df1 = label_encoder(test_df1)
    test_df1 = test_df1.groupby('symbol').apply(add_features).reset_index(drop=True)
    # Tạo một expander để chứa danh sách ngân hàng
    with st.expander("Chọn loại cổ phiếu"):
        # Sidebar or selectbox to choose Ticker Symbol (Mode)
        ticker_options = test_df1['symbol'].unique()
        choose_ticker = st.selectbox("Choose Bank:",ticker_options)
        # Filter the DataFrame based on the selected mode
    test_df = test_df1[test_df1['symbol'] == choose_ticker]

    # Load the model from file
    model = load_model('frontend/models/lstm_model_ver1.keras')
    # Fit scalers on train
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_test_scaled = scaler_x.fit_transform(test_df[features])
    y_test_scaled = scaler_y.fit_transform(test_df[[target_col]])
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, time_steps)
    current_price = scaler_y.inverse_transform(y_test)
    predicted_price = predict(scaler_y,model,X_test)
    fig=visualize_chart(test_df,current_price,predicted_price)
    result_df = pd.DataFrame({
        'actual_price': current_price.reshape(-1),
        'predicted_price': predicted_price.reshape(-1),
        # 'predicted_growth_pct': growth_rate_pct
    })
    result = growth_percent_cal(result_df)
    # st.dataframe(result)
    with st.expander("Biểu đồ giá dự đoán và thực tế"):
        st.plotly_chart(fig)
    # price_growth_analysis(current_price,predicted_price)
    with st.expander("Tăng trưởng lũy kế"):
        import plotly.express as px
        fig = px.line(result, y=['cumulative_growth_GT', 'cumulative_growth_pred'],
                    labels={'value': 'Phần trăm tăng trưởng (%)', 'index': 'Thời gian'},
                    title='Biểu đồ tăng trưởng lũy kế: Thực tế vs Mô hình dự đoán')
        st.plotly_chart(fig)
    with st.expander("Tăng trưởng theo ngày"):
        import plotly.express as px
        fig = px.line(result, y=['daily_growth_pct_GT', 'daily_growth_pct_pred'],
              labels={'value': 'Tăng trưởng theo ngày (%)', 'index': 'Thời gian'},
              title='Biểu đồ tăng trưởng theo ngày: Thực tế vs Mô hình dự đoán')
        st.plotly_chart(fig)





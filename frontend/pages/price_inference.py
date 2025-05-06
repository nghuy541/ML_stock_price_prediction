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
import joblib



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
    fig.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines+markers', name="Price prediction"))
    fig.update_layout(title="Predicted Stock Chart (1 Year Ahead)",xaxis_title="Time", yaxis_title="Stock price")
    return fig

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

def rank_stock(df):
    with st.expander(f"🏆 Top stocks with high growth percentage"):
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
        chart_button = st.button("📊 View stock ranking chart ", help="Click to view analysis chart")

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

        if chart_button:
            st.success("✅ Completed analysis chart")
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

class LSTMPipeline():
    def __init__(self,df,features_cols,target_cols,symbol):
        self.df = df
        self.label_encoder = LabelEncoder()
        self.symbol = symbol
        self.features_cols = features_cols
        self.target_cols = target_cols
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.lstm_model = self.init_model()

    def group_stock_by_symbol(self):
        df = self.df[self.df['symbol'] == self.symbol].copy()
        return df

    def init_model(self):
        # Load model và scaler 1 lần ngoài vòng lặp
        model = load_model("frontend/models/lstm_model_ver1.keras")
        return model
    
    def add_features(self,data):
        data['lag_1'] = data['close'].shift(1)
        data['lag_2'] = data['close'].shift(2)
        data['rolling_mean_3'] = data['close'].rolling(window=3).mean()
        data['rolling_std_3'] = data['close'].rolling(window=3).std()
        data['momentum_1'] = data['close'].diff()
        data.fillna(method='bfill', inplace=True)
        data.fillna(0, inplace=True)
        return data

    def normalize_data(self,df,features_cols,target_cols):
        X_scaled = self.scaler_x.fit_transform(df[features_cols])
        y_scaled = self.scaler_y.fit_transform(df[[target_cols]])
        return X_scaled, y_scaled
    
    def create_features(self,df):
        df = self.add_features(df)
        df['index_encoded'] = self.label_encoder.fit_transform(df['symbol'])
        return df

    def forecast_next_days(self, data, scaler_y, y_test,n_days):
        last_sequence = data[-60:]  # (60, num_features)
        num_features = last_sequence.shape[1]
        predictions = []

        for _ in range(n_days):
            input_seq = np.expand_dims(last_sequence, axis=0)  # (1, 60, num_features)
            pred_scaled = self.lstm_model.predict(input_seq)  # (1, 1)
            pred_scaled = pred_scaled.reshape(-1, 1)
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred)

            # Cập nhật: tạo 1 step mới lặp lại pred_scaled thành 1 dòng có num_features
            new_step = np.tile(pred_scaled[0], (1, num_features))  # (1, num_features)
            last_sequence = np.append(last_sequence[1:], new_step, axis=0)
        # ✅ Inverse transform y_test here
        y_test_inversed = scaler_y.inverse_transform(y_test.reshape(-1, 1))#.flatten()
        return predictions,y_test_inversed

    def plot_predictions_with_growth_bk(self,y_true, y_pred, periods=[7, 21, 63, 252]):
        """
        y_true: np.array, shape (n, 1) — inverse-transformed ground truth
        y_pred: np.array, shape (n, 1) — inverse-transformed predicted values
        periods: list of time horizons (in days)
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        days = np.arange(1, len(y_pred) + 1)

        fig = go.Figure()

        # Add traces
        # fig.add_trace(go.Scatter(x=days, y=y_true, mode='lines+markers', name='Ground Truth', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=days, y=y_pred, mode='lines+markers', name='Predicted', line=dict(color='orange', dash='dash')))

        # Add growth percent annotations
        for period in periods:
            if len(y_pred) > period:
                start_price = y_pred[0]
                future_price = y_pred[period]
                growth = ((future_price - start_price) / start_price) * 100

                fig.add_annotation(
                    x=period,
                    y=future_price,
                    text=f"{period}d: {growth:.2f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor='green' if growth >= 0 else 'red',
                    font=dict(color='green' if growth >= 0 else 'red'),
                    ax=0,
                    ay=-40
                )

        fig.update_layout(
            title='📈 Predicted vs Ground Truth Prices with Growth %',
            xaxis_title='Days',
            yaxis_title='Price',
            legend=dict(x=0.01, y=0.99),
            template='plotly_white'
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def plot_predictions_with_growth_bk1(self, y_true, y_pred):
        """
        y_true: np.array, shape (n, 1) — inverse-transformed ground truth
        y_pred: np.array, shape (n, 1) — inverse-transformed predicted values
        """
        import numpy as np
        import plotly.graph_objects as go

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        days = np.arange(1, len(y_pred) + 1)

        # Growth time points in days
        growth_days = {
            'start_day': 0,
            '7 days': 7,
            '1 month': 21,
            '1 quarter': 63,
            '1 year': 252
        }

        fig = go.Figure()

        # Add predicted trace
        fig.add_trace(go.Scatter(x=days, y=y_pred, mode='lines+markers',
                                name='Predicted', line=dict(color='orange', dash='dash')))

        skipped_labels = []

        for label, day_offset in growth_days.items():
            if len(y_pred) > day_offset:
                price = y_pred[day_offset]
                if label == 'start_day':
                    fig.add_annotation(
                        x=day_offset,
                        y=price,
                        text=f"Start: {price:.2f}",
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowcolor='blue',
                        font=dict(color='blue'),
                        ax=0,
                        ay=-40
                    )
                else:
                    start_price = y_pred[0]
                    growth = ((price - start_price) / start_price) * 100
                    fig.add_annotation(
                        x=day_offset,
                        y=price,
                        text=f"{label}: {growth:.2f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowcolor='green' if growth >= 0 else 'red',
                        font=dict(color='green' if growth >= 0 else 'red'),
                        ax=0,
                        ay=-40
                    )
            else:
                skipped_labels.append(label)
                print(f"⚠️ Skipped '{label}' ({day_offset}d): only {len(y_pred)} days of data.")

        if skipped_labels:
            fig.add_annotation(
                x=len(y_pred),
                y=max(y_pred),
                text=f"Skipped: {', '.join(skipped_labels)}",
                showarrow=False,
                font=dict(color='gray', size=12),
                xanchor='right',
                yanchor='bottom'
            )

        fig.update_layout(
            title='📈 Predicted vs Ground Truth Prices with Growth %',
            xaxis_title='Days',
            yaxis_title='Price',
            legend=dict(x=0.01, y=0.99),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_predictions_with_growth(self, y_true, y_pred):
        """
        y_true: np.array, shape (n, 1) — inverse-transformed ground truth
        y_pred: np.array, shape (n, 1) — inverse-transformed predicted values
        """
        import numpy as np
        import plotly.graph_objects as go
        import streamlit as st

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        days = np.arange(1, len(y_pred) + 1)

        # Growth time points in days
        growth_days = {
            'start_day': 0,
            '7 days': 7,
            '1 month': 21,
            '1 quarter': 63,
            '1 year': 252
        }

        fig = go.Figure()

        # Add predicted trace
        fig.add_trace(go.Scatter(x=days, y=y_pred, mode='lines+markers',
                                name='Predicted', line=dict(color='orange', dash='dash')))

        skipped_labels = []
        metric_labels = []
        metric_values = []
        metric_deltas = []

        for label, day_offset in growth_days.items():
            if len(y_pred) > day_offset:
                price = y_pred[day_offset]
                if label == 'start_day':
                    fig.add_annotation(
                        x=day_offset,
                        y=price,
                        text=f"Start: {price:.2f}",
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowcolor='blue',
                        font=dict(color='blue'),
                        ax=0,
                        ay=-40
                    )
                else:
                    start_price = y_pred[0]
                    growth_pct = ((price - start_price) / start_price) * 100
                    fig.add_annotation(
                        x=day_offset,
                        y=price,
                        text=f"{label}: {growth_pct:.2f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowcolor='green' if growth_pct >= 0 else 'red',
                        font=dict(color='green' if growth_pct >= 0 else 'red'),
                        ax=0,
                        ay=-40
                    )
                    # Store metrics for display later
                    metric_labels.append(f"Growth after {label}")
                    metric_values.append(f"{growth_pct:.2f}%")
                    metric_deltas.append(f"{growth_pct:.2f}%")
            else:
                skipped_labels.append(label)
                print(f"⚠️ Skipped '{label}' ({day_offset}d): only {len(y_pred)} days of data.")

        # Display metrics in columns
        if metric_labels:
            cols = st.columns(len(metric_labels))
            for col, label, value, delta in zip(cols, metric_labels, metric_values, metric_deltas):
                col.metric(label=label, value=value, delta=delta)

        # Show skipped note on plot
        if skipped_labels:
            fig.add_annotation(
                x=len(y_pred),
                y=max(y_pred),
                text=f"Skipped: {', '.join(skipped_labels)}",
                showarrow=False,
                font=dict(color='gray', size=12),
                xanchor='right',
                yanchor='bottom'
            )

        fig.update_layout(
            title='📈 Predicted vs Ground Truth Prices with Growth %',
            xaxis_title='Days',
            yaxis_title='Price',
            legend=dict(x=0.01, y=0.99),
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)


    def run(self):
        df = self.group_stock_by_symbol()
        df = self.create_features(df)
        X_scaled, y_scaled = self.normalize_data(df,self.features_cols,self.target_cols)
        # Inverse transform y_test using the same scaler
        # y_test_inversed = self.target_scaler.inverse_transform(y_test)  
        y_pred,y_test= self.forecast_next_days(X_scaled,self.scaler_y,y_scaled,60)
        self.plot_predictions_with_growth(y_test, y_pred)
        return y_pred,y_test

class XGboostPipeline():
    def __init__(self,df,features_cols,target_cols,symbol):
        self.df = df
        self.label_encoder = LabelEncoder()
        self.symbol = symbol
        self.features_cols = features_cols
        self.target_cols = target_cols
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.xgb_model = self.init_model()

    def group_stock_by_symbol(self):
        df = self.df[self.df['symbol'] == self.symbol].copy()
        return df

    def add_features(self,data):
        data['lag_1'] = data['close'].shift(1)
        data['lag_2'] = data['close'].shift(2)
        data['rolling_mean_3'] = data['close'].rolling(window=3).mean()
        data['rolling_std_3'] = data['close'].rolling(window=3).std()
        data['momentum_1'] = data['close'].diff()
        data.fillna(method='bfill', inplace=True)
        data.fillna(0, inplace=True)
        return data

    def create_features(self,df):
        df = self.add_features(df)
        df['index_encoded'] = self.label_encoder.fit_transform(df['symbol'])
        return df

    def normalize_data(self,df,features_cols,target_cols):
        X_scaled = self.scaler_x.fit_transform(df[features_cols])
        y_scaled = self.scaler_y.fit_transform(df[[target_cols]])
        return X_scaled, y_scaled

    def preprocess_input_data(self,X_scaled, y_scaled):
        X_test_all, y_test_all = [], []
        X_test_all.append(X_scaled)
        y_test_all.append(y_scaled.ravel())
        X_test_final = np.concatenate(X_test_all)
        y_test_final = np.concatenate(y_test_all)
        return X_test_final,y_test_final

    def init_model(self):
        # Load model và scaler 1 lần ngoài vòng lặp
        xgb_model = joblib.load('frontend/models/xgboost_model.pkl')
        return xgb_model

    def inverse_to_price(self,price):
        inversed_prices = self.scaler_y.inverse_transform(price.reshape(1,-1))
        return inversed_prices

    def forecast_price(self,X_test,y_test):
        preds = self.xgb_model.predict(X_test)
        inversed_y_pred= self.inverse_to_price(preds)
        inversed_y_gt = self.inverse_to_price(y_test)
        return inversed_y_pred,inversed_y_gt

    def growth_analysis(self,preds,start_price=None):
        growth_data = []
        days = {
            "7 days": 7,
            "1 month": 21,
            "1 quarter": 63,
            "1 year": 252
        }

        # Convert to NumPy array and flatten
        preds = np.array(preds).flatten()

        # Use first prediction as start price if not given
        if start_price is None:
            start_price = preds[0]

        st.header("🔍 Forecast Growth Analysis")
        
        for label, day in days.items():
            if day <= len(preds):  # Make sure we don't exceed the prediction horizon
                future_price = preds[day - 1]  # 0-based indexing
                growth_pct = (future_price - start_price) / start_price * 100
                growth_data.append({'label': label, 'growth_pct': f"{growth_pct:.2f}%"})
                st.metric(label=f"Growth after {label}", value=f"{growth_pct:.2f}%")
            else:
                st.warning(f"Not enough prediction data for {label} ({day} days)")

        return growth_data



    def plot_chart1(self,preds,truth):
        import streamlit as st
        import plotly.graph_objects as go
        import numpy as np

        # Flatten in case they're nested
        preds = np.array(preds).flatten()
        truth = np.array(truth).flatten()

        # Generate x-axis (Day 1 to Day 7)
        days = list(range(1, len(preds) + 1))

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=days, y=truth, mode='lines+markers', name='Ground Truth',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=days, y=preds, mode='lines+markers', name='Predicted',
            line=dict(color='orange', width=2, dash='dash')
        ))

        # Customize layout
        fig.update_layout(
            title='📈 Predicted vs Ground Truth Prices',
            xaxis_title='Days',
            yaxis_title='Price',
            legend=dict(x=0.01, y=0.99),
            template='plotly_white'
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_chart(self, preds):
        import streamlit as st
        import plotly.graph_objects as go
        import numpy as np

        # Sample predicted prices
        preds = np.array(preds).flatten()[:300]

        # Start price (e.g., the latest known actual price before prediction)
        start_price = preds[0]

        # Growth time points in days (assuming 1 prediction per day)
        growth_days = {
            'start_day': 0,
            "7 days": 7,
            "1 month": 21,
            "1 quarter": 63,
            "1 year": 252
        }

        # Create x-axis
        days = list(range(1, len(preds) + 1))

        # Create figure
        fig = go.Figure()

        # Add prediction line
        fig.add_trace(go.Scatter(
            x=days, y=preds, mode='lines+markers', name='Predicted Price',
            line=dict(color='orange', width=2, dash='dash')
        ))

        # Add annotations for percent growth
        for label, day in growth_days.items():
            if day <= len(preds):
                future_price = preds[day - 1]
                if label == 'start_day':
                    growth_pct = 0
                    fig.add_annotation(
                        x=day,
                        y=start_price,
                        text=f"{label}: {growth_pct:.2f}%",
                        showarrow=True,
                        arrowcolor='green' if growth_pct >= 0 else 'red',
                        arrowhead=1,
                        ax=0,
                        ay=-40,
                        font=dict(color="green" if growth_pct >= 0 else "red")
                    )
                else:
                    growth_pct = (future_price - start_price) / start_price * 100
                    fig.add_annotation(
                        x=day,
                        y=future_price,
                        text=f"{label}: {growth_pct:.2f}%",
                        showarrow=True,
                        arrowcolor='green' if growth_pct >= 0 else 'red',
                        arrowhead=1,
                        ax=0,
                        ay=-40,
                        font=dict(color="green" if growth_pct >= 0 else "red")
                    )

        # Customize layout
        fig.update_layout(
            title='📈 Predicted Price with Growth Annotations',
            xaxis_title='Days',
            yaxis_title='Price',
            template='plotly_white'
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)


    def run(self):
        df = self.group_stock_by_symbol()
        df = self.create_features(df)
        X_scaled, y_scaled = self.normalize_data(df,self.features_cols,self.target_cols)
        X_test,y_test = self.preprocess_input_data(X_scaled, y_scaled)
        inversed_y_pred,inversed_y_gt= self.forecast_price(X_test,y_test)
        return inversed_y_pred,inversed_y_gt

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
    # X_scaled,y_scaled,scaler_x,scaler_y = preprocess_and_normalize_data(df_symbol,features,target)
    preds = forecast_next_days(model, X_scaled, scaler_x, scaler_y, n_days=252)
    # preds = batch_forecast_next_days(model, X_scaled, scaler_x, scaler_y, n_days=252)
    last_close = df_symbol['close'].iloc[-1]
    last_date = df_symbol['time'].iloc[-1]
    growth_data = growth_analysis(last_close, preds)
    # for result in results:
    # st.plotly_chart(plot_predictions(preds, last_date), use_container_width=True)
    # st.plotly_chart(plot_predictions(preds, last_date), use_container_width=True)
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
        title="Average percentage growth",
        xaxis_title="Symbol",
        yaxis_title="Growth(%)",
        barmode="group"
    )
    # Show in Streamlit
    st.plotly_chart(fig)

def wrapper(symbol_df_tuple):
    symbol, df = symbol_df_tuple
    return process_symbol(symbol, df)


def run_bk():
    st.title("Overview of growth percentage")
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
        for col in ["7 days", "1 month", "1 quarter"]:
            df[col] = df[col].str.replace('%', '').astype(float)
        # Group by symbol and calculate the mean for the 3 columns
        mean_df = df.groupby("symbol")[["7 days", "1 month", "1 quarter"]].mean().round(2)
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
    st.title("Overview of growth percentage")
    file_path = "frontend/stock_price_with_growth.csv"
    df = pd.read_csv("frontend/stock_price.csv")
    df = convert_time(df)
    symbols = df['symbol'].unique()
    # rf_model = joblib.load('frontend/models/random_forest_model.pkl')
    list_model = ['lstm','xgboost']
    features = ['open', 'high', 'low', 'volume', 'index_encoded',
                'lag_1', 'lag_2', 'rolling_mean_3', 'rolling_std_3', 'momentum_1']
    target = 'close'

    def thread_worker(symbol):
        return process_symbol(symbol, df, model)

    with st.expander(f"🏆Stock growth trend analysis"):
        choose_stock = st.selectbox("Choose stocks:",symbols)
        choose_model = st.selectbox("Choose models:",list_model)
        analyze_trend = st.button("Trend analysis")
        if analyze_trend:
            if choose_model == 'lstm':
                pipeline = LSTMPipeline(df,features,target,choose_stock)
                # model = load_model("frontend/models/lstm_model_ver1.keras")
                # process_single_symbol(choose_stock, df, model)
                # X_scaled, y_scaled,pred = pipeline.run()
                pred,y_scale = pipeline.run()
                # st.write(f'{X_scaled.shape}|{y_scaled.shape}|{pred.shape}')

            if choose_model == 'xgboost':
                pipeline = XGboostPipeline(df,features,target,choose_stock)
                inversed_y_pred,inversed_y_gt= pipeline.run()
                growth_data = pipeline.growth_analysis(inversed_y_pred)
                pipeline.plot_chart(inversed_y_pred)

    if os.path.exists(file_path):
        print(f"The file '{file_path}' exists.")
        df = pd.read_csv(file_path)
        # Remove percentage signs and convert columns to numeric
        for col in ["7 days", "1 month", "1 quarter"]:
            df[col] = df[col].str.replace('%', '').astype(float)
        # Group by symbol and calculate the mean for the 3 columns
        mean_df = df.groupby("symbol")[["7 days", "1 month", "1 quarter"]].mean().round(2)
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
    

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import load_model
import plotly.graph_objects as go
from datetime import timedelta
import plotly.express as px

def convert_time(data):
    data['time'] = pd.to_datetime(data['time'])
    return data.sort_values(by=['symbol', 'time'])

def visualize_chart():
    df = pd.read_csv("frontend/stock_price.csv")
    df = convert_time(df)
    time_stamp = ['1 tuần','1 tháng','1 quý','1 năm']
    years = list(range(2014, 2025))  # [2004, 2005, ..., 2025]
    year_stamp = [str(year) for year in years]
    # year_stamp = 
    with st.expander("Choose symbol"):
        symbols = df['symbol'].unique()
        symbol = st.selectbox("Choose symbol", symbols)
        choose_year = st.selectbox("Year:", year_stamp)
        # time_stamp = {
        #     "1 tuần": pd.DateOffset(weeks=1),
        #     "1 tháng": pd.DateOffset(months=1),
        #     "1 quý": pd.DateOffset(months=3),
        #     "1 năm": pd.DateOffset(years=1)
        # }
        # time_label = st.selectbox("Chọn khoảng thời gian:", list(time_stamp.keys()))
        quarter_options = ['Q1', 'Q2', 'Q3', 'Q4']
        selected_quarter = st.selectbox("Choose quarter:", quarter_options)
        quarter_map = {
                'Q1': 1,
                'Q2': 2,
                'Q3': 3,
                'Q4': 4,
        }
        selected_q_number = quarter_map[selected_quarter]
    df_symbol = df[df['symbol'] == symbol].copy()
    df_symbol_display = df_symbol.copy()
    df_symbol_display['date'] = pd.to_datetime(df_symbol_display['time'])  # giữ nguyên kiểu datetime

    # Chia nho them thanh tung quy
    df_symbol_display['quarter'] = df_symbol_display['date'].dt.quarter
    df_symbol_display['year'] = df_symbol_display['date'].dt.year.astype(str)
    df_symbol_display = df_symbol_display[df_symbol_display['year'] == choose_year]
    df_quarter = df_symbol_display[df_symbol_display['quarter'] == selected_q_number]
    with st.expander(f"📈📊 Close chart - {selected_quarter} year {choose_year}"):
        fig = px.line(df_quarter, x='date', y='close', title='Growth chart (Close)')
        fig.update_traces(line_color='green')
        st.plotly_chart(fig, use_container_width=True)
    


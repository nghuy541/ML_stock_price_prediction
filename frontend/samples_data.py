import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 30 rows of fake data
num_rows = 30
years = np.arange(1995, 2025)[-num_rows:]
tickers = ['ACB', 'VCB', 'TCB']
ticker_symbols = np.random.choice(tickers, size=num_rows)
ticker_map = {sym: idx for idx, sym in enumerate(tickers)}

# Generate realistic financial values
data = pd.DataFrame({
    'year': years,
    'open': np.random.uniform(20, 60, size=num_rows),
})

# High is slightly higher than open
data['high'] = data['open'] + np.random.uniform(0.1, 2.0, size=num_rows)

# Low is slightly lower than open
data['low'] = data['open'] - np.random.uniform(0.1, 2.0, size=num_rows)

# Close is between low and high
data['close'] = data[['low', 'high']].apply(lambda row: np.random.uniform(row['low'], row['high']), axis=1)

# Add volume in realistic range
data['volume'] = np.random.randint(1e5, 5e6, size=num_rows)

# Add company-related financials
data['Ticker_symbol'] = ticker_symbols
data['ROE'] = np.random.uniform(0.05, 0.3, size=num_rows)
data['ROA'] = np.random.uniform(0.01, 0.1, size=num_rows)
data['PE'] = np.random.uniform(5, 20, size=num_rows)
data['PB'] = np.random.uniform(0.5, 3.0, size=num_rows)
data['Ticker_encoded'] = data['Ticker_symbol'].map(ticker_map)

# Preview
print(data.head())

# Save to CSV
data.to_csv('frontend/fake_stock_data.csv', index=False)

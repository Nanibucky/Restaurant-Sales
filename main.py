import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 1. Load and prepare data
def load_data(filepath):
    # Read data with explicit date format (DD-MM-YYYY)
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df.set_index('date', inplace=True)
    
    print("Date range:", df.index.min(), "to", df.index.max())
    
    # Calculate total sales and handle missing values
    df['total_sales'] = df['inside_sales'] + df['outside_sales'].fillna(0)
    
    # Basic cleaning
    df = df.sort_index()
    df['total_sales'] = df['total_sales'].replace(0, np.nan)
    df['total_sales'] = df['total_sales'].interpolate()
    
    # Print basic statistics
    print("\nSales Statistics:")
    print(df['total_sales'].describe())
    
    return df['total_sales']
def validate_data(data):
    """Basic data validation and information"""
    print("\nData Validation:")
    print(f"Total records: {len(data)}")
    print(f"Missing values: {data.isna().sum()}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Minimum sales: ${data.min():.2f}")
    print(f"Maximum sales: ${data.max():.2f}")
    print(f"Average sales: ${data.mean():.2f}")
    
    # Plot original data
    plt.figure(figsize=(12,6))
    plt.plot(data)
    plt.title('Original Sales Data')
    plt.xlabel('Date')
    plt.ylabel('Sales ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# 2. Check stationarity
def check_stationarity(data):
    # Perform ADF test
    result = adfuller(data.dropna())
    
    print('Stationarity Test Results:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
        
    # Plot rolling statistics
    plt.figure(figsize=(10,6))
    plt.plot(data, label='Original')
    plt.plot(data.rolling(7).mean(), label='Rolling Mean')
    plt.plot(data.rolling(7).std(), label='Rolling Std')
    plt.legend()
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    return result[1] < 0.05

# 3. Plot ACF and PACF
def plot_acf_pacf(data, lags=40):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
    plot_acf(data, lags=lags, ax=ax1)
    plot_pacf(data, lags=lags, ax=ax2)
    plt.tight_layout()
    plt.show()

# 4. Decompose time series
def decompose_series(data):
    decomposition = seasonal_decompose(data, period=7)
    
    plt.figure(figsize=(12,10))
    plt.subplot(411)
    plt.plot(data)
    plt.title('Original')
    plt.subplot(412)
    plt.plot(decomposition.trend)
    plt.title('Trend')
    plt.subplot(413)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal')
    plt.subplot(414)
    plt.plot(decomposition.resid)
    plt.title('Residual')
    plt.tight_layout()
    plt.show()

# 5. Fit models and make predictions
def fit_and_predict(train, test, order=(1,1,1), seasonal_order=(1,1,1,7)):
    # Fit ARIMA
    arima = ARIMA(train, order=order).fit()
    arima_pred = arima.forecast(len(test))
    
    # Fit SARIMA
    sarima = SARIMAX(train, 
                     order=order, 
                     seasonal_order=seasonal_order).fit(disp=False)
    sarima_pred = sarima.forecast(len(test))
    
    return arima_pred, sarima_pred, arima, sarima

# 6. Calculate metrics
def calculate_metrics(actual, predicted, model_name):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f'\n{model_name} Metrics:')
    print(f'RMSE: ${rmse:.2f}')
    print(f'MAE: ${mae:.2f}')
    print(f'MAPE: {mape:.2f}%')
    
    return rmse, mae, mape

# 7. Plot results
def plot_predictions(train, test, arima_pred, sarima_pred):
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, arima_pred, label='ARIMA Predictions')
    plt.plot(test.index, sarima_pred, label='SARIMA Predictions')
    plt.title('Sales Forecasting Results')
    plt.legend()
    plt.show()
def main():
    # 1. Load data
    print("Loading data...")
    sales = load_data('here goes ur data file path')
    
    # 2. Validate data
    print(validate_data(sales))
    
    # 3. Check stationarity
    print("\nChecking stationarity...")
    is_stationary = check_stationarity(sales)
    print(f"\nData is{' ' if is_stationary else ' not '} stationary")
    
    # 4. Plot ACF/PACF of original and differenced data
    print("\nPlotting ACF/PACF for original data...")
    plot_acf_pacf(sales)
    
    print("\nPlotting ACF/PACF for differenced data...")
    plot_acf_pacf(sales.diff().dropna())
    
    # 5. Decompose series
    print("\nDecomposing time series...")
    decompose_series(sales)
    
    # 6. Split data
    train_size = int(len(sales) * 0.8)
    train = sales[:train_size]
    test = sales[train_size:]
    print(f"\nTrain size: {len(train)}, Test size: {len(test)}")
    
    # 7. Fit models and predict
    print("\nFitting models and making predictions...")
    arima_pred, sarima_pred, arima_model, sarima_model = fit_and_predict(train, test)
    
    # 8. Calculate metrics
    calculate_metrics(test, arima_pred, "ARIMA")
    calculate_metrics(test, sarima_pred, "SARIMA")
    
    # 9. Plot results
    print("\nPlotting results...")
    plot_predictions(train, test, arima_pred, sarima_pred)
    
    # 10. Future forecast
    print("\nGenerating future forecast...")
    future_steps = 30
    future_arima = arima_model.forecast(future_steps)
    future_sarima = sarima_model.forecast(future_steps)
    
    # Plot future forecast
    plt.figure(figsize=(12,6))
    plt.plot(sales.index, sales, label='Historical Data')
    plt.plot(pd.date_range(start=sales.index[-1], periods=future_steps+1)[1:],
             future_arima, label='ARIMA Forecast')
    plt.plot(pd.date_range(start=sales.index[-1], periods=future_steps+1)[1:],
             future_sarima, label='SARIMA Forecast')
    plt.title('Future Sales Forecast')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
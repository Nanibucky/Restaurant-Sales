# Restaurant Sales Forecasting Using Time Series Analysis

## Project Overview
A time series analysis project implementing ARIMA and SARIMA models to forecast restaurant sales. The project analyzes 2.5 years of daily sales data to predict future revenue trends and provide actionable business insights.

## Key Features
- Time Series Analysis with ARIMA and SARIMA models
- Stationarity Testing and Transformation
- ACF/PACF Analysis
- Seasonal Pattern Recognition
- Sales Forecasting with Confidence Intervals
- Model Performance Comparison

## Results
The SARIMA model demonstrated superior performance:
- SARIMA: RMSE: $2,637.83, MAPE: 10.24%
- ARIMA: RMSE: $3,332.19, MAPE: 13.09%

## Dataset Information
- Time Period: January 2017 to June 2019
- Records: 910 daily sales entries
- Average Daily Sales: $16,591
- Total Revenue Analyzed: ~$15.1M

## Technical Implementation
### Data Preprocessing
- Date parsing and formatting
- Missing value handling
- Outlier detection and treatment
- Time series validation

### Analysis Steps
1. Stationarity Testing
   - ADF Test (Statistic: -2.21)
   - P-value analysis
   - Transformation if needed

2. Pattern Recognition
   - ACF/PACF plotting
   - Seasonal decomposition
   - Trend analysis

3. Model Building
   - ARIMA implementation
   - SARIMA with seasonal components
   - Parameter optimization

4. Performance Evaluation
   - Error metrics calculation
   - Model comparison
   - Forecast visualization

## Requirements
```python
pandas==2.0.0
numpy==1.24.0
matplotlib==3.7.0
statsmodels==0.14.0
scikit-learn==1.2.0
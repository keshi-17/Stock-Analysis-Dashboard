# Stock Analysis Dashboard

A comprehensive web-based stock analysis application built with Python Dash that provides advanced financial analytics including technical analysis, volatility modeling, GARCH analysis, ARIMA forecasting, and multi-stock comparison.

## Features

### Technical Analysis
- **Moving Averages**: 50-day and 200-day moving averages
- **Bollinger Bands**: 20-period bands with 2 standard deviations
- **Price Charts**: Interactive candlestick and line charts
- **Technical Indicators**: Multiple technical analysis tools

###  Volatility Analysis
- **Rolling Volatility**: 30-day annualized volatility calculations
- **Volatility Clustering**: Visual representation of volatility patterns
- **Risk Assessment**: Historical volatility metrics

### GARCH Modeling
- **Advanced Volatility Models**: GARCH(1,1), GARCH(1,2), GARCH(2,1)
- **Multiple Distributions**: Normal, Student-t, and Skewed-t distributions
- **Conditional Volatility**: Real-time volatility forecasting
- **Model Diagnostics**: Automatic model selection and validation

### ARIMA Forecasting
- **Price Prediction**: ARIMA(1,1,1) model for price forecasting
- **Confidence Intervals**: Statistical confidence bands
- **Future Projections**: 30-day business day forecasts
- **Trend Analysis**: Automated trend detection

### Multi-Stock Comparison
- **Performance Metrics**: Total return, volatility, Sharpe ratio, max drawdown
- **Correlation Analysis**: Interactive correlation heatmap
- **Normalized Comparison**: Base-100 price normalization for easy comparison
- **Risk-Return Visualization**: Comprehensive performance dashboard

##Technology Stack

- **Frontend**: Dash (Python web framework)
- **Data Source**: Yahoo Finance API (yfinance)
- **Financial Modeling**: 
  - ARCH/GARCH models for volatility
  - ARIMA models for forecasting
  - Statistical analysis with pandas/numpy
- **Visualization**: Plotly for interactive charts
- **Styling**: Custom dark theme with responsive design

## Prerequisites

- Python 3.7 or higher
- Internet connection for real-time data fetching
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-analysis-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare ticker data (Optional)**
   - Create a `ticker.csv` file with a `SYMBOL` column containing stock symbols
   - If no CSV is provided, the app will use default Indian stock tickers

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the dashboard**
   - Open your browser and navigate to `http://127.0.0.1:8050`

## Project Structure

```
stock-analysis-dashboard/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── ticker.csv            # Stock symbols (optional)
└── README.md             # This file
```

## Usage Guide

### Getting Started
1. **Select a Stock**: Choose from the dropdown menu or add your own tickers
2. **Set Date Range**: Use the date picker to select your analysis period
3. **Choose Analysis Type**: Navigate through the tabs for different analyses

### Tab Descriptions

#### Technical Analysis
- View moving averages and Bollinger Bands
- Analyze price trends and support/resistance levels
- Requires minimum 200 data points for 200-day MA

#### Volatility Analysis
- Monitor 30-day rolling volatility
- Identify high and low volatility periods
- Assess risk levels over time

#### GARCH Modeling
- Advanced volatility modeling and forecasting
- Multiple model specifications with automatic selection
- Conditional volatility estimation
- Requires minimum 100 data points

#### ARIMA Forecasting
- Price prediction with statistical confidence
- 30-day business day forecasts
- Trend analysis and future price projections
- Requires minimum 50 data points

#### Stock Comparison
- Compare multiple stocks (2-10 stocks)
- Performance metrics dashboard
- Correlation analysis
- Risk-return visualization

## 🔧 Configuration

### Adding Custom Tickers
Create a `ticker.csv` file with the following format:
```csv
  RELIANCE.NS
  TCS.NS
  HDFCBANK.NS
  BHARTIARTL.NS
  ICICIBANK.NS
  INFY.NS
  SBIN.NS
  ITC.NS
```

### Supported Markets
- **US Stocks**: Use symbols like `AAPL`, `MSFT`, `GOOGL`
- **Indian Stocks**: Use `.NS` suffix like `RELIANCE.NS`, `TCS.NS`
- **Other Markets**: Follow Yahoo Finance symbol conventions

## Features

### Dark Theme UI
- Modern dark theme optimized for extended use
- Responsive design for desktop and mobile
- Interactive charts with hover details
- Professional color scheme

### Data Handling
- Automatic data validation and cleaning
- Error handling for missing or invalid data
- Outlier detection and management
- Real-time data fetching from Yahoo Finance

### Performance Optimization
- Efficient data processing with pandas
- Cached calculations for better performance
- Optimized chart rendering with Plotly

### Error Handling

The application includes comprehensive error handling for:
- **Data Availability**: Graceful handling of missing data
- **Model Convergence**: Fallback strategies for GARCH models
- **API Limits**: Retry mechanisms for data fetching
- **Invalid Inputs**: User-friendly error messages

## Metrics Calculated

### Performance Metrics
- **Total Return**: Percentage gain/loss over period
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline

### Technical Indicators
- **Moving Averages**: 50-day and 200-day
- **Bollinger Bands**: 20-period with 2σ bands
- **Rolling Volatility**: 30-day annualized

### Future Enhancements

- [ ] Additional technical indicators (RSI, MACD, Stochastic)
- [ ] More sophisticated forecasting models
- [ ] Portfolio optimization tools
- [ ] Real-time alerts and notifications
- [ ] Export functionality for charts and data
- [ ] User authentication and saved preferences

## Known Issues

- GARCH models may fail to converge with highly volatile or limited data
- Yahoo Finance API may occasionally return incomplete data
- Some international markets may have limited data availability

**Note**: This application is for educational and research purposes only. Always consult with a qualified financial advisor before making investment decisions.
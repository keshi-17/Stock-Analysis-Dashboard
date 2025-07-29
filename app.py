import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import acf, pacf
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os
warnings.filterwarnings('ignore')

try:
    csv = pd.read_csv("ticker.csv")
    TICKERS = csv["SYMBOL"].tolist()
except:
    TICKERS = ['BBOX.NS', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 
               'WIPRO.NS', 'LT.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS']

app = dash.Dash(__name__)
server = app.server
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Remove all default margins and padding */
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            html, body {
                margin: 0 !important;
                padding: 0 !important;
                background-color: #1F1F1F !important;
                min-height: 100vh;
                font-family: Arial, sans-serif;
            }
            
            #react-entry-point {
                margin: 0 !important;
                padding: 0 !important;
                background-color: #1F1F1F;
            }
            
            /* Dark theme for dropdown */
            .Select-menu-outer {
                background-color: #2F2F2F !important;
                border: 1px solid #404040 !important;
            }
            .Select-option {
                background-color: #2F2F2F !important;
                color: #E0E0E0 !important;
            }
            .Select-option:hover {
                background-color: #404040 !important;
                color: #FFFFFF !important;
            }
            .Select-control {
                background-color: #2F2F2F !important;
                border: 1px solid #404040 !important;
            }
            .Select-value-label {
                color: #E0E0E0 !important;
            }
            .Select-arrow {
                border-color: #E0E0E0 transparent transparent !important;
            }
            
            /* Dark theme for date picker */
            .DateRangePickerInput {
                background-color: #2F2F2F !important;
                border: 1px solid #404040 !important;
            }
            .DateInput {
                background-color: #2F2F2F !important;
            }
            .DateInput_input {
                background-color: #2F2F2F !important;
                color: #E0E0E0 !important;
                border: none !important;
            }
            .DateInput_input::placeholder {
                color: #B0B0B0 !important;
            }
            .DateRangePickerInput_arrow {
                color: #E0E0E0 !important;
            }
            .DateRangePickerInput_clearDates {
                color: #E0E0E0 !important;
            }
            .DateRangePickerInput_clearDates:hover {
                color: #FF6B6B !important;
            }
            
            /* Calendar popup styling */
            .DayPicker {
                background-color: #2F2F2F !important;
                border: 1px solid #404040 !important;
            }
            .DayPicker_weekHeader {
                background-color: #404040 !important;
                color: #E0E0E0 !important;
            }
            .CalendarDay {
                background-color: #2F2F2F !important;
                color: #E0E0E0 !important;
                border: 1px solid #404040 !important;
            }
            .CalendarDay:hover {
                background-color: #404040 !important;
                color: #FFFFFF !important;
            }
            .CalendarDay__selected {
                background-color: #00D4FF !important;
                color: #1F1F1F !important;
            }
            .CalendarDay__selected:hover {
                background-color: #00B8E6 !important;
            }
            .DayPickerNavigation_button {
                background-color: #404040 !important;
                color: #E0E0E0 !important;
                border: 1px solid #505050 !important;
            }
            .DayPickerNavigation_button:hover {
                background-color: #505050 !important;
            }
            .DayPicker_monthTable {
                background-color: #2F2F2F !important;
            }
            .CalendarMonth_caption {
                color: #E0E0E0 !important;
                background-color: #404040 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    html.H1("Stock Analysis Dashboard", 
            style={
                'text-align': 'center', 
                'margin-bottom': '30px',
                'color': '#E0E0E0',
                'font-family': 'Arial, sans-serif'
            }),
    
    html.Div([
        html.Div([
            html.Label("Select Stock:", style={
                'font-weight': 'bold',
                'color': '#E0E0E0',
                'margin-bottom': '10px',
                'display': 'block'
            }),
            dcc.Dropdown(
                id='stock-dropdown',
                options=[{'label': ticker, 'value': ticker} for ticker in TICKERS],
                value=TICKERS[0] if TICKERS else 'BBOX.NS',
                style={'width': '300px'},
                className='dark-dropdown'
            )
        ], style={'margin': '20px', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.Label("Select Stocks for Comparison (Max 10):", style={
                'font-weight': 'bold',
                'color': '#E0E0E0',
                'margin-bottom': '10px',
                'display': 'block'
            }),
            dcc.Dropdown(
                id='multi-stock-dropdown',
                options=[{'label': ticker, 'value': ticker} for ticker in TICKERS],
                value=TICKERS[:5] if len(TICKERS) >= 5 else TICKERS,
                multi=True,
                style={'width': '400px'},
                className='dark-dropdown'
            )
        ], style={'margin': '20px', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        # Date range picker
        html.Div([
            html.Label("Select Date Range:", style={
                'font-weight': 'bold',
                'color': '#E0E0E0',
                'margin-bottom': '10px',
                'display': 'block'
            }),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date='2020-01-01',
                end_date=pd.Timestamp.now().strftime('%Y-%m-%d'),
                display_format='YYYY-MM-DD',
                className='dark-date-picker'
            )
        ], style={'margin': '20px', 'display': 'inline-block', 'vertical-align': 'top'}),
    ], style={'background-color': '#2F2F2F', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '20px'}),
    
    dcc.Tabs(id="tabs", value='tab-1', 
             style={'backgroundColor': '#1F1F1F'},
             children=[
        dcc.Tab(label='Technical Analysis', value='tab-1',
                style={'backgroundColor': '#2F2F2F', 'color': '#E0E0E0'},
                selected_style={'backgroundColor': '#404040', 'color': '#FFFFFF'}),
        dcc.Tab(label='Volatility Analysis', value='tab-2',
                style={'backgroundColor': '#2F2F2F', 'color': '#E0E0E0'},
                selected_style={'backgroundColor': '#404040', 'color': '#FFFFFF'}),
        dcc.Tab(label='GARCH Modeling', value='tab-3',
                style={'backgroundColor': '#2F2F2F', 'color': '#E0E0E0'},
                selected_style={'backgroundColor': '#404040', 'color': '#FFFFFF'}),
        dcc.Tab(label='ARIMA Forecasting', value='tab-4',
                style={'backgroundColor': '#2F2F2F', 'color': '#E0E0E0'},
                selected_style={'backgroundColor': '#404040', 'color': '#FFFFFF'}),
        dcc.Tab(label='Stock Comparison', value='tab-5',
                style={'backgroundColor': '#2F2F2F', 'color': '#E0E0E0'},
                selected_style={'backgroundColor': '#404040', 'color': '#FFFFFF'}),
    ]),
    
    html.Div(id='tab-content', style={'backgroundColor': '#1F1F1F', 'minHeight': '100vh'})
], style={'backgroundColor': '#1F1F1F', 'minHeight': '100vh', 'padding': '20px'})

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data using yfinance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if data.empty:
            return None
        
        if len(data.columns) == 6:
            expected_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            data.columns = expected_cols
        else:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if 'Adj Close' in data.columns:
                required_columns.append('Adj Close')
            
            available_columns = [col for col in required_columns if col in data.columns]
            data = data[available_columns]
        
        data[ticker] = data['Close']
        return data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def fetch_multiple_stocks_data(tickers, start_date, end_date):
    """Fetch data for multiple stocks"""
    all_data = {}
    
    for ticker in tickers:
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            all_data[ticker] = data['Close']
    
    if not all_data:
        return None
    
    combined_data = pd.DataFrame(all_data)
    return combined_data.dropna()

def normalize_prices(data, base_value=100):
    """Normalize prices to a base value for comparison"""
    return data.div(data.iloc[0]) * base_value

def calculate_performance_metrics(data):
    """Calculate performance metrics for stocks"""
    metrics = {}
    
    for col in data.columns:
        returns = data[col].pct_change().dropna()
        
        total_return = (data[col].iloc[-1] / data[col].iloc[0] - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        max_drawdown = ((data[col] / data[col].cummax()) - 1).min() * 100
        
        metrics[col] = {
            'Total Return (%)': round(total_return, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Current Price': round(data[col].iloc[-1], 2)
        }
    
    return metrics

def fit_garch_model(returns, max_attempts=5):
    """Fit GARCH model with multiple fallback strategies"""
    returns_pct = returns * 100
    returns_pct = returns_pct.dropna()
    
    q1 = returns_pct.quantile(0.01)
    q99 = returns_pct.quantile(0.99)
    returns_pct = returns_pct.clip(lower=q1, upper=q99)
    
    if len(returns_pct) < 100:
        raise ValueError("Insufficient data for GARCH modeling")
    
    garch_specs = [
        {'p': 1, 'q': 1, 'dist': 'normal'},
        {'p': 1, 'q': 1, 'dist': 't'},
        {'p': 1, 'q': 2, 'dist': 'normal'},
        {'p': 2, 'q': 1, 'dist': 'normal'},
        {'p': 1, 'q': 1, 'dist': 'skewt'}
    ]
    
    for i, spec in enumerate(garch_specs):
        try:
            model = arch_model(
                returns_pct, 
                vol='GARCH', 
                p=spec['p'], 
                q=spec['q'], 
                dist=spec['dist']
            )
            
            results = model.fit(
                disp='off',
                options={'maxiter': 1000},
                update_freq=0
            )
            
            return results, spec['p'], spec['q']
            
        except Exception as e:
            print(f"GARCH({spec['p']},{spec['q']}) with {spec['dist']} distribution failed: {e}")
            if i == len(garch_specs) - 1:
                raise ValueError("All GARCH model specifications failed")
            continue
    
    raise ValueError("Unable to fit GARCH model")

@callback(Output('tab-content', 'children'),
          [Input('tabs', 'value'),
           Input('stock-dropdown', 'value'),
           Input('multi-stock-dropdown', 'value'),
           Input('date-picker-range', 'start_date'),
           Input('date-picker-range', 'end_date')])
def render_content(tab, selected_ticker, multi_selected_tickers, start_date, end_date):
    
    if tab == 'tab-5':  
        if not multi_selected_tickers or len(multi_selected_tickers) < 2:
            return html.Div([
                html.H3("Stock Comparison", style={'color': '#E0E0E0'}),
                html.P("Please select at least 2 stocks for comparison.", style={'color': '#FF6B6B'}),
                html.P("You can select up to 10 stocks from the dropdown above.", style={'color': '#B0B0B0'})
            ], style={'padding': '20px'})
        
        if len(multi_selected_tickers) > 10:
            multi_selected_tickers = multi_selected_tickers[:10]
        
        comparison_data = fetch_multiple_stocks_data(multi_selected_tickers, start_date, end_date)
        
        if comparison_data is None or comparison_data.empty:
            return html.Div([
                html.H3("Error Loading Comparison Data", style={'color': '#FF6B6B'}),
                html.P("Could not fetch data for the selected stocks. Please try different stocks or date range.", 
                       style={'color': '#E0E0E0'})
            ], style={'padding': '20px'})
        
        normalized_data = normalize_prices(comparison_data)
        
        metrics = calculate_performance_metrics(comparison_data)
        
        # Color palette for different stocks
        colors = ['#00D4FF', '#FFB347', '#FF6B6B', '#4ECDC4', '#45B7D1', 
                  '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        
        fig1 = go.Figure()
        for i, col in enumerate(normalized_data.columns):
            fig1.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig1.update_layout(
            title='Normalized Price Comparison (Base = 100)',
            xaxis_title='Date',
            yaxis_title='Normalized Price',
            hovermode='x unified',
            template='plotly_dark',
            height=600,
            paper_bgcolor='#1F1F1F',
            plot_bgcolor='#2F2F2F',
            font=dict(color='#E0E0E0')
        )
        
        returns_data = comparison_data.pct_change().dropna() * 100
        
        fig2 = go.Figure()
        for i, col in enumerate(returns_data.columns):
            fig2.add_trace(go.Scatter(
                x=returns_data.index,
                y=returns_data[col],
                mode='lines',
                name=f'{col} Returns',
                line=dict(color=colors[i % len(colors)], width=1.5)
            ))
        
        fig2.update_layout(
            title='Daily Returns Comparison (%)',
            xaxis_title='Date',
            yaxis_title='Daily Returns (%)',
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            paper_bgcolor='#1F1F1F',
            plot_bgcolor='#2F2F2F',
            font=dict(color='#E0E0E0')
        )
        
        correlation_matrix = returns_data.corr()
        
        fig3 = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdYlBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig3.update_layout(
            title='Correlation Matrix - Daily Returns',
            template='plotly_dark',
            height=500,
            paper_bgcolor='#1F1F1F',
            plot_bgcolor='#2F2F2F',
            font=dict(color='#E0E0E0')
        )
        
        metrics_df = pd.DataFrame(metrics).T
        
        summary_cards = []
        for i, (stock, data) in enumerate(metrics.items()):
            card_color = colors[i % len(colors)]
            card = html.Div([
                html.H4(stock, style={'color': card_color, 'margin-bottom': '10px'}),
                html.P(f"Total Return: {data['Total Return (%)']}%", style={'color': '#E0E0E0', 'margin': '5px 0'}),
                html.P(f"Volatility: {data['Volatility (%)']}%", style={'color': '#E0E0E0', 'margin': '5px 0'}),
                html.P(f"Sharpe Ratio: {data['Sharpe Ratio']}", style={'color': '#E0E0E0', 'margin': '5px 0'}),
                html.P(f"Max Drawdown: {data['Max Drawdown (%)']}%", style={'color': '#E0E0E0', 'margin': '5px 0'}),
                html.P(f"Current Price: ₹{data['Current Price']}", style={'color': '#E0E0E0', 'margin': '5px 0'})
            ], style={
                'background-color': '#2F2F2F',
                'padding': '15px',
                'border-radius': '8px',
                'margin': '10px',
                'border-left': f'4px solid {card_color}',
                'display': 'inline-block',
                'vertical-align': 'top',
                'width': '200px'
            })
            summary_cards.append(card)
        
        return html.Div([
            html.H3(f"Stock Comparison - {len(multi_selected_tickers)} Stocks", style={'color': '#E0E0E0'}),
            html.P(f"Analysis Period: {comparison_data.index[0].strftime('%Y-%m-%d')} to {comparison_data.index[-1].strftime('%Y-%m-%d')}", 
                   style={'color': '#B0B0B0'}),
            
            html.Div([
                html.H4("Performance Summary", style={'color': '#E0E0E0', 'margin-bottom': '15px'}),
                html.Div(summary_cards)
            ], style={'margin-bottom': '30px'}),
            
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3)
        ], style={'padding': '20px'})
    
    data = fetch_stock_data(selected_ticker, start_date, end_date)
    
    if data is None or data.empty:
        return html.Div([
            html.H3("Error loading data", style={'color': '#FF6B6B'}),
            html.P(f"Could not fetch data for {selected_ticker}. Please try a different ticker or date range.", 
                   style={'color': '#E0E0E0'}),
            html.P("Note: Make sure the ticker symbol is correct and has sufficient historical data.",
                   style={'color': '#E0E0E0'})
        ], style={'padding': '20px'})
    
    if tab == 'tab-1':
        bbox = data.copy()
        
        if len(bbox) < 200:
            return html.Div([
                html.H3("Insufficient Data", style={'color': '#FF6B6B'}),
                html.P(f"Need at least 200 data points for 200-day moving average. Current data has {len(bbox)} points.",
                       style={'color': '#E0E0E0'}),
                html.P("Try selecting a longer date range or a different stock.",
                       style={'color': '#E0E0E0'})
            ], style={'padding': '20px'})
        
        bbox['MA 50'] = bbox[selected_ticker].rolling(window=50).mean()
        bbox['MA 200'] = bbox[selected_ticker].rolling(window=200).mean()
        
        fig1 = go.Figure()
        columns = [selected_ticker, 'MA 50', 'MA 200']
        colors = ['#00D4FF', '#FFB347', '#FF6B6B']
        
        for i, column in enumerate(columns):
            if column in bbox.columns and not bbox[column].isna().all():
                fig1.add_trace(go.Scatter(
                    x=bbox.index,
                    y=bbox[column],
                    mode='lines',
                    name=column,
                    line=dict(color=colors[i], width=2)
                ))
        
        fig1.update_layout(
            title=f'{selected_ticker} - Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            paper_bgcolor='#1F1F1F',
            plot_bgcolor='#2F2F2F',
            font=dict(color='#E0E0E0')
        )
        
        bbox['middle'] = bbox[selected_ticker].rolling(window=20).mean()
        bbox['upper'] = bbox['middle'] + 2 * (bbox[selected_ticker].rolling(window=20).std())
        bbox['lower'] = bbox['middle'] - 2 * (bbox[selected_ticker].rolling(window=20).std())
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=bbox.index, y=bbox['upper'],
            mode='lines', name='Upper Band',
            line=dict(color='#FF6B6B', dash='dash', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=bbox.index, y=bbox['lower'],
            mode='lines', name='Lower Band',
            line=dict(color='#FF6B6B', dash='dash', width=2),
            fill='tonexty', fillcolor='rgba(255,107,107,0.1)'
        ))
        fig2.add_trace(go.Scatter(
            x=bbox.index, y=bbox['middle'],
            mode='lines', name='Middle Band (20 MA)',
            line=dict(color='#FFB347', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=bbox.index, y=bbox[selected_ticker],
            mode='lines', name=selected_ticker,
            line=dict(color='#00D4FF', width=2)
        ))
        
        fig2.update_layout(
            title=f'{selected_ticker} - Bollinger Bands',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            paper_bgcolor='#1F1F1F',
            plot_bgcolor='#2F2F2F',
            font=dict(color='#E0E0E0')
        )
        
        return html.Div([
            html.H4(f"Technical Analysis - {selected_ticker}", style={'color': '#E0E0E0'}),
            html.P(f"Data points: {len(bbox)} | Date range: {bbox.index[0].strftime('%Y-%m-%d')} to {bbox.index[-1].strftime('%Y-%m-%d')}",
                   style={'color': '#B0B0B0'}),
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)
        ], style={'padding': '20px'})
    
    elif tab == 'tab-2':
        bbox = data.copy()
        returns = bbox[selected_ticker].pct_change().dropna()
        bbox['volatility'] = returns.rolling(window=30).std() * np.sqrt(252)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bbox.index,
            y=bbox['volatility'],
            mode='lines',
            name='30-Day Annualized Volatility',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        fig.update_layout(
            title=f'{selected_ticker} - Rolling Volatility (30-Day)',
            xaxis_title='Date',
            yaxis_title='Volatility',
            hovermode='x unified',
            template='plotly_dark',
            height=600,
            paper_bgcolor='#1F1F1F',
            plot_bgcolor='#2F2F2F',
            font=dict(color='#E0E0E0')
        )
        
        return html.Div([
            dcc.Graph(figure=fig)
        ], style={'padding': '20px'})
    
    elif tab == 'tab-3':
        bbox = data.copy()
        returns = bbox[selected_ticker].pct_change().dropna()
        
        if len(returns) < 100:
            return html.Div([
                html.H3("Insufficient Data", style={'color': '#FF6B6B'}),
                html.P("GARCH modeling requires at least 100 data points.", style={'color': '#E0E0E0'})
            ], style={'padding': '20px'})
        
        try:
            results, p, q = fit_garch_model(returns)
            volatility = results.conditional_volatility
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=(f"Returns (%) - {selected_ticker}", 
                                            f"GARCH({p},{q}) Conditional Volatility"),
                              vertical_spacing=0.1)
            
            fig.add_trace(go.Scatter(
                x=returns.index, y=returns * 100,
                name="Returns (%)", 
                line=dict(color='#00D4FF', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=volatility.index, y=volatility,
                name=f"GARCH({p},{q}) Volatility",
                line=dict(color='#FF6B6B', width=2)
            ), row=2, col=1)
            
            fig.update_layout(
                title=f"GARCH({p},{q}) Model Results - {selected_ticker}",
                height=700,
                showlegend=True,
                template='plotly_dark',
                paper_bgcolor='#1F1F1F',
                plot_bgcolor='#2F2F2F',
                font=dict(color='#E0E0E0')
            )
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Return (%)", row=1, col=1)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            
            avg_volatility = volatility.mean()
            max_volatility = volatility.max()
            current_volatility = volatility.iloc[-1]
            
            return html.Div([
                html.H4(f"GARCH Model Summary - {selected_ticker}", style={'color': '#E0E0E0'}),
                html.P(f"Model: GARCH({p},{q})", style={'color': '#B0B0B0'}),
                html.P(f"Average Volatility: {avg_volatility:.2f}%", style={'color': '#B0B0B0'}),
                html.P(f"Maximum Volatility: {max_volatility:.2f}%", style={'color': '#B0B0B0'}),
                html.P(f"Current Volatility: {current_volatility:.2f}%", style={'color': '#B0B0B0'}),
                dcc.Graph(figure=fig)
            ], style={'padding': '20px'})
            
        except Exception as e:
            return html.Div([
                html.H3("GARCH Modeling Error", style={'color': '#FF6B6B'}),
                html.P(f"Error fitting GARCH model: {str(e)}", style={'color': '#E0E0E0'}),
                html.P("This can happen due to:", style={'color': '#E0E0E0'}),
                html.Ul([
                    html.Li("Insufficient data points", style={'color': '#B0B0B0'}),
                    html.Li("Extreme volatility in the data", style={'color': '#B0B0B0'}),
                    html.Li("Model convergence issues", style={'color': '#B0B0B0'}),
                    html.Li("Try selecting a different stock or longer time period", style={'color': '#B0B0B0'})
                ])
            ], style={'padding': '20px'})
    
    elif tab == 'tab-4':
        bbox = data.copy()
        price_data = bbox[selected_ticker].dropna()
        
        if len(price_data) < 50:
            return html.Div([
                html.H3("Insufficient Data", style={'color': '#FF6B6B'}),
                html.P("ARIMA forecasting requires at least 50 data points.", style={'color': '#E0E0E0'})
            ], style={'padding': '20px'})
        
        try:
            model = ARIMA(price_data, order=(1, 1, 1))
            results = model.fit()
            
            n_steps = 30
            forecast = results.get_forecast(steps=n_steps)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()
            
            future_dates = pd.date_range(
                start=price_data.index[-1], 
                periods=n_steps + 1, 
                freq='B'
            )[1:]
            
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=price_data.index, y=price_data,
                mode='lines', name='Actual Price',
                line=dict(color='#00D4FF', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates, y=forecast_mean,
                mode='lines', name='Forecast Price',
                line=dict(color='#FFB347', dash='dash', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates, y=forecast_ci.iloc[:, 0],
                mode='lines', name='Lower CI',
                line=dict(color='#FF6B6B', dash='dot', width=1),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=forecast_ci.iloc[:, 1],
                mode='lines', name='Upper CI',
                line=dict(color='#FF6B6B', dash='dot', width=1),
                fill='tonexty', fillcolor='rgba(255,107,107,0.2)',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f'ARIMA(1,1,1) Price Forecast - {selected_ticker}',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                height=600,
                paper_bgcolor='#1F1F1F',
                plot_bgcolor='#2F2F2F',
                font=dict(color='#E0E0E0'),
                hovermode='x unified'
            )
            
            return html.Div([
                html.H4(f"ARIMA Forecast Summary - {selected_ticker}", style={'color': '#E0E0E0'}),
                html.P(f"Forecasting next {n_steps} business days", style={'color': '#B0B0B0'}),
                html.P(f"Current Price: ₹{price_data.iloc[-1]:.2f}", style={'color': '#B0B0B0'}),
                html.P(f"Forecast Price (30 days): ₹{forecast_mean.iloc[-1]:.2f}", style={'color': '#B0B0B0'}),
                html.P(f"Expected Change: {((forecast_mean.iloc[-1] / price_data.iloc[-1]) - 1) * 100:.2f}%", 
                       style={'color': '#B0B0B0'}),
                dcc.Graph(figure=fig)
            ], style={'padding': '20px'})
            
        except Exception as e:
            return html.Div([
                html.H3("ARIMA Forecasting Error", style={'color': '#FF6B6B'}),
                html.P(f"Error fitting ARIMA model: {str(e)}", style={'color': '#E0E0E0'})
            ], style={'padding': '20px'})

@callback(
    Output('multi-stock-dropdown', 'value'),
    [Input('multi-stock-dropdown', 'value')]
)
def limit_stock_selection(selected_stocks):
    if selected_stocks and len(selected_stocks) > 10:
        return selected_stocks[:10]
    return selected_stocks

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8050)),
        debug=False
    )

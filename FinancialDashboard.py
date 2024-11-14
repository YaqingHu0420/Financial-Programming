import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# set page title
st.set_page_config(page_title="Financial Dashboard – S&P 500 Stock")

# Set title
st.title("📈 Financial Dashboard")

# connect to the S&P 500 dataset
@st.cache_data
def load_data():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]  # First table contains the S&P 500 companies list

# Fetch the S&P 500 stock list
df = load_data()
stock_name = df['Symbol'].tolist()

# Select stock
select_stock = st.selectbox("Select Stock", stock_name, help="Choose a stock from the S&P 500 list.")
stock = yf.Ticker(select_stock)

# Get the overall stock info
stock_info = stock.info

# Define the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo simulation", "Analysis"])

# Update button
with tab1:
        company_name = stock_info.get('longName', select_stock)
        st.markdown(f"##### {company_name} Summary")
        stock_data = stock.history(period="max", interval = '1d')

        #Plot line chart for stock prices
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Stock price area plot
        area_plot = go.Scatter(x=stock_data.index, y=stock_data['Close'],
                               mode='lines', name = 'Close Price', line=dict(color='rgba(255, 77, 77, 1)', width=1),
                            fill='tozeroy', fillcolor='rgba(255, 77, 77, 0.2)', showlegend=False)
        fig.add_trace(area_plot, secondary_y=True)

        # Stock volume bar plot
        bar_plot = go.Bar(x=stock_data.index, y=stock_data['Volume'], name = 'Volume',marker_color=np.where(stock_data['Close'].pct_change() < 0, 'red', 'green'),
                        showlegend=False)
        fig.add_trace(bar_plot, secondary_y=False)
        
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(label = "MAX", step="all")
                ])
            )
        )

        fig.update_layout(
            title=f'{select_stock} Stock Close Price and Volume',
            xaxis_title='Date',
            yaxis=dict(
            title="Volume"),
            yaxis2=dict(
            title="Close Price (USD)", 
            overlaying='y', 
            side='right',    # Position on the right side of the chart
            showgrid=False 
        ),
            template="plotly_white"
        )

        st.plotly_chart(fig, theme="streamlit",use_container_width=True)
        
        # Market cap format
        def format_market_cap(market_cap):
            if market_cap == '--':
                return '--'
            elif market_cap >= 1000000000:
                return f"{market_cap / 1000000000:.2f}B"
            elif market_cap >= 1000000:
                return f"{market_cap / 1000000:.2f}M"
            elif market_cap >= 1000:
                return f"{market_cap / 1000:.2f}K"
            else:
                return f"{market_cap:.2f}"
        
        # Datetime format
        def format_date(timestamp):
            return datetime.fromtimestamp(timestamp).strftime('%b %d, %Y') if timestamp and timestamp != '--' else '--'

        table_data = {
            "Detail 1": ["Previous Close", "Open", "Bid", "Ask", "Day's Range", "52 Week Range"],
            "Info 1": [
                stock_info.get('previousClose', '--'),
                stock_info.get('open', '--'),
                f"{stock_info.get('bid')} x {stock_info.get('bidSize')}",
                f"{stock_info.get('ask')} x {stock_info.get('askSize')}",
                f"{round(stock_info.get('dayLow', '--'), 2)} - {round(stock_info.get('dayHigh', '--'), 2)}",
                f"{round(stock_info.get('fiftyTwoWeekLow', '--'), 2)} - {round(stock_info.get('fiftyTwoWeekHigh', '--'), 2)}"
            ],
            "Detail 2": ["Market Cap (intraday)", "Volume", "Avg. Volume", "Forward Dividend & Yield", "Ex-Dividend Date", ""],
            "Info 2": [
                format_market_cap(stock_info.get('marketCap', '--')),
                f"{stock_info.get('volume', '--'):,}" if stock_info.get('volume', '--') != '--' else '--',
                f"{stock_info.get('averageVolume', '--'):,}" if stock_info.get('averageVolume', '--') != '--' else '--',
                f"${stock_info.get('dividendRate', '--')}  ({stock_info.get('dividendYield', 0):.2%})",
                format_date(stock_info.get('lastDividendDate', '--')),
                ""
            ],
            "Detail 3": ["Shares Out","Beta (5Y Monthly)", "P/E Ratio (TTM)", "EPS (TTM)", "1Y Target Est" , ""],
            "Info 3": [
                format_market_cap(stock_info.get('sharesOutstanding', '--')),
                round(stock_info.get('beta', '--'), 2) if stock_info.get('beta', '--') != '--' else '--',
                round(stock_info.get('trailingPE', '--'), 2) if stock_info.get('trailingPE', '--') != '--' else '--',
                stock_info.get('trailingEps', '--'),
                stock_info.get('targetMeanPrice', '--'),
                ""
            ]
        }

        #key values table
        table = st.dataframe(table_data, use_container_width=True, hide_index=True, column_config={"B": None})

        #Stock overview
        with st.expander(f"{company_name} Overview"):
            st.markdown(f"**{company_name} Description**")
            #st.markdown(f"<h2 style='font-size: 18px;'>{company_name} Description</h2>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3.5, 0.01, 1.49])
            with col1:
                #company description
                description = stock_info.get('longBusinessSummary', "No description available.")
                st.markdown(
                f'<div style="text-align: justify;">{description}</div><br>',
                unsafe_allow_html=True
                )

            with col3:
                #company detail list
                country = stock_info.get('country', '--')
                industry = stock_info.get('industry', '--')
                sector = stock_info.get('sector', '--')
                employees = stock_info.get('fullTimeEmployees', '--')
                ceo = stock_info.get('companyOfficers', [{}])[0].get('name', '--')  # Assumes the first officer is the CEO

                st.write(f"**Country**: {country}")
                st.write(f"**Industry**: {industry}")
                st.write(f"**Sector**: {sector}")
                st.write(f"**Employees**: {employees}")
                st.write(f"**CEO**: {ceo}")

                #stock detail table
                st.markdown(f"**Stock Details**")
                stock_details = {
                    "Detail": [
                        "Ticker Symbol",
                        "Exchange",
                        "Fiscal Year Ends",
                        "Reporting Currency"
                    ],
                    "Info": [
                        stock_info.get('symbol', '--'),
                        stock_info.get('exchange', '--'),
                        datetime.fromtimestamp(stock.info.get('lastFiscalYearEnd', 0)).strftime("%b %d"),  
                        stock_info.get('financialCurrency', '--'),  
                    ]
                }

                stock_df = pd.DataFrame(stock_details)
                st.dataframe(stock_df, use_container_width=True, hide_index=True)

            #major holders
            st.markdown(f"**Major Shareholders**")
            holders = stock.institutional_holders
            st.dataframe(holders, use_container_width=True, hide_index=True)

            #contact details table
            st.markdown(f"**Contact Details**")
            contact_details = {
                "Detail": [
                    "Address",
                    "Phone",
                    "Website"
                ],
                "Information": [
                    f"{stock_info.get('address1', 'N/A')}, {stock_info.get('city', 'N/A')}, {stock_info.get('state', 'N/A')} {stock_info.get('zip', 'N/A')}, {stock_info.get('country', 'N/A')}",
                    stock_info.get('phone', 'N/A'),
                    stock_info.get('website', 'N/A')
                ]
            }

            contact_df = pd.DataFrame(contact_details)
            st.dataframe(contact_df, use_container_width=True, hide_index=True)


with tab2:
    st.markdown(f"##### {company_name} Charts ")
    st.markdown("")
    stock_data = stock.history(period="max", interval='1d')
    #Calculate moving average (MA) for the stock price using a window size of 50 days
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    select_plot = st.radio("Select Plot Type", ['Line', 'Candlestick'],index=0, horizontal=True)

    if select_plot == 'Line':
        # Line plot
        plot = go.Scatter(x=stock_data.index, y=stock_data['Close'],
                            mode='lines', name='Close Price', line=dict(color='rgba(133, 133, 241, 1)', width=1),
                            fill='tozeroy', fillcolor='rgba(133, 133, 241, 0.2)', showlegend=False)
    else:
        # Candlestick plot
        plot = go.Candlestick(x=stock_data.index,
                                open=stock_data['Open'],
                                high=stock_data['High'],
                                low=stock_data['Low'],
                                close=stock_data['Close'], name='Candlestick')
        
    fig.add_trace(plot, secondary_y=True)

    # Volume plot
    volume_plot = go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume',
                     marker_color=np.where(stock_data['Close'].pct_change() < 0, 'red', 'green'),
                     showlegend=False)
    fig.add_trace(volume_plot, secondary_y=False)

    # MA plot
    sma_plot = go.Scatter(x=stock_data.index, y=stock_data['MA_50'],
                        mode='lines', name='50-Day SMA', line=dict(color='purple', width=1))
    fig.add_trace(sma_plot, secondary_y=True)

    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=3, label="3Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(label="MAX", step="all")
        ])
        ))

    # Update layout and axis labels
    fig.update_layout(
    title=f'{select_stock} Stock Price and Volume',
    xaxis_title='Date',
    yaxis=dict(
        title="Volume",
        showgrid=False
    ),
    yaxis2=dict(
        title="Price (USD)",
        overlaying='y',
        side='right',
        showgrid=False
    ),
    legend=dict(
        x=1,     
        y=1.15,      
        xanchor="right",
        yanchor="top",
        orientation="h"
    ),
    hovermode="closest",
    template="plotly_white")

    # Display the plot
    st.plotly_chart(fig, theme="streamlit", width=800, height=600)


with tab3:
    st.markdown(f"##### {company_name} Financial Data")
    st.markdown("")
    financial_data = {
    "Income Statement": {
        "Annual": stock.financials.fillna("--") if stock.financials is not None else pd.DataFrame(),
        "Quarterly": stock.quarterly_financials.fillna("--") if stock.quarterly_financials is not None else pd.DataFrame()},
    "Balance Sheet": {
        "Annual": stock.balance_sheet.fillna("--") if stock.balance_sheet is not None else pd.DataFrame(),
        "Quarterly": stock.quarterly_balance_sheet.fillna("--") if stock.quarterly_balance_sheet is not None else pd.DataFrame()},
    "Cash Flow": {
        "Annual": stock.cashflow.fillna("--") if stock.cashflow is not None else pd.DataFrame(),
        "Quarterly": stock.quarterly_cashflow.fillna("--") if stock.quarterly_cashflow is not None else pd.DataFrame()}
    }

    report_type = st.radio("Select Report Type", ("Income Statement", "Balance Sheet", "Cash Flow"), index=0, horizontal=True) 
    period_type = st.radio("Select Period", ("Annual", "Quarterly"), index=0, horizontal=True)

    selected_data = financial_data[report_type][period_type]
    st.write(f"### {report_type} ({period_type})")
    st.dataframe(selected_data)


with tab4:
    st.markdown(f"##### Monte Carlo simulation")
    n_simulation = st.selectbox("Number of Simulations", [200, 500, 1000], help="Choose the number of simulations.")
    time_horizon = st.selectbox("Time Horizon (days)", [30, 60, 90], help="Choose the time horizon.")

    def run_simulation(stock_price, time_horizon, n_simulation, seed):

        # Daily return (of close price)
        daily_return = stock_data['Close'].pct_change()
        # Daily mean (of close price)
        daily_mean = daily_return.mean()
        # Daily volatility (of close price)
        daily_volatility = np.std(daily_return)

        # Run the simulation
        np.random.seed(seed)
        simulation_df = pd.DataFrame()  # Initiate the data frame

        for i in range(n_simulation):

            # The list to store the next stock price
            next_price = []

            # Create the next stock price
            last_price = stock_data['Close'].iloc[-1]

            for j in range(time_horizon):

                # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(daily_mean, daily_volatility)

                # Generate the random future price
                future_price = last_price * (1 + future_return)

                # Save the price and go next
                next_price.append(future_price)
                last_price = future_price

            # Store the result of the simulation
            next_price_df = pd.Series(next_price).rename('sim' + str(i))
            simulation_df = pd.concat([simulation_df, next_price_df], axis=1)

        return simulation_df
        
    simulation_df = run_simulation(stock_price=stock_data, time_horizon=time_horizon, n_simulation=n_simulation, seed=123)
            
    def plot_simulation_price(stock_price, simulation_df):
            
        # Plot the simulation stock price in the future
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(simulation_df)
        ax.set_title('Monte Carlo simulation for the stock price in next ' + str(simulation_df.shape[0]) + ' days', fontsize = 14)
        ax.set_xlabel('Day', fontsize = 14)
        ax.set_ylabel('Price', fontsize = 14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.axhline(y=stock_price['Close'].iloc[-1], color='red')
        ax.legend(['Current stock price is: ' + str(np.round(stock_price['Close'].iloc[-1], 2))])
        ax.get_legend().legend_handles[0].set_color('red')
        st.pyplot(fig)
            
    # Test the function
    plot_simulation_price(stock_data, simulation_df)


    st.markdown(f"##### Value at Risk")

    def value_at_risk(stock_price, simulation_df):

        # Price at 95% confidence interval
        future_price_95ci = np.percentile(simulation_df.iloc[-1:, :].values[0, ], 5)

        # Value at Risk
        VaR = stock_price['Close'].iloc[-1] - future_price_95ci
        st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')
                
    # Test the function
    value_at_risk(stock_data, simulation_df)

    def plot_simulation_hist(stock_price, simulation_df):
    
        # Get the simulated prices at the last day
        ending_price = simulation_df.iloc[-1:, :].values[0, ]

        # Plot using histogram
        fig, ax = plt.subplots()
        ax.hist(ending_price, bins=50)
        ax.axvline(x=stock_price['Close'].iloc[-1], color='red')
        ax.legend(['Current stock price is: ' + str(np.round(stock_price['Close'].iloc[-1], 2))], fontsize=6)
        ax.get_legend().legend_handles[0].set_color('red')
        ax.set_xlabel('Price', fontsize = 6)
        ax.set_ylabel('Frequency', fontsize = 6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        st.pyplot(fig)
            
    # Test the function
    plot_simulation_hist(stock_data, simulation_df)


with tab5:
    select_analytics = st.selectbox("Select an analytics type", 
    ["Statistics", "Dividends", "Comparison", "News"],index=0)
    st.markdown("")

    if select_analytics == 'Statistics':
        st.session_state['active_section'] = "Statistics"
        st.markdown(f"##### {company_name} Statistics ")
        #calculation
        market_cap = stock_info.get("marketCap", 0)
        total_debt = stock_info.get("totalDebt", 0)
        cash = stock_info.get("cash", 0)
        enterprise_value = market_cap + total_debt - cash

        # Calculating EBIT and FCF
        ebit = stock_info.get("ebitda") - stock_info.get("interestExpense", 0) 
        fcf = stock_info.get("freeCashflow")

        col1, col2, col3, col4, col5 = st.columns([2, 0.1, 2, 0.1, 2.2])
        with col1:
            st.markdown(f"**Total Valuation**")
            # Total_Valuation
            st.write(
            f"{select_stock} Company has a market cap or net worth of {format_market_cap(stock_info.get('marketCap', '--'))}. The enterprise value is {format_market_cap(stock_info.get('enterpriseValue', '--'))}.")
            Total_Valuation = {"Detail": ["Market Cap", "Enterprise Value", "", "", ""],
                "Info": [
                    format_market_cap(stock_info.get('marketCap', '--')),
                    format_market_cap(stock_info.get("enterpriseValue", '--')), "", "", ""
                ]}
            valuation_df = st.dataframe(Total_Valuation, use_container_width=True, hide_index=True)

            # Valuation_Ratios
            st.markdown(f"**Valuation Ratios**")
            st.write(
            f"The trailing PE ratio is {round(stock_info.get('trailingPE'),2)} and the forward PE ratio is {round(stock_info.get('forwardPE'),2)}. ")
            Valuation_Ratios= {
            "Detail": ["PE Ratio", "Forward PE", "PS Ratio", "PB Ratio", ""],
                "Info": [
                    round(stock_info.get("trailingPE"),2),
                    round(stock_info.get("forwardPE"),2),
                    round(stock_info.get("priceToSalesTrailing12Months"),2),
                    round(stock_info.get("priceToBook"),2), ""
                ]}
            ratios_df = st.dataframe(Valuation_Ratios, use_container_width=True, hide_index=True)

        with col3:
            # Dividends & Yields
            st.markdown(f"**Dividends & Yields**")
            dividends = stock.dividends
            st.write(
                f"This stock pays an annual dividend of ${round(stock_info.get('dividendRate', 0))}, which amounts to a dividend yield of {stock_info.get('dividendYield', 0):.2%}.")
            DividendsYields = {
            "Detail": ["Dividend Per Share", "Dividend Yield", "Payout Ratio", "", ""],
                "Info": [
                    round(stock_info.get("dividendRate", 'None'),2),
                    round(stock_info.get("dividendYield"),2),
                    round(stock_info.get("payoutRatio"),2), "", ""
                ]}
            DividendsYields_df = st.dataframe(DividendsYields, use_container_width=True, hide_index=True)

            # Enterprise_Valuation
            st.markdown(f"**Enterprise Valuation**")
            st.write(
            f"The stock's EV/EBITDA ratio is {round(enterprise_value / stock_info.get("ebitda"),2)}, with an EV/FCF ratio of {round(enterprise_value / fcf,2)}.")
            Enterprise_Valuation = {
            "Detail": ["EV/Earnings", "EV/Sales", "EV/EBITDA", "EV/EBIT", "EV/FCF"],
                "Info": [
                    round(enterprise_value / stock_info.get("netIncomeToCommon"),2) if stock_info.get("netIncomeToCommon") else None,
                    round(enterprise_value / stock_info.get("totalRevenue"),2) if stock_info.get("totalRevenue") else None,
                    round(enterprise_value / stock_info.get("ebitda"),2) if stock_info.get("ebitda") else None,
                    round(enterprise_value / ebit,2) if ebit else None,
                    round(enterprise_value / fcf,2) if fcf else None
                ]} 
            Enterprise_Valuation_df = st.dataframe(Enterprise_Valuation, use_container_width=True, hide_index=True)

        with col5:
            # Stock_Price_Statistics
            st.markdown(f"**Stock Price Statistics**")
            percent_change = round(stock_info.get('52WeekChange', 0) * 100, 2)
            if percent_change < 0:
                change_text = f"decreased by {abs(percent_change)}%"
            else:
                change_text = f"increased by +{percent_change}%"

            st.write(f"The stock price has {change_text} in the last 52 weeks. The beta is {round(stock_info.get('beta'), 2)}.")
            Stock_Price_Statistics = {
            "Detail": [
                    "Beta (5Y)",
                    "52-Week Range",
                    "50-Day Moving Average",
                    "200-Day Moving Average",
                    "Average Volume (20 Days)"
                ],
                "Info": [
                    round(stock_info.get("beta"),2),
                    f"{round(stock_info.get('52WeekChange', 0) * 100, 2):.2f}%",
                    round(stock_info.get("fiftyDayAverage"),2),
                    round(stock_info.get("twoHundredDayAverage"),2),
                    format_market_cap(stock_info.get("averageVolume"))
                ]}
            Stock_Price_Statistics_df = st.dataframe(Stock_Price_Statistics, use_container_width=True, hide_index=True)

            # Financial_Position
            st.markdown(f"**Financial Position**")
            st.write(
            f"The company has a current ratio of {round(stock_info.get("currentRatio"), 2)}, with a Debt / Equity ratio of {round(stock_info.get("debtToEquity"),2)}.")
            Financial_Position = {
            "Detail": ["Current Ratio", "Quick Ratio", "Debt/Equity", "Debt/EBITDA", "Debt/FCF"],
                "Info": [
                    round(stock_info.get("currentRatio"),2),
                    round(stock_info.get("quickRatio"),2),
                    round(stock_info.get("debtToEquity"),2),
                    round(total_debt / stock_info.get("ebitda")/100,2) if stock_info.get("ebitda") else None,
                    round(total_debt / fcf,2) if fcf else None
                ]}
            financial_position_df = st.dataframe(Financial_Position, use_container_width=True, hide_index=True)
    
    elif select_analytics == 'Dividends':
        st.markdown(f"##### Dividend history data")
        dividends = stock.dividends
        col1, col2, col3 = st.columns([2, 0.2, 2.2])
        with col1:
            st.write(
                f"This stock pays an annual dividend of ${round(stock_info.get('dividendRate', 0))}, which amounts to a dividend yield of {stock_info.get('dividendYield', 0):.2%}.")
            DividendsYields = {
            "Detail": ["Dividend Per Share", "Dividend Yield", "Payout Ratio"],
                "Info": [
                    round(stock_info.get("dividendRate"),2),
                    round(stock_info.get("dividendYield"),2),
                    round(stock_info.get("payoutRatio"),2)]}
            DividendsYields_df = st.dataframe(DividendsYields, use_container_width=True, hide_index=True)

        with col3:
            st.dataframe(dividends)
        
        st.markdown(f"##### Dividend charts")
        fig = go.Figure()
        fig.add_scatter(x=dividends.index, y=dividends.values, mode='lines+markers')

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=5, label="5Y", step="year", stepmode="backward"),  # 5 years
                    dict(count=10, label="10Y", step="year", stepmode="backward"), # 10 years
                    dict(label="MAX", step="all")  # Show all available data
                ])
            ))

        # Add title and labels
        fig.update_layout(
            title=f'{select_stock} Dividend History',
            title_x=0.5, 
            title_xanchor='center', 
            xaxis_title="Date",
            yaxis_title="Dividend (USD)",
            template="plotly_dark" )
        st.plotly_chart(fig)

        dividend_growth = dividends.pct_change() * 100
        fig_growth = go.Figure()
        fig_growth.add_scatter(x=dividend_growth.index, y=dividend_growth.values, mode='lines+markers')

        fig_growth.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=5, label="5Y", step="year", stepmode="backward"),  # 5 years
                    dict(count=10, label="10Y", step="year", stepmode="backward"), # 10 years
                    dict(label="MAX", step="all")  # Show all available data
                ])
            )
        )

        fig_growth.update_layout(
            title=f'{select_stock} Dividend Growth',
            title_x=0.5, 
            title_xanchor='center', 
            xaxis_title="Date",
            yaxis_title="Growth (%)",
            yaxis=dict(
            range=[-100, 100]),
            template="plotly_dark"
        )

        # Show the Dividend Growth Chart
        st.plotly_chart(fig_growth)

    elif select_analytics == "Comparison":
        st.markdown(f"##### Compare Stocks ")  
        # Select stock
        selected_stocks = st.multiselect("Select two stocks for comparison", stock_name, default=stock_name[:2])
        st.markdown("")

        stock1_info = yf.Ticker(selected_stocks[0]).info
        stock2_info = yf.Ticker(selected_stocks[1]).info
        stock1 = yf.Ticker(selected_stocks[0]).history(period="max", interval = '1d')
        stock2 = yf.Ticker(selected_stocks[1]).history(period="max", interval = '1d')

        st.write("###### Stock Information Comparison")
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Profile", "Dividends", "Financials & Technicals"])

        with tab1:
            #overview comparison
            comparison_data1 = {
            "Symbol": [selected_stocks[0], selected_stocks[1]],
            "Company Name": [stock1_info.get("longName", "--"), stock2_info.get("longName", "--")],
            "Market Cap": [format_market_cap(stock1_info.get('marketCap', '--')) if stock1_info.get('marketCap') else "--",
                        format_market_cap(stock2_info.get('marketCap', '--')) if stock2_info.get('marketCap') else "--"],
            "Stock Price": [stock1_info.get("previousClose", "--"), stock2_info.get("previousClose", "--")],
            "52W High": [round(stock1_info.get("fiftyTwoWeekHigh", "--"),2), round(stock2_info.get("fiftyTwoWeekHigh", "--"),2)],
            "52W Low": [round(stock1_info.get("fiftyTwoWeekLow", "--"),2), round(stock2_info.get("fiftyTwoWeekLow", "--"),2)]}
        
            comparison_df1 = pd.DataFrame(comparison_data1)
            st.dataframe(comparison_df1, use_container_width=True, hide_index=True)

        with tab2:
            #profile comparison
            comparison_data2 = {
            "Symbol": [selected_stocks[0], selected_stocks[1]],
            "Exchange": [stock1_info.get('exchange', '--'), stock2_info.get('exchange', '--')],
            "Currency": [stock1_info.get('currency', '--'), stock2_info.get('exchange', '--')],
            "Sector": [stock1_info.get("sector", "--"), stock2_info.get("sector", "--")],
            "Industry": [stock1_info.get("industry", "--"), stock2_info.get("industry", "--")],
            "Country": [stock1_info.get("country", "--"), stock2_info.get("country", "--")],
            "Employees": [stock1_info.get('fullTimeEmployees', '--'),stock2_info.get('fullTimeEmployees', '--')]}
        
            comparison_df2 = pd.DataFrame(comparison_data2)
            st.dataframe(comparison_df2, use_container_width=True, hide_index=True)

        with tab3:
            # dividend comparison
            comparison_data3 = {
            "Symbol": [selected_stocks[0], selected_stocks[1]],
            "Div.($)": [f"${stock1_info.get('dividendRate', '--'):.2f}" if stock1_info.get('dividendRate') else "--",
                        f"${stock2_info.get('dividendRate', '--'):.2f}" if stock2_info.get('dividendRate') else "--"],
            "Div. Yield": [f"{stock1_info.get("dividendYield", '--')*100:.2f}%" if stock1_info.get('dividendYield') else "--",
                        f"{stock2_info.get('dividendYield', '--')*100:.2f}%" if stock2_info.get('dividendYield') else "--"],
            "Ex-Div Date": [format_date(stock1_info.get("exDividendDate", '--')),format_date(stock2_info.get('exDividendDate', '--'))],
            "Payout Ratio": [f"{stock1_info.get("payoutRatio", '--')*100:.2f}%",f"{stock2_info.get('payoutRatio', '--')*100:.2f}%"],
            "5Y Avg Div. Yeld": [f"{stock1_info.get("fiveYearAvgDividendYield", '--'):.2f}%" if stock1_info.get('fiveYearAvgDividendYield') else "--",
                        f"{stock2_info.get('fiveYearAvgDividendYield', '--'):.2f}%" if stock2_info.get('fiveYearAvgDividendYield') else "--"]}
        
            comparison_df3 = pd.DataFrame(comparison_data3)
            st.dataframe(comparison_df3, use_container_width=True, hide_index=True)

        with tab4:
            # financial&technicals comparison
            comparison_data4 = {
            "Symbol": [selected_stocks[0], selected_stocks[1]],
            "Revenue": [format_market_cap(stock1_info.get("totalRevenue", "--")), format_market_cap(stock2_info.get("totalRevenue", "--"))],
            "Net Income": [format_market_cap(stock1_info.get("netIncomeToCommon", "--")), format_market_cap(stock2_info.get("netIncomeToCommon", "--"))],
            "EPS": [stock1_info.get("trailingEps", "--"), stock2_info.get("trailingEps", "--")],
            "50 MA": [round(stock1_info.get("fiftyDayAverage", "--"),2), round(stock2_info.get("fiftyDayAverage", "--"),2)],
            "200 MA": [round(stock1_info.get('twoHundredDayAverage', '--'),2),round(stock2_info.get('twoHundredDayAverage', '--'),2)],
            "Beta(5Y)": [round(stock1_info.get('beta', '--'),2),round(stock2_info.get('beta', '--'),2)]}
        
            comparison_df4 = pd.DataFrame(comparison_data4)
            st.dataframe(comparison_df4, use_container_width=True, hide_index=True)
        
        # Close price comparision
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock1.index,
            y=stock1['Close'],
            mode='lines',
            name=selected_stocks[0]))

        fig.add_trace(go.Scatter(
            x=stock2.index,
            y=stock2['Close'],
            mode='lines',
            name=selected_stocks[1]
        ))

        fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(label = "MAX", step="all")
                    ])
                )
            )

        fig.update_layout(
            title=f"Closing Price Comparison: {selected_stocks[0]} vs {selected_stocks[1]}",
            xaxis_title="Date",
            yaxis_title="Close Price (USD)",
            legend_title="Stocks",
            template="plotly_dark"
        )

        # Display the chart
        st.plotly_chart(fig)

        # Trading volume comparison
        stock_data1 = yf.Ticker(selected_stocks[0]).history(period="1Y", interval = '1d')
        stock_data2 = yf.Ticker(selected_stocks[1]).history(period="1Y", interval = '1d')
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
                x=stock_data1.index,
                y=stock_data1['Volume'],
                name=selected_stocks[0],
                marker_color='blue'
            ))

        fig1.add_trace(go.Bar(
            x=stock_data2.index,
            y=stock_data2['Volume'],
            name=selected_stocks[1],
            marker_color='red'
        ))

        fig1.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=5, label="5D", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward")
                    ])
                )
            )

        fig1.update_layout(
            title=f"Daily Trading Volume Comparison: {selected_stocks[0]} vs {selected_stocks[1]}",
            xaxis_title="Date",
            yaxis_title="Volume",
            barmode='group',  # Group bars side by side
            template="plotly_white"
        )

        # Display the chart
        st.plotly_chart(fig1)

    elif select_analytics == "News":
        news = stock.news
        st.markdown(f"##### 📰 Latest News for {select_stock}")
        for article in news:
            st.markdown(f"**{article['title']}**")
            st.markdown(f"[Read more]({article['link']})")
            st.markdown("---") 
    


















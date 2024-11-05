import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo simulation", "My own analysis"])

# Update button

with tab1:
        company_name = stock_info.get('longName', select_stock)
        st.markdown(f"##### {company_name} Summary")
        stock_data = stock.history(period="max", interval = '1d')

        #Plot line chart for stock prices
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Stock price area plot
        area_plot = go.Scatter(x=stock_data.index, y=stock_data['Close'],
                               mode='lines', name = 'Close Price', line=dict(color='rgba(133, 133, 241, 1)', width=1),
                            fill='tozeroy', fillcolor='rgba(133, 133, 241, 0.2)', showlegend=False)
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
            title=f'{select_stock} Stock Close Prices',
            xaxis_title='Date',
            yaxis_title='Volume',
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
                return f"{market_cap / 1000000000:.3f}B"
            elif market_cap >= 1000000:
                return f"{market_cap / 1000000:.3f}M"
            elif market_cap >= 1000:
                return f"{market_cap / 1000:.3f}K"
            else:
                return f"{market_cap:.3f}"
        
        # Datetime format
        def format_date(timestamp):
            return datetime.fromtimestamp(timestamp).strftime('%b %d, %Y') if timestamp and timestamp != '--' else '--'



        table_data = {
            "Detail 1": ["Previous Close", "Open", "Bid", "Ask", "Day's Range", "52 Week Range"],
            "Info 1": [
                stock_info.get('previousClose', '--'),
                stock_info.get('open', '--'),
                stock_info.get('bid', '--'),
                stock_info.get('ask', '--'),
                f"{round(stock_info.get('dayLow', '--'), 2)} - {round(stock_info.get('dayHigh', '--'), 2)}",
                f"{round(stock_info.get('fiftyTwoWeekLow', '--'), 2)} - {round(stock_info.get('fiftyTwoWeekHigh', '--'), 2)}"
            ],
            "Detail 2": ["Market Cap (intraday)", "Volume", "Avg. Volume", "Earnings Date", "1Y Target Est", ""],
            "Info 2": [
                format_market_cap(stock_info.get('marketCap', '--')),
                f"{stock_info.get('volume', '--'):,}" if stock_info.get('volume', '--') != '--' else '--',
                f"{stock_info.get('averageVolume', '--'):,}" if stock_info.get('averageVolume', '--') != '--' else '--',
                stock_info.get('earnings_dates', '--'),
                stock_info.get('targetMeanPrice', '--'),
                ""
            ],
            "Detail 3": ["Beta (5Y Monthly)", "P/E Ratio (TTM)", "EPS (TTM)", "Forward Dividend & Yield", "Ex-Dividend Date", ""],
            "Info 3": [
                round(stock_info.get('beta', '--'), 2) if stock_info.get('beta', '--') != '--' else '--',
                round(stock_info.get('trailingPE', '--'), 2) if stock_info.get('trailingPE', '--') != '--' else '--',
                stock_info.get('trailingEps', '--'),
                f"{stock_info.get('dividendRate', '--')} ({stock_info.get('dividendYield', 0):.2%})",
                format_date(stock_info.get('exDividendDate', '--')),
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
                st.write(description)

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
    st.markdown(f"##### {company_name} ({select_stock}) ")


with tab3:
    st.markdown(f"##### {company_name} Financial Data")
    financial_data = {
        "Income Statement": {
            "Annual": stock.financials,
            "Quarterly": stock.quarterly_financials
        },
        "Balance Sheet": {
            "Annual": stock.balance_sheet,
            "Quarterly": stock.quarterly_balance_sheet
        },
        "Cash Flow": {
            "Annual": stock.cashflow,
            "Quarterly": stock.quarterly_cashflow
        }
        }

    report_type = st.radio("", ("Income Statement", "Balance Sheet", "Cash Flow"), index=0, horizontal=True) 
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
    st.markdown(f"##### {company_name} ({select_stock}) ")
















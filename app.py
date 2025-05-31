import streamlit as st
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from concurrent.futures import ThreadPoolExecutor, as_completed
import altair as alt
import re
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
from streamlit_tags import st_tags
import boto3
from io import StringIO


#api key below
api_token = st.secrets['API_KEY']
key_dict = json.loads(st.secrets['textkey'])
# Initialize Firestore if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()


real_sectors = ['XLK', 'XLC', 'XLV', 'XLF', 'XLP', 'XLI', 'XLE', 'XLY', 'XLB', 'XME', 'XLU', 'XLRE']
real_subsectors = ['GDX', 'UFO', 'KBE', 'AMLP', 'ITA', 'ITB', 'IAK', 'SMH', 'PINK', 'XBI', 'NLR', 'FXI', 'WGMI', 'JETS', 'PEJ']

sectors = ['XLK', 'XLC', 'XLV', 'XLF', 'XLP', 'XLI', 'XLE', 'XLY', 'XLB', 'XLU', 'XME', 'XLRE', 'MAGS', 'SPY']
subsectors = ['GDX', 'UFO', 'KBE', 'KRE', 'AMLP', 'ITA', 'ITB', 'IAK', 'SMH', 'PINK', 'XBI', 'NLR', 'BOAT', 'WGMI', 'JETS', 'PEJ', 'QTUM', 'HACK', 'SHLD']
ratecut_etfs = ['AAAU', 'SLV', 'COPX', 'URA', 'CANE', 'XOP', 'UNG', 'WOOD', 'LIT', 'PPLT', 'PALL', 'SLX', 'BNO', 'IBIT', 'SILJ', 'URNJ']
macro_etfs = ['INDA', 'IDX', 'EWM', 'THD', 'EIS', 'FXI', 'ENZL', 'EZA', 'EWY','EWU', 'ARGT', 'EWC', 'EWW', 'UAE', 'EWS', 'GXG', 'EWG', 'EPOL', 'EWD', 'VGK', 'EWO', 'EWP']

# Function to remove rows with any null values
def remove_nulls(df):
    return df.dropna()

def get_previous_business_day(date):
    """
    Adjust the date to the nearest previous business day if it falls on a non-business day (weekend or holiday).
    """
    holidays = USFederalHolidayCalendar().holidays()
    bday_range = pd.bdate_range(date, date, freq='C', holidays=holidays)
    
    while not len(bday_range):
        date -= BDay(1)
        bday_range = pd.bdate_range(date, date, freq='C', holidays=holidays)
    
    return date


# Function to load CSV from S3 and convert to DataFrame
def load_csv_from_s3(s3_client, bucket, file_key):
    """
    Load CSV from S3 bucket and convert it to a pandas DataFrame
    """
    obj = s3_client.get_object(Bucket=bucket, Key=file_key)
    csv_content = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_content))
    return df
    
# Function to calculate consecutive appearances and track the first appearance date
def calculate_consecutive_appearances(df):
    # Extract date columns and sort them in descending order
    date_columns = sorted([col for col in df.columns if col != 'Rank'], reverse=True)
    
    # Get all unique symbols, excluding NaN
    symbols = pd.unique(df[date_columns].values.ravel('K'))
    symbols = [symbol for symbol in symbols if pd.notna(symbol)]
    
    # Initialize a dictionary to store counts and first appearance dates
    consecutive_counts = {}
    first_appearance_dates = {}

    # Iterate over each symbol
    for symbol in symbols:
        count = 0
        first_appearance = None
        for date in date_columns:
            # Check if the symbol appears on this date
            if symbol in df[date].values:
                count += 1
                first_appearance = date  # Track the first date it appears
            else:
                break  # Stop counting when the symbol does not appear on a date
        
        consecutive_counts[symbol] = count
        first_appearance_dates[symbol] = first_appearance if count > 0 else "N/A"

    # Convert the dictionary to a DataFrame
    consecutive_df = pd.DataFrame(list(consecutive_counts.items()), columns=['Symbol', 'consecutive_appearance'])
    consecutive_df['first_appearance_date'] = consecutive_df['Symbol'].map(first_appearance_dates)
    
    return consecutive_df

# Function to calculate the return using the adjusted close of one period before the signal date
def calculate_consecutive_returns(symbol, first_appearance_date, df, api_key):
    # Ensure the historical data is sorted by date in ascending order
    df = df.sort_values(by='date')
    
    # Find the index of the first_appearance_date
    signal_index = df.index[df['date'] == first_appearance_date].tolist()
    
    if signal_index:
        signal_index = signal_index[0]  # Get the index of the first appearance date
        
        # Ensure there's a row before the signal date for the price_at_signal
        if signal_index > 0:
            # Get the adjusted close price of one period before the signal date
            price_at_signal = df.iloc[signal_index - 1]['adjusted_close']
            
            # Fetch the real-time price using the API
            current_price = fetch_real_time_price(symbol, api_key)
            
            if current_price:
                # Calculate the return
                consecutive_day_return = ((current_price - price_at_signal) / price_at_signal) * 100

                # Collect the required data
                return {
                    'symbol': symbol,
                    'date_signaled': first_appearance_date,
                    'price_at_signal': price_at_signal,
                    'current_price': current_price,
                    'consecutive_day_return': consecutive_day_return
                }
    
    # If no valid signal date or no previous price available, return None
    return None

# Main process to calculate returns for consecutive appearances
def process_consecutive_returns(sector_df, subsector_df, api_key):
    # Merge and filter the consecutive appearance data
    combined_df = merge_consecutive_dfs(sector_df, subsector_df)
    
    results = []
    
    # Loop over each symbol and fetch the data
    for idx, row in combined_df.iterrows():
        symbol = row['Symbol']
        first_appearance_date = row['first_appearance_date']
        
        if first_appearance_date != "N/A":
            # Fetch historical data for the symbol
            _, historical_data = fetch_data(symbol, api_key)
            
            if not historical_data.empty:
                # Calculate the return based on the adjusted close one period before the signal date
                result = calculate_consecutive_returns(symbol, first_appearance_date, historical_data, api_key)
                if result:
                    results.append(result)
    
    # Convert results to DataFrame
    return pd.DataFrame(results)
# Merging the sector and subsector dataframes based on Symbol
def merge_consecutive_dfs(sector_df, subsector_df):
    # Filter rows where consecutive appearances > 0
    sector_df = sector_df[sector_df['consecutive_appearance'] > 0]
    subsector_df = subsector_df[subsector_df['consecutive_appearance'] > 0]
    
    # Merge the two dataframes
    combined_df = pd.concat([sector_df, subsector_df]).drop_duplicates(subset='Symbol').reset_index(drop=True)
    return combined_df

# Main process
def process_consecutive_returns(sector_df, subsector_df, api_key):
    # Merge and filter the consecutive appearance data
    combined_df = merge_consecutive_dfs(sector_df, subsector_df)
    
    results = []
    
    # Loop over each symbol and fetch the data
    for idx, row in combined_df.iterrows():
        symbol = row['Symbol']
        first_appearance_date = row['first_appearance_date']
        
        if first_appearance_date != "N/A":
            # Fetch historical data for the symbol
            _, historical_data = fetch_data(symbol, api_key)
            
            if not historical_data.empty:
                # Calculate the return based on the adjusted close one period before the signal date
                result = calculate_consecutive_returns(symbol, first_appearance_date, historical_data, api_key)
                if result:
                    results.append(result)
    
    # Convert results to DataFrame
    return pd.DataFrame(results)


# Firestore data fetching functions
def fetch_bbands_data_from_firestore(sector):
    collection_ref = db.collection('BBands_Results').document(sector).collection('Symbols')
    docs = collection_ref.stream()
    data = []
    for doc in docs:
        doc_dict = doc.to_dict()
        data.append({
            'Symbol': doc_dict.get('Symbol', ''),
            'Crossing Daily Band': doc_dict.get('Crossing Daily Band', ''),
            'Crossing Weekly Band': doc_dict.get('Crossing Weekly Band', ''),
            'Crossing Monthly Band': doc_dict.get('Crossing Monthly Band', '')
        })
    return pd.DataFrame(data)

def fetch_roc_stddev_data_from_firestore(performance_type):
    collection_ref = db.collection('ROCSTDEV_Results').document(performance_type).collection('Top_Symbols')
    docs = collection_ref.stream()
    data = [doc.to_dict() for doc in docs]
    df = pd.DataFrame(data)
    
    # Reorder columns
    df = df[['Symbol', 'ROC/STDDEV', 'RSI', 'Sector']]
    
    # Sort by ROC/STDDEV in descending order
    df = df.sort_values(by='ROC/STDDEV', ascending=False)
    
    return df

def fetch_z_score_data_from_firestore(score_type):
    collection_ref = db.collection('Z_score_results').document(score_type).collection('Records')
    docs = collection_ref.stream()
    data = [doc.to_dict() for doc in docs]


    return pd.DataFrame(data)

# Fetch historical data for a given symbol
def fetch_data(symbol, api_key):
    url = f'https://eodhistoricaldata.com/api/eod/{symbol}.US?api_token={api_key}&period=d&fmt=json'
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            return symbol, pd.DataFrame(data)
        except ValueError as e:
            print(f"Error decoding JSON for {symbol}: {e}")
            return symbol, pd.DataFrame()
    else:
        print(f"Failed to fetch data for {symbol}: {response.status_code}")
        return symbol, pd.DataFrame()

# Fetch real-time data for a given symbol
def fetch_real_time_price(symbol, api_key):
    url = f'https://eodhd.com/api/real-time/{symbol}.US?api_token={api_key}&fmt=json'
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            return data['close']  # Assuming 'close' represents the latest real-time price
        except ValueError as e:
            print(f"Error decoding real-time JSON for {symbol}: {e}")
            return None
    else:
        print(f"Failed to fetch real-time data for {symbol}: {response.status_code}")
        return None


def fetch_current_price(symbol, api_token):
    url = f'https://eodhd.com/api/real-time/{symbol}.US?api_token={api_token}&fmt=json'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current_price = data.get('close', None)
        if current_price is not None:
            current_price = pd.to_numeric(current_price, errors='coerce')
        return current_price
    else:
        print(f"Failed to fetch current price for {symbol}: {response.status_code}, {response.text}")
        return None

def fetch_previous_close_price(symbol, api_token):
    end_date = datetime.now() - BDay(1)
    start_date = end_date - BDay(5)
    url = f'https://eodhistoricaldata.com/api/eod/{symbol}.US?api_token={api_token}&from={start_date.strftime("%Y-%m-%d")}&to={end_date.strftime("%Y-%m-%d")}&fmt=json&adjusted=true'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['adjusted_close'] = pd.to_numeric(df['adjusted_close'], errors='coerce')
        df.dropna(subset=['adjusted_close'], inplace=True)
        previous_close_price = df['adjusted_close'].iloc[-1] if not df.empty else None
        return previous_close_price
    else:
        print(f"Failed to fetch previous close price for {symbol}: {response.status_code}, {response.text}")
        return None

def fetch_historical_data(symbol, api_token, start_date, end_date):
    url = f'https://eodhistoricaldata.com/api/eod/{symbol}.US?api_token={api_token}&from={start_date}&to={end_date}&fmt=json&adjusted=true'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        
        # Check if data is returned and contains the 'date' field
        if data and isinstance(data, list) and 'date' in data[0]:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['adjusted_close'] = pd.to_numeric(df['adjusted_close'], errors='coerce')
            df.dropna(subset=['adjusted_close'], inplace=True)
            return df
        else:
            print("Error: 'date' column is missing in the API response data.")
            return pd.DataFrame()  # Return an empty DataFrame if 'date' is absent
    else:
        print(f"Failed to fetch data for {symbol}: {response.status_code}, {response.text}")
        return pd.DataFrame()

    
def calculate_rolling_correlations(symbols, benchmarks, api_token, rolling_window):
    current_date = datetime.now()
    start_date = (current_date - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = current_date.strftime('%Y-%m-%d')
    
    all_symbols = symbols + benchmarks
    historical_data = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_historical_data, symbol, api_token, start_date, end_date): symbol for symbol in all_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    df.set_index('date', inplace=True)
                    historical_data[symbol] = df['adjusted_close']
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")

    combined_df = pd.DataFrame(historical_data)
    rolling_correlations = combined_df.rolling(window=rolling_window).corr(pairwise=True)
    
    results = {}
    for symbol in symbols:
        results[symbol] = {}
        for benchmark in benchmarks:
            rolling_corr = rolling_correlations.loc[(slice(None), benchmark), symbol].unstack().dropna()
            rolling_corr = rolling_corr.rename(columns={benchmark: f'{benchmark}_correlation'})
            results[symbol][benchmark] = rolling_corr
    
    return results
# Firestore data fetching for correlations (with correct structure: fields as symbols with dictionaries)
def fetch_correlations_from_firestore(ticker, etf_name):
    """
    Fetch the correlation data for a given ticker symbol from Firestore.
    The Firestore document contains fields representing each symbol, 
    and these fields hold the dictionaries for '5 Highest' and '5 Lowest' correlations.
    """
    try:
        # Query the Firestore path: ETF_Correlations -> {ETF Symbol} -> fields for each ticker
        doc_ref = db.collection('ETF_Correlations').document(etf_name)
        doc = doc_ref.get()
        
        # If the document exists, extract the field for the specific ticker symbol
        if doc.exists:
            data = doc.to_dict()
            # The ticker symbol (e.g., AAPL) is a field in the document
            return data.get(ticker, {}).get('Top Correlations', {})
        else:
            st.error(f"No correlation data found for {ticker} in {etf_name}")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Extract the 5 highest and 5 lowest correlations
def extract_top_correlations(correlations):
    """
    Extract the '5 Highest' and '5 Lowest' correlations from the Firestore data.
    """
    if not correlations:
        return pd.DataFrame(), pd.DataFrame()

    # Extract '5 Highest' and '5 Lowest' correlations
    highest_5 = correlations.get('5 Highest', {})
    lowest_5 = correlations.get('5 Lowest', {})

    # Convert to DataFrames for easy display in Streamlit
    highest_5_df = pd.DataFrame(list(highest_5.items()), columns=['Stock', 'Correlation'])
    lowest_5_df = pd.DataFrame(list(lowest_5.items()), columns=['Stock', 'Correlation'])

    return highest_5_df, lowest_5_df

def visualize_rolling_correlations(results):
    charts = []
    for symbol, benchmarks in results.items():
        df_list = []
        for benchmark, df in benchmarks.items():
            df = df.reset_index()
            df_list.append(df[['date', f'{benchmark}_correlation']].rename(columns={f'{benchmark}_correlation': 'correlation'}).assign(benchmark=benchmark))
        
        combined_df = pd.concat(df_list)
        
        highlight = alt.selection_point(fields=['benchmark'], bind='legend')
        
        base = alt.Chart(combined_df).mark_line().encode(
            x='date:T',
            y='correlation:Q',
            color='benchmark:N',
            opacity=alt.condition(highlight, alt.value(1), alt.value(0.2)),
            tooltip=['date:T', 'correlation:Q', 'benchmark:N']
        ).properties(
            title=f'Rolling Correlation of {symbol} with Benchmarks',
            width=800,
            height=400
        ).add_params(
            highlight
        )

        charts.append(base)
    
    final_chart = alt.vconcat(*charts).resolve_scale(
        y='shared'
    )
    
    st.altair_chart(final_chart, use_container_width=True)

def analyze_symbol(symbol, api_token):
    current_date = datetime.now()
    # Adjusting dates to the nearest previous business day
    start_of_month = get_previous_business_day(current_date.replace(day=1))
    start_of_quarter = get_previous_business_day(pd.Timestamp((current_date - pd.offsets.QuarterBegin(startingMonth=1)).strftime('%Y-%m-%d')))
    start_of_year = get_previous_business_day(current_date.replace(month=1, day=1))
    start_of_5_days = get_previous_business_day(current_date - BDay(5))
    

    with ThreadPoolExecutor() as executor:
        current_price_future = executor.submit(fetch_current_price, symbol, api_token)
        previous_close_price_future = executor.submit(fetch_previous_close_price, symbol, api_token)
        df_month_future = executor.submit(fetch_historical_data, symbol, api_token, start_of_month.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
        df_quarter_future = executor.submit(fetch_historical_data, symbol, api_token, start_of_quarter, current_date.strftime('%Y-%m-%d'))
        df_year_future = executor.submit(fetch_historical_data, symbol, api_token, start_of_year.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
        df_5_days_future = executor.submit(fetch_historical_data, symbol, api_token, start_of_5_days.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))

        current_price = current_price_future.result()
        previous_close_price = previous_close_price_future.result()
        df_month = df_month_future.result()
        df_quarter = df_quarter_future.result()
        df_year = df_year_future.result()
        df_5_days = df_5_days_future.result()

    if current_price is None:
        return symbol, None, None, None, None, None

    if previous_close_price is None:
        today_percentage = None
    else:
        today_percentage = round(((current_price - previous_close_price) / previous_close_price) * 100, 2)

    start_month_price = df_month['adjusted_close'].iloc[0] if not df_month.empty else None
    start_quarter_price = df_quarter['adjusted_close'].iloc[0] if not df_quarter.empty else None
    start_year_price = df_year['adjusted_close'].iloc[0] if not df_year.empty else None
    start_5_days_price = df_5_days['adjusted_close'].iloc[0] if not df_5_days.empty else None

    mtd_percentage = round(((current_price - start_month_price) / start_month_price) * 100, 2) if start_month_price is not None else None
    qtd_percentage = round(((current_price - start_quarter_price) / start_quarter_price) * 100, 2) if start_quarter_price is not None else None
    ytd_percentage = round(((current_price - start_year_price) / start_year_price) * 100, 2) if start_year_price is not None else None
    five_day_percentage = round(((current_price - start_5_days_price) / start_5_days_price) * 100, 2) if start_5_days_price is not None else None

    return symbol, current_price, today_percentage, five_day_percentage, mtd_percentage, qtd_percentage, ytd_percentage

# Function Definitions
def create_dataframe(symbols, api_token):
    data = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(analyze_symbol, symbol, api_token) for symbol in symbols]
        for future in as_completed(futures):
            result = future.result()
            if result[1] is not None:  # Skip if current_price is None
                data.append(result)
    
    df = pd.DataFrame(data, columns=['Symbol', 'Current Price', 'Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
    return df

# Custom CSS for a darker gray sidebar
st.markdown(
    """
    <style>
    /* Change the background color of the sidebar to a darker gray */
    [data-testid="stSidebar"] {
        background-color: #626770; /* You can adjust this hex code for a lighter or darker shade */
    }

    /* Change the text color in the sidebar to dark gray */
    [data-testid="stSidebar"] .css-1d391kg p,
    [data-testid="stSidebar"] .css-1d391kg label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #333333;
    }
    
    /* Customize the selectbox dropdown to match the darker gray theme */
    .stSelectbox label {
        color: #333333;
    }
    
    .stSelectbox .css-11unzgr {
        background-color: #c0c0c0; /* Slightly darker than the sidebar background */
        color: #333333;
    }

    /* Customize the dataframe header and cell colors to match the theme */
    .dataframe thead th {
        background-color: #c0c0c0; /* Slightly darker gray */
        color: #333333;
    }

    .dataframe tbody tr td {
        background-color: #e6e6e6;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the image from a URL or a local file
image_url = "momento_logo.png"  # Update this URL
image = Image.open(image_url)

# Display the logo in the sidebar
st.sidebar.image(image, use_column_width=True)

# Sidebar for analysis selection
st.sidebar.title("Select Analysis Type")
selected_analysis = st.sidebar.radio("Analysis Type", ["BBands analysis", "Sector Overall Performance", "ROC/STDDEV analysis", "Z Score Analysis", "Trailing Correlation Analysis"])

# Sidebar for sector/subsector selection based on analysis type
if selected_analysis == "BBands analysis":
    st.sidebar.title("Select Sector")
    selected_sector = st.sidebar.radio("Sectors", real_sectors + real_subsectors)
    df = fetch_bbands_data_from_firestore(selected_sector)

elif selected_analysis == "ROC/STDDEV analysis":
    st.sidebar.title("Select Performance Type")
    
    # Add "Consecutive_Appearances" option
    performance_type = st.sidebar.radio("Performance Type", ["Sector_Performers", "Subsector_Performers", "Consecutive_Appearances"])
    
    if performance_type in ["Sector_Performers", "Subsector_Performers"]:
        # Fetch data from Firestore for Sector/Subsector Performers
        df = fetch_roc_stddev_data_from_firestore(performance_type)
    
    elif performance_type == "Consecutive_Appearances":
        # Load the S3 credentials and paths from st.secrets
        aws_access_key = st.secrets['aws_access_key']
        aws_secret_key = st.secrets['aws_secret_key']
        region_name = st.secrets['region_name']
        bucket_name = st.secrets['bucket_name']
        sector_csv_path = st.secrets['sector_csv_path']
        subsector_csv_path = st.secrets['subsector_csv_path']
        
        # Set up the S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        
        # Load the CSV files from S3
        sector_df = load_csv_from_s3(s3, bucket_name, sector_csv_path)
        subsector_df = load_csv_from_s3(s3, bucket_name, subsector_csv_path)
        
        # Remove nulls and calculate consecutive appearances
        sector_df_clean = remove_nulls(sector_df)
        sector_consecutive_df = calculate_consecutive_appearances(sector_df_clean)
        subsector_consecutive_df = calculate_consecutive_appearances(subsector_df)
        
        # Process consecutive returns
        api_key = st.secrets['API_KEY']  # Use the API key stored in st.secrets
        df = process_consecutive_returns(sector_consecutive_df, subsector_consecutive_df, api_key)
        
        # Sort the DataFrame by consecutive day return
        df = df.sort_values(['consecutive_day_return'], ascending=[False])

    else:
        st.write("Invalid performance type selected.")


elif selected_analysis == "Z Score Analysis":
    st.sidebar.title("Select Score Type")
    score_type = st.sidebar.radio("Score Type", ["Top_Sectors", "Top_Subsectors"])
    df = fetch_z_score_data_from_firestore(score_type)



# Color mapping for different bands (for BBands analysis)
color_map = {
    'LBand 1STD': 'background-color: lightcoral',  # light red
    'LBand 2STD': 'background-color: red',  # stronger red
    'UBand 1STD': 'color: black; background-color: lightgreen',  # light green with black text
    'UBand 2STD': 'background-color: green',  # stronger green
    'Mid Zone': '',  # no color
}

def highlight_cells(val):
    return color_map.get(val, '')

def prioritize_bands(df):
    band_priority = {
        'UBand 2STD': 1,
        'UBand 1STD': 2,
        'LBand 1STD': 3,
        'LBand 2STD': 4,
        'Mid Zone': 5
    }
    df['Priority'] = df[['Crossing Daily Band', 'Crossing Weekly Band', 'Crossing Monthly Band']].apply(
        lambda x: min(band_priority.get(x[0], 5), band_priority.get(x[1], 5), band_priority.get(x[2], 5)), axis=1
    )
    return df.sort_values('Priority').drop(columns='Priority')

def generate_tradingview_embed(ticker):
    return f"""
    <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_c2a09&symbol={ticker}&interval=D&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[%7B%22id%22%3A%22BB%40tv-basicstudies%22%2C%22inputs%22%3A%5B20%2C2%5D%7D]&theme=Dark&style=1&timezone=exchange&withdateranges=1&hideideas=1&studies_overrides={{}}&overrides={{}}&enabled_features=[]&disabled_features=[]&locale=en&utm_source=www.tradingview.com&utm_medium=widget&utm_campaign=chart&utm_term={ticker}" width="100%" height="600" frameborder="0" allowfullscreen></iframe>
    """

# Color mapping for percentage columns
def color_percentages(val):
    if pd.isna(val):
        return ''
    elif val < 0:
        return 'background-color: lightcoral; color: black'
    else:
        return 'background-color: lightgreen; color: black'

# Main code
if selected_analysis == "BBands analysis":
    sorted_df = prioritize_bands(df)
    highlighted_df = sorted_df.style.applymap(highlight_cells, subset=['Crossing Daily Band', 'Crossing Weekly Band', 'Crossing Monthly Band'])

    st.title(f"{selected_sector} - Bollinger Bands Analysis")
    st.dataframe(highlighted_df, height=500, width=1000)

    # Display chart and data for selected symbol
    selected_ticker = st.selectbox("Select Ticker to View Chart", sorted_df['Symbol'])

    # Generate and display the TradingView chart
    col1, col2 = st.columns([3, 1])

    with col1:
        chart_html = generate_tradingview_embed(selected_ticker)
        st.components.v1.html(chart_html, height=600)

    # Fetch correlations for selected ticker from Firestore
    correlations = fetch_correlations_from_firestore(selected_ticker, selected_sector)
    
    if correlations:
        lowest_5, highest_5 = extract_top_correlations(correlations)
        
        # Display the top 5 highest and lowest correlations
        st.subheader(f"5 Highest Correlations for {selected_ticker}")
        st.dataframe(lowest_5)

        st.subheader(f"5 Lowest Correlations for {selected_ticker}")
        st.dataframe(highest_5)
    

    # Perform and display the analysis for the selected ticker
    symbol, current_price, today_percentage, five_day_percentage, mtd_percentage, qtd_percentage, ytd_percentage = analyze_symbol(selected_ticker, api_token)

    if current_price is not None:
        with col2:
            st.subheader(f"{selected_ticker}")
            st.write(f"**Current Price:** {current_price}")
            st.write(f"**Today:** {today_percentage}%")
            st.write(f"**5-Day:** {five_day_percentage}%")
            st.write(f"**MTD:** {mtd_percentage}%")
            st.write(f"**QTD:** {qtd_percentage}%")
            st.write(f"**YTD:** {ytd_percentage}%")
    else:
        st.write(f"Could not fetch data for {selected_ticker}. Please try again later.")

elif selected_analysis == "Z Score Analysis":
    st.title(f"{score_type} - Z Score Analysis")
    st.dataframe(df, height=500, width=1000)

    # Display chart and data for selected ETF symbol
    selected_ticker = st.selectbox("Select Ticker to View Chart", df['Ticker'])
    selected_sector = df.loc[df['Ticker'] == selected_ticker, 'Sector'].values[0]
    
    # Generate and display the TradingView chart
    col1, col2 = st.columns([3, 1])

    with col1:
        chart_html = generate_tradingview_embed(selected_ticker)
        st.components.v1.html(chart_html, height=600)

    # Perform and display the analysis for the selected ticker
    symbol, current_price, today_percentage, five_day_percentage, mtd_percentage, qtd_percentage, ytd_percentage = analyze_symbol(selected_ticker, api_token)


    if current_price is not None:
        with col2:
            st.subheader(f"{selected_ticker}")
            st.write(f"**Current Price:** {current_price}")
            st.write(f"**Today:** {today_percentage}%")
            st.write(f"**5-Day:** {five_day_percentage}%")
            st.write(f"**MTD:** {mtd_percentage}%")
            st.write(f"**QTD:** {qtd_percentage}%")
            st.write(f"**YTD:** {ytd_percentage}%")
    else:
        st.write(f"Could not fetch data for {selected_ticker}. Please try again later.")
    # Fetch correlations
    correlations = fetch_correlations_from_firestore(selected_ticker, selected_sector)

    if correlations:
        lowest_5, highest_5 = extract_top_correlations(correlations)
        st.subheader(f"5 Highest Correlations for {selected_ticker}")
        st.dataframe(lowest_5)
        st.subheader(f"5 Lowest Correlations for {selected_ticker}")
        st.dataframe(highest_5)

elif selected_analysis == "ROC/STDDEV analysis":
    st.title(f"{performance_type} - ROC/STDDEV Analysis")

    # Check if it's the "Consecutive Appearances" case
    if performance_type == "Consecutive_Appearances":
        # Display the DataFrame with consecutive appearances sorted by returns
        st.dataframe(df, height=500, width=1000)
    else:
        # Display the ROC/STDDEV analysis DataFrame
        st.dataframe(df, height=500, width=1000)

        # Display chart and data for selected symbol
        selected_ticker = st.selectbox("Select Ticker to View Chart", df['Symbol'])

        # Extract the ETF symbol for the selected ticker from the dataframe
        selected_sector = df.loc[df['Symbol'] == selected_ticker, 'Sector'].values[0]
        
        # Generate and display the TradingView chart
        col1, col2 = st.columns([3, 1])

        with col1:
            chart_html = generate_tradingview_embed(selected_ticker)
            st.components.v1.html(chart_html, height=600)

        # Perform and display the analysis for the selected ticker
        symbol, current_price, today_percentage, five_day_percentage, mtd_percentage, qtd_percentage, ytd_percentage = analyze_symbol(selected_ticker, api_token)

        if current_price is not None:
            with col2:
                st.subheader(f"{selected_ticker}")
                st.write(f"**Current Price:** {current_price}")
                st.write(f"**Today:** {today_percentage}%")
                st.write(f"**5-Day:** {five_day_percentage}%")
                st.write(f"**MTD:** {mtd_percentage}%")
                st.write(f"**QTD:** {qtd_percentage}%")
                st.write(f"**YTD:** {ytd_percentage}%")
        else:
            st.write(f"Could not fetch data for {selected_ticker}. Please try again later.")

        # Fetch correlations for selected ticker from Firestore
        correlations = fetch_correlations_from_firestore(selected_ticker, selected_sector)
        
        if correlations:
            lowest_5, highest_5 = extract_top_correlations(correlations)
            
            # Display the top 5 highest and lowest correlations
            st.subheader(f"5 Highest Correlations for {selected_ticker}")
            st.dataframe(lowest_5)

            st.subheader(f"5 Lowest Correlations for {selected_ticker}")
            st.dataframe(highest_5)
            
        # Create the scatter plot for ROC/STDDEV vs RSI
        if 'ROC/STDDEV' in df.columns and 'RSI' in df.columns:
            # Calculate min and max for x and y axes
            x_min, x_max = df['ROC/STDDEV'].min(), df['ROC/STDDEV'].max()
            y_min, y_max = df['RSI'].min(), df['RSI'].max()
            
            # Create scatter plot with dynamic domain
            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('ROC/STDDEV', scale=alt.Scale(domain=[x_min, x_max]), title='ROC/STDDEV'),
                y=alt.Y('RSI', scale=alt.Scale(domain=[y_min, y_max]), title='RSI'),
                color=alt.Color('Symbol', legend=None),
                tooltip=['Symbol', 'ROC/STDDEV', 'RSI']
            ).interactive()
            
            # Add text labels to the points
            text = scatter.mark_text(
                align='left',
                baseline='middle',
                dx=7,
                fontSize=10
            ).encode(
                text='Symbol'
            )
            
            # Combine the scatter plot and text
            chart = scatter + text
            
            chart = chart.properties(
                title='ROC/STDDEV vs RSI Scatter Plot'
            ).configure_axis(
                grid=True
            ).configure_title(
                fontSize=20
            ).configure_legend(
                labelFontSize=12,
                titleFontSize=14
            )
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("The dataframe does not contain the required columns for the scatter plot.")


# New Section: Sector and Subsector Performance
elif selected_analysis == "Sector Overall Performance":
    # Initial load of the dataframes
    sector_df = create_dataframe(sectors, api_token)
    subsector_df = create_dataframe(subsectors, api_token)
    ratecut_etfs_df = create_dataframe(ratecut_etfs, api_token)
    macro_etfs_df = create_dataframe(macro_etfs, api_token)

    # Create a refresh button
    if st.button("Refresh Data"):
        # When button is pressed, refresh and rerun the dataframes
        sector_df = create_dataframe(sectors, api_token)
        subsector_df = create_dataframe(subsectors, api_token)
        ratecut_etfs_df = create_dataframe(ratecut_etfs, api_token)
        macro_etfs_df = create_dataframe(macro_etfs, api_token)
    
    st.title("Sector and Subsector Performance")

    # Create tabs for different dataframes
    tab1, tab2, tab3, tab4 = st.tabs(["Sector Performance", "Subsector Performance", "Commodities & Metals Performance", "Macro Performance"])

        # Sector performance tab
    with tab1:
        st.subheader("Sector DataFrame")
        sector_df_styled = sector_df.style.applymap(color_percentages, subset=['Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
        sector_df_styled = sector_df_styled.format({
            'Current Price': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'Today %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            '5-Day %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'MTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'QTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'YTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
        })
        st.dataframe(sector_df_styled, height=500, width=1000)
    
    # Subsector performance tab
    with tab2:
        st.subheader("Subsector DataFrame")
        subsector_df_styled = subsector_df.style.applymap(color_percentages, subset=['Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
        subsector_df_styled = subsector_df_styled.format({
            'Current Price': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'Today %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            '5-Day %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'MTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'QTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'YTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
        })
        st.dataframe(subsector_df_styled, height=500, width=1000)
    
    # Bonds & Metals performance tab
    with tab3:
        st.subheader("Commodities & Metals DataFrame")
        ratecut_etfs_df_styled = ratecut_etfs_df.style.applymap(color_percentages, subset=['Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
        ratecut_etfs_df_styled = ratecut_etfs_df_styled.format({
            'Current Price': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'Today %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            '5-Day %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'MTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'QTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'YTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
        })
        st.dataframe(ratecut_etfs_df_styled, height=500, width=1000)
    
    # Countries performance tab
    with tab4:
        st.subheader("Countries DataFrame")
        macro_etfs_df_styled = macro_etfs_df.style.applymap(color_percentages, subset=['Today %', '5-Day %', 'MTD %', 'QTD %', 'YTD %'])
        macro_etfs_df_styled = macro_etfs_df_styled.format({
            'Current Price': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'Today %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            '5-Day %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'MTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'QTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            'YTD %': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
        })
        st.dataframe(macro_etfs_df_styled, height=500, width=1000)
    

# Add Trailing Correlation Analysis
elif selected_analysis == "Trailing Correlation Analysis":

    # Allow user to input symbols and benchmarks
    st.write("Enter the symbols and benchmarks for analysis.")
    symbols = st_tags(label='### Symbols', text='Add symbols (e.g., AAPL)', suggestions=["AAPL", "GOOGL", "TSLA"], maxtags=10)
    benchmarks = st_tags(label='### Benchmarks', text='Add benchmarks (e.g., SPY)', suggestions=["SPY", "DIA", "QQQ"], maxtags=10)

    rolling_window = st.slider("Rolling Window (Days)", min_value=10, max_value=100, value=30)
        # Execute upon button click
    if st.button("Run Trailing Correlation Analysis"):
        results = calculate_rolling_correlations(symbols, benchmarks, api_token, rolling_window)
        visualize_rolling_correlations(results)

import ccxt
import pandas as pd
import ta

def get_data(exchange_name, symbol, period):
    '''
    Extract daily cryptocurrency stock data

    @param exchange_name: Exchange market to retrieve data
    @param symbol: Cryptocurrency symbol to retrieve data
    @param period: Period for data retrieval
    '''
    # Retrieve data from the exchange market for a specific cryptocurrency symbol
    exchange = getattr(ccxt, exchange_name)()
    ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=period)

    # Create a table for the data
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Return the data table
    return df

if __name__ == "__main__":
    df = get_data('kraken', 'BTC/AUD', 720)
    df.to_csv(f'aco_data.csv', index=False)
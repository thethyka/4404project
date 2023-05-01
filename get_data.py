import ccxt
import pandas as pd
import ta
from ta.trend import SMAIndicator, EMAIndicator


def get_data(exchange_name, symbol, window, sma_window_size, min_ema_window_size, max_ema_window_size, ema_increment):
    '''
    Extract daily cryptocurrency stock data

    @param exchange_name: Exchange market to retrieve data
    @param symbol: Cryptocurrency symbol to retrieve data
    @param window: Period for data retrieval
    @param sma_window_size: Window size of SMA for data retrieval
    @param min_ema_window_size: Minimum window size of EMA for data retrieval
    @param max_ema_window_size: Maximum window size of EMA for data retrieval
    @param ema_increment: Window size increments for EMA
    '''
    # Retrieve data from the exchange market for a specific cryptocurrency symbol
    exchange = getattr(ccxt, exchange_name)()
    ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=window)

    # Create a table for the data
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Add SMA data into the table
    sma_indicator = SMAIndicator(df['close'], sma_window_size, True)
    df[f'sma-{sma_window_size}'] = sma_indicator.sma_indicator()

    # Add EMA data into the table
    for i in range(min_ema_window_size, max_ema_window_size + 1, ema_increment):
        ema_indicator = EMAIndicator(df['close'], i, True)
        df[f'ema-{i}'] = ema_indicator.ema_indicator()

    # Return the data table
    return df

if __name__ == "__main__":
    df = get_data('kraken', 'BTC/AUD', 720, 50, 5, 100, 5)
    df.to_csv(f'aco_data.csv', index=False)
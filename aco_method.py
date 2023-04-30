# python3 aco_method.py 100 0.01 720 50 5 100 5
# python3 aco_method.py 125 0.01 720 50 5 100 5
# python3 aco_method.py 250 0.01 720 50 5 100 5

import sys
import random
import math
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
    ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=window, since=1620691200000)

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

def buy(wallet, price, amount):
    '''
    Purchase cryptocurrency

    @param wallet: Current amount of AUD in wallet
    @param price: Current price of the cryptocurrency symbol
    @param amount: Current amount of crytocurrency symbol in hand
    '''
    new_amount = amount + wallet/price
    new_wallet = 0
    new_trigger = "BUY"
    return new_wallet, new_amount, new_trigger

def sell(wallet, price, amount):
    '''
    Sell cryptocurrency

    @param wallet: Current amount of AUD in wallet
    @param price: Current price of the cryptocurrency symbol
    @param amount: Current amount of crytocurrency symbol in hand
    '''
    new_wallet = wallet + price*amount
    new_amount = 0
    new_trigger = "SELL"
    return new_wallet, new_amount, new_trigger

def run_simulation(num_ants, Q, period, fix_sma, min_ema, max_ema, ema_increment):
    '''
    Set a sell trigger and sell cryptocurrency

    @param num_ants: Number of ants
    @param Q: Pheremone constant
    @param period: Period of data
    @param fix_sma: Window size of SMA 
    @param min_ema: Minimum window size of EMA
    @param max_ema: Maximum window size of EMA
    @param ema_increment: Window size increments for EMA
    '''
    # Random seed to provide same results each run
    random.seed(100)

    # Retrive data from exchange market
    data = get_data('kraken', 'BTC/AUD', int(period), int(fix_sma), int(min_ema), int(max_ema), int(ema_increment))
    
    # Initialise variables
    best_ema, best_wallet = 0, 0
    initial_wallet = 100
    
    # Create probabilistic array of EMA periods
    ema_periods = []
    for periods in range(min_ema, max_ema + 1, ema_increment):
        ema_periods.append(periods)
    ema_probabilities = [1/len(ema_periods) for r in range(len(ema_periods))]

    # Create headers for table
    results = pd.DataFrame(columns=['Ant'] + [f'EMA-{ema_periods[r]}' for r in range(len(ema_periods))] + ['Chosen EMA'])
    
    # Run iterations
    for n in range(num_ants):
        # Initialise local best variables for the ant
        local_best_ema_index, local_best_wallet = 0, 0

        # Choosing EMA period based on probabilities
        chosen_ema_value = random.choices(ema_periods, ema_probabilities)[0]

        # Add values into table as a new row
        result_row = {'Ant': n + 1, 'Chosen EMA': f'EMA-{chosen_ema_value}'}
        for r in range(len(ema_probabilities)):
            result_row[f'EMA-{ema_periods[r]}'] = ema_probabilities[r]
        results = pd.concat([results, pd.DataFrame(result_row, index=[0])], ignore_index=True)

        # Iterate through each EMA period
        for m in range(len(ema_periods)):
            # Initialise ant for each EMA period
            ema_value = ema_periods[m]
            wallet = initial_wallet
            btc_amount = 0
            trigger = None
            min_sell_price = 0

            # Buy/Sell BTC depending on conditions set
            ''' TO DO - REDEFINE BUY/SELL TRIGGERS '''
            for i in range(period):
                # Initial buy trigger
                if i == 0: 
                    price = data['open'][i]*1.02
                    wallet, btc_amount, trigger = buy(wallet, price, btc_amount)
                    min_sell_price = price
                
                # Final sell trigger
                elif i == period - 1:
                    price = data['close'][i]*0.98
                    wallet, btc_amount, trigger = sell(wallet, price, btc_amount)

                else:
                    # Buy trigger
                    if data[f'sma-{fix_sma}'][i] > data[f'ema-{ema_value}'][i] and data[f'sma-{fix_sma}'][i - 1] < data[f'ema-{ema_value}'][i - 1] and trigger != "BUY":
                        price = data['close'][i]*1.02
                        wallet, btc_amount, trigger = buy(wallet, price, btc_amount)
                        min_sell_price = price

                    # Sell trigger
                    elif data[f'sma-{fix_sma}'][i] < data[f'ema-{ema_value}'][i] and data[f'sma-{fix_sma}'][i - 1] > data[f'ema-{ema_value}'][i -1] and min_sell_price <= data['close'][i] and trigger != "SELL":
                        price = data['close'][i]*0.98
                        wallet, btc_amount, trigger = sell(wallet, price, btc_amount)

            # Checking for the best parameters locally
            if wallet > local_best_wallet:
                local_best_wallet = wallet
                local_best_ema_index = m

        # Changing probabilities of EMA periods
        diff = - Q / (len(ema_probabilities) - 1)

        ema_probabilities[local_best_ema_index] += Q

        for p in range(len(ema_probabilities)):
            if p != local_best_ema_index:
                ema_probabilities[p] += diff

        prob_sum = sum(ema_probabilities)
        if prob_sum != 1:
            ema_probabilities = [p/prob_sum for p in ema_probabilities]

        for q in range(len(ema_probabilities)):
            if ema_probabilities[q] < 0:
                ema_probabilities[q] = 0
            elif ema_probabilities[q] > 1:
                ema_probabilities[q] = 1

        # Checking for best parameters globally
        if local_best_wallet > best_wallet:
            best_wallet = local_best_wallet
            best_ema = ema_periods[local_best_ema_index]

    # Printing results and saving as Excel CSV
    print(results.to_string(index=False))
    print(f'Best overall: EMA-{best_ema}, ${best_wallet}')
    #results.to_csv(f'aco_results.csv', index=False)
    #data.to_csv(f'aco_data.csv', index=False)

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(f'Usage: python3 {__file__} num_ants pheremone_constant period fixed_sma_window min_ema_window max_ema_window ema_increment')
        print('num_ants: Number of ants to generate\n'
              'pheremone_constant: Probability constant for updating ACO (value from 0 to 1)\n'
              'period: Period of dataset\n'
              'fixed_sma_window: SMA window size that is fixed\n'
              'min_ema_window: Minimum EMA window size\n'
              'max_ema_window: Maximum EMA window size\n'
              'ema_increment: Step increment for EMA window size')
        sys.exit(1)

    num_ants = int(sys.argv[1])
    pheremone_constant = float(sys.argv[2])
    period = int(sys.argv[3])
    fixed_sma_window = int(sys.argv[4])
    min_ema_window = int(sys.argv[5])
    max_ema_window = int(sys.argv[6])
    ema_increment = int(sys.argv[7])

    if pheremone_constant > 1:
        print(f'Pheremone value of {sys.argv[2]} is not a valid input. Please enter a value from 0 to 1.')
        sys.exit(1)

    run_simulation(num_ants, pheremone_constant, period, fixed_sma_window, min_ema_window, max_ema_window, ema_increment)
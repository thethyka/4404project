# python3 aco_method.py 100 0.01 720 50 5 100 5
# python3 aco_method.py 125 0.01 720 50 5 100 5
# python3 aco_method.py 250 0.01 720 50 5 100 5

import sys
import random
import math
import pandas as pd

def csv_to_df(filename):
    df = pd.read_csv(filename)
    print(df)
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

def run_simulation(num_ants, Q):
    '''
    Set a sell trigger and sell cryptocurrency

    @param num_ants: Number of ants
    @param Q: Pheremone constant
    @param period: Period of data
    '''
    # Random seed to provide same results each run
    random.seed(100)

    # Retrive data from exchange market
    data = csv_to_df('aco_data.csv')
    
    # Initialise variables
    best_ema, best_wallet = 0, 0
    initial_wallet = 100

    # Fixed based on aco_data.csv
    period = 720
    fix_sma = 50
    min_ema = 5
    max_ema = 100
    ema_increment = 5
    
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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f'Usage: python3 {__file__} num_ants pheremone_constant')
        print('num_ants: Number of ants to generate\n'
              'pheremone_constant: Probability constant for updating ACO (value from 0 to 1)\n')
        sys.exit(1)

    num_ants = int(sys.argv[1])
    pheremone_constant = float(sys.argv[2])

    if pheremone_constant > 1:
        print(f'Pheremone value of {sys.argv[2]} is not a valid input. Please enter a value from 0 to 1.')
        sys.exit(1)

    run_simulation(num_ants, pheremone_constant)
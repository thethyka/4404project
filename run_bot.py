import pandas as pd


class TradingBot:
    def __init__(self, data):
        self.data = data

        # a dictionary of all parameter options
        
        # a huge list of all parameter dictionaries in search space
        # as of now, parameter list is [SMA period, EMA period, resistance threshold, support threshold, volume threshold]
        self.all_parameters = None

        # found once we've optimised - dictionary
        self.best_parameters = {
            'SMA_period': None,
            'EMA_period': None,
            'resistance_threshold': None,
            'support_threshold': None,
            'volume_threshold': None,
        }

        self.period = 720
        self.wallet = 100
        self.btc = 0

    def buy_pulse(self, t, P):
        pass
        # Implement your buy pulse logic here

    def sell_pulse(self, t, P):
        # Implement your sell pulse logic here
        pass

    def optimise(self):
        # Implement your optimization algorithm here.
        # Uses all paramters to find the best parameters, sets self.best_parameters accordingly


        self.best_parameters = None

    def execute_buy(self, t):
        print(f"Buy at t={t}")
        # Execute buy order
        price = self.data['open'][t]*1.02
        self.btc = self.wallet / price       
        self.wallet = 0

    def execute_sell(self, t):        
        print(f"Sell at t={t}")
        # Execute sell order

        price = self.data['close'][t]*0.98
        self.wallet = self.btc * price
        self.btc = 0


    def reset(self):
        # reset our wallet and btc once we've run the bot
        self.wallet = 100
        self.btc = 0

    def run(self, P):
        # We haven't bought our bitcoin yet at t=0
        # P is a list of parameters
        bought = False

        for t in range(1, len(self.period)):
            buy_t = self.buy_pulse(t, P)
            buy_prev = self.buy_pulse(t - 1, P)
            sell_t = self.sell_pulse(t, P)
            sell_prev = self.sell_pulse(t - 1, P)

            buy_trigger = buy_t and not buy_prev and not (sell_t and not sell_prev)
            sell_trigger = sell_t and not sell_prev and not (buy_t and not buy_prev)

            if self.wallet != 0 and buy_trigger:
                self.execute_buy(t)
                bought = True
            elif self.wallet == 0 and sell_trigger:
                self.execute_sell(t)
                bought = False

        # Assume a sale on the close of the last day
        if bought:
            self.execute_sell(len(self.period) - 1)

        # reset our wallet once we've run
        final_wallet = self.wallet
        self.reset()
        return final_wallet

# Load your OHLCV data for the past 720 days
data = pd.read_csv("aco_data.csv")

# Initialize the trading bot with the data
bot = TradingBot(data)

# Optimize the bot's parameters and logical expressions
bot.optimise()

# Run the trading bot
bot.run(bot.best_parameters)

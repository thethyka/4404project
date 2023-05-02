import pandas as pd
import ta

from aco_optimiser import ACOOptimiser

class TradingBot:
    def __init__(self, data):
        self.data = data

        self.all_indicators = [('ta.momentum.KAMAIndicator', 'close', 'kama()'), ('ta.momentum.RSIIndicator', 'close', 'rsi()'), ('ta.momentum.ROCIndicator', 'close', 'roc()'), ('ta.momentum.StochasticOscillator', 'high, low, close', 'stoch()'), ('ta.momentum.StochRSIIndicator', 'close', 'stochrsi()'), ('ta.trend.ADXIndicator', 'high, low, close', 'adx()'), ('ta.trend.AroonIndicator', 'close', 'aroon_indicator()'), ('ta.trend.EMAIndicator', 'close', 'ema_indicator()'), ('ta.trend.SMAIndicator', 'close', 'sma_indicator()'), ('ta.trend.VortexIndicator', 'high, low, close', 'vortex_indicator_diff()'), ('ta.volatility.AverageTrueRange', 'high, low, close', 'average_true_range()'), ('ta.volatility.BollingerBands', 'close', 'bollinger_mavg()'), ('ta.volatility.DonchianChannel', 'high, low, close', 'donchian_channel_mband()'), ('ta.volatility.KeltnerChannel', 'high, low, close', 'keltner_channel_mband()'), ('ta.volatility.UlcerIndex', 'close', 'ulcer_index()'), ('ta.volume.ChaikinMoneyFlowIndicator', 'high, low, close, volume', 'chaikin_money_flow()'), ('ta.volume.EaseOfMovementIndicator', 'high, low, volume', 'ease_of_movement()'), ('ta.volume.ForceIndexIndicator', 'close, volume', 'force_index()'), ('ta.volume.MFIIndicator', 'high, low, close, volume', 'money_flow_index()'), ('ta.volume.VolumeWeightedAveragePrice', 'high, low, close, volume', 'volume_weighted_average_price()')]

        self.window_sizes = [7, 21]  # all window size options


        ### ONLY IF WE WANT TO ADD TYPES OF INDICATORS ###
        # self.options = {


        #     # each list element is of form (indicator, args, output function)
        #     'momentum': [('ta.momentum.KAMAIndicator', 'close', 'kama()'), ('ta.momentum.RSIIndicator', 'close', 'rsi()'), ('ta.momentum.ROCIndicator', 'close', 'roc()'), ('ta.momentum.StochasticOscillator', 'high, low, close', 'stoch()'), ('ta.momentum.StochRSIIndicator', 'close', 'stochrsi()')],
        #     'trend': [('ta.trend.ADXIndicator', 'high, low, close', 'adx()'), ('ta.trend.AroonIndicator', 'close', 'aroon_indicator()'), ('ta.trend.EMAIndicator', 'close', 'ema_indicator()'), ('ta.trend.SMAIndicator', 'close', 'sma_indicator()'), ('ta.trend.VortexIndicator', 'high, low, close', 'vortex_indicator_diff()')],
        #     'volatility': [('ta.volatility.AverageTrueRange', 'high, low, close', 'average_true_range()'), ('ta.volatility.BollingerBands', 'close', 'bollinger_mavg()'), ('ta.volatility.DonchianChannel', 'high, low, close', 'donchian_channel_mband()'), ('ta.volatility.KeltnerChannel', 'high, low, close', 'keltner_channel_mband()'), ('ta.volatility.UlcerIndex', 'close', 'ulcer_index()')],
        #     # maybe add sma ease of movement too to volume list
        #     'volume': [('ta.volume.ChaikinMoneyFlowIndicator', 'high, low, close, volume', 'chaikin_money_flow()'), ('ta.volume.EaseOfMovementIndicator', 'high, low, volume', 'ease_of_movement()'), ('ta.volume.ForceIndexIndicator', 'close, volume', 'force_index()'), ('ta.volume.MFIIndicator', 'high, low, close, volume', 'money_flow_index()'), ('ta.volume.VolumeWeightedAveragePrice', 'high, low, close, volume', 'volume_weighted_average_price()')],           
        # }

        # dnf formula represented by a list of lists. Each list is a clause and the elements of the list are the literals. Each literal is a tuple of (indicator, window1, window2, 0/1 (greater than or less than))
        self.best_buy_dnf_formula = None
        self.best_sell_dnf_formula = None

        self.period = 720
        self.wallet = 100
        self.btc = 0

    def test_indicators(self):
        # runs through each indicator to verify they're all valid and working.
        for indicator in self.all_indicators:
            for window in self.window_sizes:
                print(self.convert_to_indicator(indicator, window))

    def convert_to_indicator(self, indicator, window):
        # converts the index, window into an actual indicator
        # index is index of indicator in all indicators
        # window is index of window in window sizes
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        volume = self.data['volume']
        open = self.data['open']

        indicator_str = f"{indicator[0]}({indicator[1]}, window={window}).{indicator[2]}"
        return eval(indicator_str)


    
    def buy_pulse(self, t, buy_dnf_formula):
        # dnf formula represented by a list of lists. Each list is a clause and the elements of the list are the literals. Each literal is a tuple of (indicator, window1, window2, 0/1 (greater than or less than))
        pass
        

    def sell_pulse(self, t, sell_dnf_formula):
        pass


    def optimise(self):

        # Implement your optimization algorithm here.
        # Uses all paramters to find the best parameters, sets best dnfs accordingly.
        optimiser = ACOOptimiser(bot,...) 
        self.best_buy_dnf_formula, self.best_sell_dnf_formula = optimiser.aco_algorithm()
        pass

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

    def run(self, buy_dnf_formula, sell_dnf_formula):
        # We haven't bought our bitcoin yet at t=0
        # Sell and Buy are inplemented by the formulas
        bought = False

        for t in range(1, self.period):
            buy_t = self.buy_pulse(t, buy_dnf_formula)
            buy_prev = self.buy_pulse(t - 1, buy_dnf_formula)
            sell_t = self.sell_pulse(t, sell_dnf_formula)
            sell_prev = self.sell_pulse(t - 1, sell_dnf_formula)

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
            self.execute_sell(self.period - 1)

        # reset our wallet once we've run
        final_wallet = self.wallet
        self.reset()
        return final_wallet

# Load your OHLCV data for the past 720 days
data = pd.read_csv("aco_data.csv")

# Initialize the trading bot with the data
bot = TradingBot(data)

# Optimize the bot's parameters and logical expressions
# bot.optimise()

# Run the trading bot
# print(bot.run(bot.best_parameters))

bot.test_indicators()







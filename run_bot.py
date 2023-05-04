import pandas as pd
import ta
import itertools
import math
import warnings

# Ignore RuntimeWarning caused by invalid value encountered in scalar divide
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

from aco_optimiser import ACOOptimiser

class Literal:
    
    # Represents our literals in the DNF formula. 
    # Run evaluate(t) to get the value of the literal at time t.
    def __init__(self, indicator, window1, window2, data):
        self.indicator = indicator
        self.window1 = window1
        self.window2 = window2
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        open = data['open']

        indicator_str_1 = f"{self.indicator[0]}({self.indicator[1]}, window={self.window1}).{self.indicator[2]}"
        indicator_str_2 = f"{self.indicator[0]}({self.indicator[1]}, window={self.window2}).{self.indicator[2]}"
        self.ind1 = eval(indicator_str_1)
        self.ind2 = eval(indicator_str_2)

    def __str__(self):
        return f"{self.indicator[0]}_{self.window1}_{self.window2}"

    def __repr__(self):
        return self.__str__()
    
    def key(self):
        return f"{self.indicator[0].split('.')[2]}_{self.window1}_{self.window2}"


    def list_indicators(self):
        return (self.ind1, self.ind2)


    def evaluate(self, t, isPos):
        # Any time we have NaN we return false
        if math.isnan(self.ind1[t]) or math.isnan(self.ind2[t]):
            return False
        
        if isPos:
            return (self.ind1[t] > self.ind2[t])
        else:
            return (self.ind2[t] > self.ind1[t])

class TradingBot:
    def __init__(self, data):
        self.data = data

        self.all_indicators = [('ta.momentum.KAMAIndicator', 'close', 'kama()'), ('ta.momentum.RSIIndicator', 'close', 'rsi()'), ('ta.momentum.ROCIndicator', 'close', 'roc()'), ('ta.momentum.StochasticOscillator', 'high, low, close', 'stoch()'), ('ta.momentum.StochRSIIndicator', 'close', 'stochrsi()'), ('ta.trend.ADXIndicator', 'high, low, close', 'adx()'), ('ta.trend.AroonIndicator', 'close', 'aroon_indicator()'), ('ta.trend.EMAIndicator', 'close', 'ema_indicator()'), ('ta.trend.SMAIndicator', 'close', 'sma_indicator()'), ('ta.trend.VortexIndicator', 'high, low, close', 'vortex_indicator_diff()'), ('ta.volatility.AverageTrueRange', 'high, low, close', 'average_true_range()'), ('ta.volatility.BollingerBands', 'close', 'bollinger_mavg()'), ('ta.volatility.DonchianChannel', 'high, low, close', 'donchian_channel_mband()'), ('ta.volatility.KeltnerChannel', 'high, low, close', 'keltner_channel_mband()'), ('ta.volatility.UlcerIndex', 'close', 'ulcer_index()'), ('ta.volume.ChaikinMoneyFlowIndicator', 'high, low, close, volume', 'chaikin_money_flow()'), ('ta.volume.EaseOfMovementIndicator', 'high, low, volume', 'ease_of_movement()'), ('ta.volume.ForceIndexIndicator', 'close, volume', 'force_index()'), ('ta.volume.MFIIndicator', 'high, low, close, volume', 'money_flow_index()'), ('ta.volume.VolumeWeightedAveragePrice', 'high, low, close, volume', 'volume_weighted_average_price()')]

        # Window issue prevented by evaluating literals with NaN as false.
        self.window_sizes = [7, 21]  # all window size options, ascending order.


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


        self.literal_dict = self.set_literal_dict()

        self.period = 720
        self.wallet = 100
        self.btc = 0

    def test_indicators(self):
        # runs through each indicator to verify they're all valid and working.
        for key, literal in self.literal_dict.items():
            print(key)
            print(literal.list_indicators())


    def set_literal_dict(self):
        literal_dict = {}
        for indicator in self.all_indicators:
            for i, window1 in enumerate(self.window_sizes):
                for window2 in self.window_sizes[i+1:]:
                    literal = Literal(indicator, window1, window2, self.data)
                    literal_dict[literal.key()] = literal
        return literal_dict

    
    def buy_pulse(self, t, buy_dnf_formula):
        # Given a buy_dnf_formula is a list of lists of tuples (key, isPos)
        for clause in buy_dnf_formula:
            clause_true = True
            for key, isPos in clause:
                literal = self.literal_dict[key]
                if not literal.evaluate(t, isPos):
                    clause_true = False
                    break
            if clause_true:
                return True
        return False

    def sell_pulse(self, t, sell_dnf_formula):
        # Given a sell_dnf_formula is a list of lists of tuples (key, isPos)
        for clause in sell_dnf_formula:
            clause_true = True
            for key, isPos in clause:
                literal = self.literal_dict[key]
                if not literal.evaluate(t, isPos):
                    clause_true = False
                    break
            if clause_true:
                return True
        return False

    def optimise(self):
        # Uses all paramters to find the best parameters, sets best dnfs accordingly.
        cost_function = lambda buy_dnf, sell_dnf: self.run(buy_dnf, sell_dnf)
        optimiser = ACOOptimiser(cost_function, self.literal_dict.keys())
        self.best_buy_dnf_formula, self.best_sell_dnf_formula = optimiser.aco_algorithm()

    def execute_buy(self, t):
        print(f"Buy at t={t}")
        # Execute buy order
        price = self.data['open'][t]
        self.btc = self.wallet*0.98 / price       
        self.wallet = 0

    def execute_sell(self, t):        
        print(f"Sell at t={t}")
        # Execute sell order
        price = self.data['close'][t]
        self.wallet = self.btc*0.98 * price
        self.btc = 0


    def reset(self):
        # reset our wallet and btc once we've run the bot
        self.wallet = 100
        self.btc = 0

    def run(self, buy_dnf_formula, sell_dnf_formula):
        # We haven't bought our bitcoin yet at t=0
        # Sell and Buy are implemented by the formulas
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
bot.test_indicators()
bot.optimise()


print(bot.best_buy_dnf_formula)
print(bot.best_sell_dnf_formula)
# Optimize the bot's parameters and logical expressions
# bot.optimise()

# Run the trading bot
# print(bot.run(bot.best_parameters))






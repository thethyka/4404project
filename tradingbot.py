import pandas as pd
import numpy as np
import ta
import itertools
import math
import warnings
import subprocess
from aco_optimiser import Ant
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    def __init__(self, data, window_sizes, acoparams):
        self.data = data

        self.all_indicators = [('ta.momentum.KAMAIndicator', 'close', 'kama()'), ('ta.momentum.RSIIndicator', 'close', 'rsi()'), ('ta.momentum.ROCIndicator', 'close', 'roc()'), ('ta.momentum.StochasticOscillator', 'high, low, close', 'stoch()'), ('ta.momentum.StochRSIIndicator', 'close', 'stochrsi()'), ('ta.trend.ADXIndicator', 'high, low, close', 'adx()'), ('ta.trend.AroonIndicator', 'close', 'aroon_indicator()'), ('ta.trend.EMAIndicator', 'close', 'ema_indicator()'), ('ta.trend.SMAIndicator', 'close', 'sma_indicator()'), ('ta.trend.VortexIndicator', 'high, low, close', 'vortex_indicator_diff()'), ('ta.volatility.AverageTrueRange', 'high, low, close', 'average_true_range()'), ('ta.volatility.BollingerBands', 'close', 'bollinger_mavg()'), ('ta.volatility.DonchianChannel', 'high, low, close', 'donchian_channel_mband()'), ('ta.volatility.KeltnerChannel', 'high, low, close', 'keltner_channel_mband()'), ('ta.volatility.UlcerIndex', 'close', 'ulcer_index()'), ('ta.volume.ChaikinMoneyFlowIndicator', 'high, low, close, volume', 'chaikin_money_flow()'), ('ta.volume.EaseOfMovementIndicator', 'high, low, volume', 'ease_of_movement()'), ('ta.volume.ForceIndexIndicator', 'close, volume', 'force_index()'), ('ta.volume.MFIIndicator', 'high, low, close, volume', 'money_flow_index()'), ('ta.volume.VolumeWeightedAveragePrice', 'high, low, close, volume', 'volume_weighted_average_price()')]




        # Window issue prevented by evaluating literals with NaN as false.
        self.window_sizes = window_sizes  # all window size options, ascending order.
        self.acoparams = acoparams
        
        self.period = len(data)

        if(max(self.window_sizes) >= self.period):
            raise Exception("Window size is larger than data period.")
        
        ### ONLY IF WE WANT TO ADD TYPES OF INDICATORS ###
        # self.options = {


        #     # each list element is of form (indicator, args, output function)
        #     'momentum': [('ta.momentum.KAMAIndicator', 'close', 'kama()'), ('ta.momentum.RSIIndicator', 'close', 'rsi()'), ('ta.momentum.ROCIndicator', 'close', 'roc()'), ('ta.momentum.StochasticOscillator', 'high, low, close', 'stoch()'), ('ta.momentum.StochRSIIndicator', 'close', 'stochrsi()')],
        #     'trend': [('ta.trend.ADXIndicator', 'high, low, close', 'adx()'), ('ta.trend.AroonIndicator', 'close', 'aroon_indicator()'), ('ta.trend.EMAIndicator', 'close', 'ema_indicator()'), ('ta.trend.SMAIndicator', 'close', 'sma_indicator()'), ('ta.trend.VortexIndicator', 'high, low, close', 'vortex_indicator_diff()')],
        #     'volatility': [('ta.volatility.AverageTrueRange', 'high, low, close', 'average_true_range()'), ('ta.volatility.BollingerBands', 'close', 'bollinger_mavg()'), ('ta.volatility.DonchianChannel', 'high, low, close', 'donchian_channel_mband()'), ('ta.volatility.KeltnerChannel', 'high, low, close', 'keltner_channel_mband()'), ('ta.volatility.UlcerIndex', 'close', 'ulcer_index()')],
        #     # maybe add sma ease of movement too to volume list
        #     'volume': [('ta.volume.ChaikinMoneyFlowIndicator', 'high, low, close, volume', 'chaikin_money_flow()'), ('ta.volume.EaseOfMovementIndicator', 'high, low, volume', 'ease_of_movement()'), ('ta.volume.ForceIndexIndicator', 'close, volume', 'force_index()'), ('ta.volume.MFIIndicator', 'high, low, close, volume', 'money_flow_index()'), ('ta.volume.VolumeWeightedAveragePrice', 'high, low, close, volume', 'volume_weighted_average_price()')],           
        # }

     

        self.literal_dict = self.set_literal_dict()

        self.wallet = 100
        self.btc = 0
        self.bought = False

        self.buy_signals = []
        self.sell_signals = []

        self.historical_buy_pheromones = []
        self.historical_sell_pheromones = []


        self.best_ants = []


        self.ant_profits = []

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
        optimiser = ACOOptimiser(cost_function, self.literal_dict.keys(), self.acoparams)
        res = optimiser.aco_algorithm()
        self.best_ants = res[0]
        self.historical_buy_pheromones = res[1]
        self.historical_sell_pheromones = res[2]
        self.ant_profits = res[3]


    def execute_buy(self, t):
        #print(f"Buy at t={t}")
        # Execute buy order
        price = self.data['open'][t]
        self.btc = self.wallet*0.98 / price       
        self.wallet = 0

    def execute_sell(self, t):        
        #print(f"Sell at t={t}")
        # Execute sell order
        price = self.data['close'][t]
        self.wallet = self.btc*0.98 * price
        self.btc = 0


    def reset(self):
        # reset our wallet and btc once we've run the bot
        self.wallet = 100
        self.btc = 0
        self.bought = False
        self.buy_signals = []
        self.sell_signals = []

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
    
    def animate_run(self, buy_dnf_formula, sell_dnf_formula):
        fig, ax = plt.subplots()

        # Plot the initial data
        ax.plot(self.data.index, self.data['close'], label='close')

        # Set up the legend
        ax.legend(loc='best')

        ani = FuncAnimation(fig, self.update, fargs=(ax, buy_dnf_formula, sell_dnf_formula),
                        frames=range(self.period - 1), interval = 25, repeat=True)

        plt.show()



    def update(self, frame, ax, buy_dnf_formula, sell_dnf_formula):
        # We haven't bought our bitcoin yet at t=0
        # Sell and Buy are implemented by the formulas

        t = frame + 1

        buy_t = self.buy_pulse(t, buy_dnf_formula)
        buy_prev = self.buy_pulse(t - 1, buy_dnf_formula)
        sell_t = self.sell_pulse(t, sell_dnf_formula)
        sell_prev = self.sell_pulse(t - 1, sell_dnf_formula)

        buy_trigger = buy_t and not buy_prev and not (sell_t and not sell_prev)
        sell_trigger = sell_t and not sell_prev and not (buy_t and not buy_prev)

        if not self.bought and buy_trigger:
            self.execute_buy(t)
            self.bought = True
            self.buy_signals.append(t)

        elif self.bought and sell_trigger:
            self.execute_sell(t)
            self.bought = False
            self.sell_signals.append(t)

        # Assume a sale on the close of the last day
        if self.bought and t == self.period - 1:
            self.execute_sell(self.period - 1)
            self.sell_signals.append(self.period - 1)
            

        # reset our wallet once we've run


        ax.clear()
        ax.plot(self.data.index[:t], self.data['close'][:t], label='close')

        buy_signals_t = [x for x in self.buy_signals if x <= t]
        sell_signals_t = [x for x in self.sell_signals if x <= t]
  

        ax.scatter(buy_signals_t, self.data.loc[buy_signals_t, 'close'], color='g', marker='^', label='Buy')
        ax.scatter(sell_signals_t, self.data.loc[sell_signals_t, 'close'], color='r', marker='v', label='Sell')

        # Set up the wallet and bitcoin_wallet text
        ax.text(0.1, 0.85, f"Wallet: {self.wallet:.2f}", transform=ax.transAxes, fontsize=15, verticalalignment='top')
        ax.text(0.1, 0.95, f"Bitcoin Wallet: {self.btc:.8f}", transform=ax.transAxes, fontsize=15, verticalalignment='top')



        ax.legend(loc='best')


    def animate_pheromone_maps(self):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        fig.suptitle("Pheromone Maps")
        ax1.set_title("Buy Pheromones")
        ax2.set_title("Sell Pheromones")

        num_literals = len(self.literal_dict) * 2
        max_timesteps = len(self.historical_buy_pheromones)

        def update(frame):
            ax1.clear()
            ax2.clear()

            buy_pheromones = np.array(self.historical_buy_pheromones[frame]).reshape(num_literals // 2, 2)
            sell_pheromones = np.array(self.historical_sell_pheromones[frame]).reshape(num_literals // 2, 2)

            vmin = min(np.min(buy_pheromones), np.min(sell_pheromones))
            vmax = max(np.max(buy_pheromones), np.max(sell_pheromones))

            ax1.imshow(buy_pheromones, cmap="coolwarm", vmin=vmin, vmax=vmax)
            ax2.imshow(sell_pheromones, cmap="coolwarm", vmin=vmin, vmax=vmax)

            ax1.set_title(f"Buy Pheromones (t={frame})")
            ax2.set_title(f"Sell Pheromones (t={frame})")

        ani = FuncAnimation(fig, update, frames=range(max_timesteps), repeat=True)
        plt.show()


    def draw_single_pheromone_map(self, buy_pheromones, sell_pheromones):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        fig.suptitle("Single Pheromone Map")
        ax1.set_title("Buy Pheromones")
        ax2.set_title("Sell Pheromones")

        num_literals = len(self.literal_dict) * 2

        buy_pheromones = np.array(buy_pheromones).reshape(num_literals // 2, 2)
        sell_pheromones = np.array(sell_pheromones).reshape(num_literals // 2, 2)

        vmin = min(np.min(buy_pheromones), np.min(sell_pheromones))
        vmax = max(np.max(buy_pheromones), np.max(sell_pheromones))

        cax1 = ax1.imshow(buy_pheromones, cmap="coolwarm", vmin=vmin, vmax=vmax)
        cax2 = ax2.imshow(sell_pheromones, cmap="coolwarm", vmin=vmin, vmax=vmax)

        fig.colorbar(cax1, ax=ax1)
        fig.colorbar(cax2, ax=ax2)

        plt.show()
        print(buy_pheromones)
        print(sell_pheromones)
        print(list(self.literal_dict.keys()))


    


    def plot_run(self, buy_dnf_formula, sell_dnf_formula):
        df = self.data

        for t in range(1, self.period):
            buy_t = self.buy_pulse(t, buy_dnf_formula)
            buy_prev = self.buy_pulse(t - 1, buy_dnf_formula)
            sell_t = self.sell_pulse(t, sell_dnf_formula)
            sell_prev = self.sell_pulse(t - 1, sell_dnf_formula)

            buy_trigger = buy_t and not buy_prev and not (sell_t and not sell_prev)
            sell_trigger = sell_t and not sell_prev and not (buy_t and not buy_prev)

            if not self.bought and buy_trigger:

                self.buy_signals.append(t)
                self.bought = True
            elif self.bought and sell_trigger:

                self.sell_signals.append(t)
                self.bought = False
        
        if self.bought:
            self.sell_signals.append(self.period - 1)

        fig = go.Figure()
        print(self.buy_signals)
        print(self.sell_signals)
        # Add OHLCV data to the figure
        fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name='OHLCV'))

        # Add buy signals to the figure
        fig.add_trace(go.Scatter(x=df.iloc[self.buy_signals].index,
                                y=df.iloc[self.buy_signals]['close'],
                                mode='markers',
                                marker=dict(color='green', size=10),
                                name='Buy'))

        # Add sell signals to the figure
        fig.add_trace(go.Scatter(x=df.iloc[self.sell_signals].index,
                                y=df.iloc[self.sell_signals]['close'],
                                mode='markers',
                                marker=dict(color='red', size=10),
                                name='Sell'))

        # Configure the layout
        fig.update_layout(title='OHLCV Data with Buy and Sell Signals',
                        yaxis_title='Price',
                        xaxis_title='Time',
                        xaxis_rangeslider_visible=False)

        fig.show()

        self.reset()


    def plot_ants_over_time(self):
        ant_profits = self.ant_profits
        num_iterations = len(ant_profits)
        num_ants = len(ant_profits[0])
        
        # Prepare the x-axis for the plot
        x = list(range(1, num_iterations + 1))
        
        # Plot each ant's profit over time
        for ant in range(num_ants):
            y = [ant_profits[iteration][ant] for iteration in range(num_iterations)]
            plt.plot(x, y, color='blue', alpha=0.3)
        
        # Calculate and plot the average profits
        avg_profits = np.mean(ant_profits, axis=1)
        plt.plot(x, avg_profits, color='red', label='Average')
        
        # Customize the plot
        plt.xlabel('Iteration')
        plt.ylabel('Profit')
        plt.title('Ant Performance Over Time')
        plt.legend()
        plt.grid(True)
        
        # Show the plot
        plt.show()

    def animate_ants_over_time(self):
        # Create a figure and axis
        ant_profits = self.ant_profits
        iterations = len(ant_profits)
        num_ants = len(ant_profits[0])

        
        fig, ax = plt.subplots()

        # Prepare the x-axis for the plot
        x = list(range(1, iterations + 1))

        # Prepare lines for each ant's profit and the average profit over time
        lines = [ax.plot(x, [ant_profits[0][ant]] * iterations, color='blue', alpha=0.3)[0] for ant in range(num_ants)]
        avg_line = ax.plot(x, [np.mean(ant_profits[0])] * iterations, color='red', label='Average')[0]

        # Customize the plot
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Profit')
        ax.set_title('Ant Performance Over Time')
        ax.legend()
        ax.grid(True)

        # Update function for the animation
        def update(i):
            iterations = len(self.ant_profits)
            # Update each ant's line
            for ant, line in enumerate(lines):
                line.set_ydata([self.ant_profits[j][ant] for j in range(i+1)] + [None] * (iterations - i - 1))

            # Update the average line
            avg_line.set_ydata([np.mean(self.ant_profits[j]) for j in range(i+1)] + [None] * (iterations - i - 1))

            return lines + [avg_line]

        # Create the animation
        anim = FuncAnimation(fig, update, frames=iterations, interval=200, blit=True)

        plt.show()
        



# Initialize the trading bot with the data
# bot = TradingBot(data)
# bot.test_indicators()


# bot.optimise()
# print(bot.best_buy_dnf_formula)
# print(bot.best_sell_dnf_formula)
# print(bot.run(bot.best_buy_dnf_formula, bot.best_sell_dnf_formula))

# buy = [{('EaseOfMovementIndicator_7_21', True), ('ROCIndicator_7_21', False), ('ForceIndexIndicator_7_21', False)}, {('RSIIndicator_7_21', True)}, {('VortexIndicator_7_21', True)}]
# sell = [{('DonchianChannel_7_21', True), ('VortexIndicator_7_21', True)}, {('BollingerBands_7_21', False)}]
# print(bot.run(buy, sell))


# Optimize the bot's parameters and logical expressions
# bot.optimise()

# Run the trading bot
# print(bot.run(bot.best_parameters))






 
from tradingbot import TradingBot
from aco_optimiser import Ant
import os
import subprocess
import pandas as pd

if __name__ == "__main__":
    
    if not os.path.isfile('aco_data.csv'):
        subprocess.run(['python', 'get_data.py'])

    data = pd.read_csv("aco_data.csv")


    window_sizes = [5, 20, 40]
    best_average_literals = 2
    best_average_clauses = 3
    best_alpha = 2
    best_beta = 1
    best_evaporation_rate = 0.2

    best_params = (best_alpha, best_beta, best_evaporation_rate, best_average_literals, best_average_clauses, 100, 100) 
    data_points = len(data)
    train_ratio = 0.8

    train_size = int(data_points * train_ratio)
    test_size = data_points - train_size

    train_data = data[:train_size]
    test_data = data[train_size:].reset_index(drop=True)


    train_bot = TradingBot(train_data, window_sizes, best_params)
    test_bot = TradingBot(test_data, window_sizes, best_params)

    train_bot.optimise()

    best_ant = train_bot.best_ants[-1]
    train_bot.draw_single_pheromone_map(train_bot.historical_buy_pheromones[-1], train_bot.historical_sell_pheromones[-1])
    train_bot.plot_ants_over_time()
    train_bot.plot_run(best_ant.buy_dnf, best_ant.sell_dnf)
    test_bot.plot_run(best_ant.buy_dnf, best_ant.sell_dnf)
    print(train_bot.run(best_ant.buy_dnf, best_ant.sell_dnf))
    print(test_bot.run(best_ant.buy_dnf, best_ant.sell_dnf))
    print(best_ant.buy_dnf)
    print(best_ant.sell_dnf)


    # best_buy = [{('StochasticOscillator_5_40', False), ('StochasticOscillator_20_40', True), ('VortexIndicator_5_20', False), ('UlcerIndex_20_40', False), ('EMAIndicator_20_40', True), ('ChaikinMoneyFlowIndicator_20_40', False)}, {('ForceIndexIndicator_20_40', True), ('BollingerBands_5_40', True), ('DonchianChannel_5_40', False)}]
    # best_sell = [{('BollingerBands_5_20', False), ('KAMAIndicator_5_20', True)}]
    # # print(test_bot.run(best_buy, best_sell))
    # train_bot.animate_run(best_buy, best_sell)
    
    # test_bot.plot_run(best_buy, best_sell)

    # print(train_bot.plot_ants_over_time())



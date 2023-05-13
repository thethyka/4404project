# 4404project
Automated trading bot algorithm based on daily candle data for AUD bitcoin history.


# aco_method.py
Will need to run get_data.py to get a csv file of all data points of interest before running aco_method.py

# get_data.py
Fetches current AUD/BTC data and puts into a csv

# aco_optimiser.py
Optimiser class that uses ACO, employed by our run_bot algorithm

# run_bot.py
Main algorithm that runs the bot

## TO RUN:
clone the repo
create a virtual environment, run:
python -m venv myenv
activate your myenv (see https://docs.python.org/3/library/venv.html)
pip install -r requirements.txt
then run visualiser.py to see everything in action

(in line 23 of visualiser.py, adjust the last 2 numbers num_ants and num_iterations to determine how fast you want the algorithm to run)
(do 10 and 10 for a quick view, 100 and 100 for a comprehensive algorithm)

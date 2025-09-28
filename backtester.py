import numpy as np
import matplotlib.pylab as plt
from market_simulator import evaluate_oos, generate_data

seeds = oos_seeds = list(range(100, 150))
oos_data = [generate_data(T=1000, seed=s) for s in oos_seeds]
print(len(oos_data), "out-of-sample data sets generated")
best_params = input("Input params to backtest:" )
best_params = tuple(map(float, best_params.strip("()").split(",")))
oos_stats, pnls = evaluate_oos(best_params, oos_data)
print("Out-of-sample validation:", oos_stats)

plt.hist(pnls, bins=20, edgecolor="k")
plt.title("Out-of-sample PnL distribution")
plt.xlabel("PnL")
plt.ylabel("Frequency")
plt.show()


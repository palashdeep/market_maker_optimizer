import numpy as np
import matplotlib.pylab as plt
from market_simulator import evaluate, generate_data

seeds = oos_seeds = list(range(1100, 1200))
oos_data = [generate_data(T=1000, seed=s) for s in oos_seeds]
print(len(oos_data), "out-of-sample data sets generated")
best_params = input("Input params to backtest:" )
best_params = tuple(map(float, best_params.strip("()").split(",")))
oos_stats = evaluate(best_params, oos_data)
net_pnls = oos_stats["structural_pnls_array"]
rep_pnls = oos_stats["final_pnls_array"]
print("Out-of-sample validation:", oos_stats)

plt.hist(net_pnls, bins=20, edgecolor="k")
plt.title("Out-of-sample net PnL distribution")
plt.xlabel("PnL")
plt.ylabel("Frequency")
plt.show()

plt.hist(rep_pnls, bins=20, edgecolor="k")
plt.title("Out-of-sample rep PnL distribution")
plt.xlabel("PnL")
plt.ylabel("Frequency")
plt.show()

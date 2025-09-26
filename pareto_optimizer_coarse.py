import numpy as np
import matplotlib.pyplot as plt

from market_simulator import generate_data, simulate_market, evaluate

seeds = [37 + 3*i for i in range(10)]
all_data = [generate_data(T=1000, seed=s) for s in seeds]

# Coarse optimization using grid search
param_grid = [
    (k, alpha, hth, hsize)
    for k in [0.5, 1.0, 2.0, 3.0]
    for alpha in [0.0, 0.05, 0.1, 0.2]
    for hth in [10, 20, 50, 100]
    for hsize in [5, 10, 20, 40]
]

results = []
for params in param_grid:
    stats = evaluate(params, all_data)
    results.append({
        "params": params,
        "mean_pnl": stats["mean_pnl"],
        "mean_inv": stats["mean_inv"],
        "mean_hedge_cost": stats["mean_hedge_cost"]
    })

# Scatter: x=risk, y=pnl, color=hedge cost
x = [r["mean_inv"] for r in results]
y = [r["mean_pnl"] for r in results]
c = [r["mean_hedge_cost"] for r in results]

plt.figure(figsize=(8,6))
sc = plt.scatter(x, y, c=c, cmap="viridis", s=60, edgecolor='k', alpha=0.9)
plt.colorbar(sc, label="Mean hedge cost")
plt.xlabel("Inventory risk (mean std inv)")
plt.ylabel("Mean net PnL")
plt.title("Pareto cloud (color = mean hedge cost)")

# simple Pareto frontier
def pareto_frontier_from_results(results):
    # we sort by increasing risk
    points = sorted([(r["mean_inv"], r["mean_pnl"], r["hedge_cost"], r["params"]) for r in results], key=lambda x: x[0])
    frontier = []
    best_pnl = -float('inf')
    for risk, pnl, hedge, params in points:
        # keep only points that improve PnL
        if pnl > best_pnl:
            frontier.append((risk, pnl, hedge, params))
            best_pnl = pnl
    return frontier

frontier = pareto_frontier_from_results(results)
fx, fy, _ = zip(*frontier)
plt.plot(fx, fy, color='red', lw=2, marker='o', label="Pareto frontier")
plt.legend()
plt.show()

# Find weighted best point on Pareto frontier
weights = [0.7, 0.2, 0.1]   # weights for [pnl, inv, hedge_cost], sum to 1
best_score = -float('inf')
best_result = None
best_params = None

for r in frontier:
    risk, pnl, hedge_cost, params = r
    score = weights[0]*pnl - weights[1]*risk - weights[2]*hedge_cost
    if score > best_score:
        best_score = score
        best_result = (risk, pnl, hedge_cost)
        best_params = params

print("Weighted best params:", best_params)
print("metrics:", best_result[1], best_result[0], best_result[2])
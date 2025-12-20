import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from market_simulator import generate_data, evaluate

# simple Pareto frontier
def pareto_frontier_from_results(results):
    # we sort by increasing risk
    points = sorted([(r["mean_inv"], r["mean_pnl"], r["mean_hedge_cost"], r["params"]) for r in results], key=lambda x: x[0])
    frontier = []
    best_pnl = -float('inf')
    for risk, pnl, hedge, params in points:
        # keep only points that improve PnL
        if pnl > best_pnl:
            frontier.append((risk, pnl, hedge, params))
            best_pnl = pnl
    return frontier

# ---- objective used by optimizer (scalar) ----
def robust_objective(params, all_data,
                     w_var=0.02,   # penalty on PnL variance (tune)
                     w_inv=0.01):  # penalty on inventory risk
    stats = evaluate(params, all_data)
    # maximize mean_structural_pnl, but optimizer might minimize -objective. We'll return scalar score to maximize.
    score = stats["mean_structural_pnl"] - w_var * stats["var_final_pnl"] - w_inv * stats["mean_inv"]
    return score, stats

# Data Setup 
seeds = [37 + 3*i for i in range(50)]
all_data = [generate_data(T=1000, seed=s) for s in seeds]

for i, y_array in enumerate(all_data):
        # If x_values are provided, use them. Otherwise, use default indices.
    plt.plot(y_array, label=f'Line {i+1}')

plt.title('Multiple Line Plots from NumPy Arrays')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Coarse optimization using grid search
param_grid = [
    (k, alpha, hth, hsize)
    for k in [0.5, 1.0, 2.0, 2.5, 3.0, 4.0]
    for alpha in [0.025, 0.05, 0.075, 0.1, 0.2, 0.4]
    for hth in [10, 20, 50, 100]
    for hsize in [5, 10, 20, 40]
]

results = []
count = 0
res = {"k":[],"alpha":[],"hedge_thresh":[],"hedge_size":[],"t_stat":[]}
best = None
best_score = -float('inf')
for params in param_grid:
    score, stats = robust_objective(params, all_data)
    if score > best_score:
        best_score = score
        best = (params, stats, score)
    results.append({
        "params": params,
        "mean_pnl": stats["mean_structural_pnl"],
        "mean_inv": stats["mean_inv"],
        "mean_hedge_cost": stats["mean_hedge_cost"],
        "t_stat": stats["t_stat"],
    })
    res["k"].append(params[0])
    res["alpha"].append(params[1])
    res["hedge_thresh"].append(params[2])
    res["hedge_size"].append(params[3])
    res["t_stat"].append(stats["t_stat"])
    if stats["t_stat"] >= 2.0:
        count += 1
print("Percentage of significant results (t>=2): %.2f%%" % (100*count/len(param_grid)))
df = pd.DataFrame(res)
pt = df.pivot_table(index="k", columns="alpha", values="t_stat", aggfunc="mean")
sns.heatmap(pt, cmap="viridis", annot=False)

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

frontier = pareto_frontier_from_results(results)
fx, fy, _, _ = zip(*frontier)
plt.plot(fx, fy, color='red', lw=2, marker='x', label="Pareto frontier")
plt.legend()
plt.show()

# Find weighted best point on Pareto frontier
weights = [0.8, 0.19, 0.01]   # weights for [pnl, inv, hedge_cost], sum to 1
best_score = -float('inf')
best_result = None
best_params = None

# for r in frontier:
#     risk, pnl, hedge_cost, params = r
#     score = weights[0]*pnl - weights[1]*risk - weights[2]*hedge_cost
#     if score > best_score:
#         best_score = score
#         best_result = (risk, pnl, hedge_cost)
#         best_params = params
for r in results:
    risk, pnl, hedge_cost, params = r['mean_inv'], r['mean_pnl'], r['mean_hedge_cost'], r['params']
    score = weights[0]*pnl - weights[1]*risk - weights[2]*hedge_cost
    if score > best_score:
        best_score = score
        best_result = (risk, pnl, hedge_cost)
        best_params = params

print("Weighted best params:", best_params)
print("metrics:", best_result[1], best_result[0], best_result[2])
print("Robust best params:", best[0])
print("Robust metrics:", best[1]["mean_structural_pnl"], best[1]["mean_final_pnl"], best[1]["mean_inv"], best[1]["mean_hedge_cost"], best[1]["var_final_pnl"])
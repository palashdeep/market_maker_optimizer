import numpy as np
import pandas as pd

from simulator import generate_data, simulate_path
from market_maker import MarketMaker


def evaluate_params(params, all_data):
    """
    Evaluates params across multiple price runs
    
    Returns summary statistics
    """
    k, alpha, hedge_threshold, hedge_size = params

    controlled_pnls = []
    inv_vols = []
    hedge_costs = []

    for prices in all_data:
        mm = MarketMaker()
        out = simulate_path(prices, mm, k, alpha, hedge_threshold, hedge_size)

        controlled_pnls.append(out['controlled_pnl'])
        inv_vols.append(np.std(out["inventory_path"]))
        hedge_costs.append(out['hedge_cost'])

    controlled_pnls = np.array(controlled_pnls)
    inv_vols = np.array(inv_vols)

    t_stat = np.mean(controlled_pnls) / (np.std(controlled_pnls, ddof=1) / np.sqrt(len(all_data)))

    return {
        "params": params,
        "mean_controlled_pnl": controlled_pnls.mean(),
        "std_controlled_pnl": controlled_pnls.std(ddof=1),
        "t_stat": t_stat,
        "mean_inv_vol": inv_vols.mean(),
        "mean_hedge_cost": hedge_costs.mean()
    }

def run_parameter_sweep(param_grid, seeds):
    """
    Runs parameter sweep on common random paths
    """

    all_data = [generate_data(T=5000, seed=s) for s in seeds]
    
    results = []
    for params in param_grid:
        stats = evaluate_params(params, all_data)
        results.append(stats)

    return pd.DataFrame(results)

def pareto_front(df):
    """
    Compute Pareto-efficient points (PnL ↑, inv risk ↓).
    """

    df = df.sort_values(["mean_inv_vol"])
    frontier = []

    best_pnl = -np.inf
    for _, row in df.iterrows():
        if row["mean_controlled_pnl"] > best_pnl:
            frontier.append(row)
            best_pnl = row["mean_controlled_pnl"]

    return pd.DataFrame(frontier)

def significant_params(df, threshold=2.0):
    """
    Return parameter sets whose controlled PnL is
    statistically distinguishable from noise.
    """
    return df[df["t_stat"] >= threshold]
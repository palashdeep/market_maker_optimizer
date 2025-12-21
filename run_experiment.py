import numpy as np
import matplotlib.pyplot as plt

from optimizer import run_parameter_sweep, pareto_front, significant_params
from simulator import generate_data

def main():

    param_grid = [
        (k, alpha, hth, hsz)
        for k in [0.5, 1.0, 2.0]
        for alpha in [0.01, 0.05, 0.1]
        for hth in [20, 50]
        for hsz in [5, 10]
    ]

    seeds = range(1,51)
    oos_seeds = range(1001,1051)

    df = run_parameter_sweep(param_grid, seeds)

    significant = significant_params(df, threshold=2.0)
    print(f"Significant parameter sets (t >= 2.0): {len(significant)} / {len(df)}")

    oos_df = run_parameter_sweep(significant[["params"]].tolist(), oos_seeds)

    pareto_df = pareto_front(oos_df)

    plt.scatter(oos_df["mean_inv_vol"], 
                oos_df["mean_controlled_pnl"], 
                alpha=0.3, 
                label="OOS params"
    )
    
    plt.plot(pareto_df["mean_inv_vol"], 
                pareto_df["mean_controlled_pnl"], 
                color='red',
                linewidth=2,
                label="Pareto front (OOS)"
    )

    plt.xlabel("Inventory Risk")
    plt.ylabel("Mean Controlled PnL")
    plt.legend()
    plt.title("OOS Tradeoff: Controlled PnL vs Inventory Risk")
    plt.show()

if __name__ == "__main__":
    main()

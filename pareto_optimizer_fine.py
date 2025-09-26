import numpy as np
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem

from market_simulator import generate_data, simulate_market, evaluate


seeds = [37 + 3*i for i in range(10)]
all_data = [generate_data(T=1000, seed=s) for s in seeds]
hedge_cost_cap = 50.0   # example cap (same units as price*units)

# Fine optimization using NSGA-II and constraint on hedge cost
class MMProblem(ElementwiseProblem):
    def __init__(self, all_data, hedge_cost_cap):
        # x = [k, alpha, hedge_threshold, hedge_size]
        super().__init__(n_var=4, n_obj=2, n_constr=1,
                         xl=np.array([0.1, 0.0, 1.0, 1.0]),
                         xu=np.array([5.0, 2.0, 200.0, 200.0]))
        self.all_data = all_data
        self.hedge_cost_cap = hedge_cost_cap

    def _evaluate(self, x, out, *args, **kwargs):
        k, alpha, hedge_threshold, hedge_size = x
        stats = evaluate((k, alpha, hedge_threshold, hedge_size), self.all_data)
        mean_pnl = stats["mean_pnl"]
        mean_inv = stats["mean_inv"]
        mean_hedge_cost = stats["mean_hedge_cost"]

        # Objectives: minimize negatives so we can minimize both by pymoo convention
        # We want to maximize PnL → minimize -mean_pnl
        out["F"] = [-mean_pnl, mean_inv]
        # Constraint: mean_hedge_cost <= hedge_cost_cap  =>  g = mean_hedge_cost - cap  <= 0
        out["G"] = [mean_hedge_cost - self.hedge_cost_cap]

# Run NSGA-II
problem = MMProblem(all_data, hedge_cost_cap)
algorithm = NSGA2(pop_size=40)
res = minimize(problem, algorithm, ('n_gen', 80), verbose=True)
# res.X (Pareto designs), res.F (objectives)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

# Lexicographic selection of best design on Pareto front
pareto_pnls = -res.F[:,0]   # remember we minimized negative PnL
pareto_invs = res.F[:,1]
pareto_hedges = res.G[:,0] + hedge_cost_cap  # recover mean hedge costs

mean_pnls = np.array(pareto_pnls)
mean_invs = np.array(pareto_invs)
mean_hedges = np.array(pareto_hedges)

# pick top candidates by pnl (highest preferrence)
best_pnl = np.max(mean_pnls)
tol = 0.01 * abs(best_pnl)  # 1% tolerance
top_idx = np.where(mean_pnls >= best_pnl - tol)[0]

# step 2: pick those with minimal inventory (second preference)
if len(top_idx) > 1:
    invs_top = mean_invs[top_idx]
    min_inv = np.min(invs_top)
    top_idx = top_idx[np.where(invs_top == min_inv)[0]]

# step 3: pick minimal hedge cost (least preference)
if len(top_idx) > 1:
    hed_top = mean_hedges[top_idx]
    min_hed = np.min(hed_top)
    top_idx = top_idx[np.where(hed_top == min_hed)[0]]

best_index = int(top_idx[0])
best_result = [mean_pnls[best_index], mean_invs[best_index], mean_hedges[best_index]]
print("Lexicographic best params:", res.X[best_index])
print("metrics:", best_result[0], best_result[1], best_result[2])
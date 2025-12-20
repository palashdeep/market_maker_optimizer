import numpy as np
import matplotlib.pylab as plt
from market_maker_advanced import MarketMakerAdv

def generate_data(T=5000, seed=36, sigma=0.2, mu=0.0, S0=100, steps_per_year=25200):
    """
    Simulate price path using geometric Brownian motion.
    sigma, mu are annualized.
    steps_per_year defines the time resolution (e.g. 252 for daily, 25200 for 100 ticks/day)
    """
    np.random.seed(seed)
    dt = 1 / steps_per_year
    S = np.empty(T+1)
    S[0] = S0
    for t in range(T):
        z = np.random.randn()
        S[t+1] = S[t] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return S

def simulate_market(data, mm, k, alpha, hedge_threshold, hedge_size):
    """Simulates market making, returns PnL and inventory std dev"""
    
    # Simulation params
    T = len(data)
    order_prob = 0.3        # prob of order arrival each step
    inv = 0           # instantaneous inventory
    c = 0        # instantaneous cash

    for t in range(1, T):
        # Price return
        ret = np.log(data[t]/data[t-1])
        mid = data[t]
        # get spread from MM
        spread = mm.get_spread(k, ret)
        
        # Order arrivals
        if np.random.rand() < order_prob:
            side = np.random.choice(["buy", "sell"])
            size = np.random.randint(1, 5)  # random order size between 1 and 99
            c, inv = mm.trade(mid, spread, side, size, alpha, inv, c) # mm trades at quoted price
        
        # Hedging
        c, inv = mm.hedge(t, mid, inv, c, hedge_threshold, hedge_size)

    # Fetch Results
    value, hedge_cost, hedge_volume, num_hedges, revenue, inv = mm.report_MM_stats(data)

    structural_pnl = revenue - hedge_cost
    final_pnl = structural_pnl + inv[-1] * data[-1]

    # print("Final PnL: %.2f, Structural PnL: %.2f, PnL from run: %.2f" % (final_pnl, structural_pnl, value))

    return {
        "pnl": value,
        "final_pnl": final_pnl,
        "structural_pnl": structural_pnl,
        "inv": inv,
        "total_hedge_cost": hedge_cost,
        "total_spread_revenue": revenue,
        "num_hedges": num_hedges,
        "total_hedge_volume": hedge_volume
    }

def evaluate(params, all_data):
    k, alpha, hedge_threshold, hedge_size = params
    net_pnls, final_pnls, structural_pnls, inv_vol, hedge_costs, revenues = [], [], [], [], [], []
    n = len(all_data)
    for data in all_data:
        mm = MarketMakerAdv() # initialise with default params
        out = simulate_market(data, mm, k, alpha, hedge_threshold, hedge_size)
        net_pnls.append(out['pnl'])  # reported pnl from MM
        final_pnls.append(out['final_pnl'])
        structural_pnls.append(out['structural_pnl'])
        inv_vol.append(np.std(out['inv']))
        hedge_costs.append(out['total_hedge_cost'])
        revenues.append(out['total_spread_revenue'])

    # plt.hist(pnls, bins=20, edgecolor="k")
    # plt.title("Net PnL distribution")
    # plt.xlabel("PnL")
    # plt.ylabel("Frequency")
    # plt.show()
    
    pnls = np.array(net_pnls)
    final_pnls = np.array(final_pnls)
    structural_pnls = np.array(structural_pnls)
    invs = np.array(inv_vol)
    hedge_costs = np.array(hedge_costs)
    spread_revenues = np.array(revenues)
    t_stat = np.mean(structural_pnls) / (np.std(structural_pnls, ddof=1) / np.sqrt(n))
    if t_stat >= 2.0:
    # print("Mean Net PnL: %.2f, Mean Structural PnL: %.2f, Mean Final PnL: %.2f, Net PnL Std Dev: %.2f, Structural PnL Std Dev: %.2f, Final PnL Std Dev: %.2f" % (np.mean(pnls), np.mean(structural_pnls), np.mean(final_pnls), np.std(pnls, ddof=1), np.std(structural_pnls, ddof=1), np.std(final_pnls, ddof=1)))
        print("t-stat: %d, Mean Structural PnL: %.2f, Mean Final PnL: %.2f, Final PnL Std Dev: %d, Inv_vol: %.2f, Hedge_costs: %.2f" % (np.mean(structural_pnls) / (np.std(structural_pnls, ddof=1) / np.sqrt(n)), np.mean(structural_pnls), np.mean(pnls), np.std(pnls, ddof=1)//np.mean(pnls), np.mean(invs), np.mean(hedge_costs)))

    return {
        "t_stat": t_stat,
        "mean_final_pnl": float(np.mean(pnls)),
        "std_final_pnl": float(np.std(pnls, ddof=1)),
        "mean_inv": float(np.mean(invs)),
        "mean_hedge_cost": float(np.mean(hedge_costs)),
        "mean_spread_revenue": float(np.mean(spread_revenues)),
        "mean_structural_pnl": float(np.mean(structural_pnls)),
        "var_final_pnl": float(np.var(pnls, ddof=1)),
        "structural_pnls_array": structural_pnls,
        "final_pnls_array": pnls
    }

# def evaluate_oos(params, all_data):
#     k, alpha, hedge_threshold, hedge_size = params
#     net_pnls, inv_vol, hedge_costs, revenues = [], [], [], []
#     for data in all_data:
#         mm = MarketMakerAdv() # initialise with default params
#         out = simulate_market(data, mm, k, alpha, hedge_threshold, hedge_size)
#         net_pnls.append(out['pnl'])  # reported pnl from MM
#         inv_vol.append(np.std(out['inv']))
#         hedge_costs.append(out['total_hedge_cost'])
#         revenues.append(out['total_spread_revenue'])

#     pnls = np.array(pnls)
#     invs = np.array(invs)
#     hedge_costs = np.array(hedge_costs)
#     spread_revenues = np.array(spread_revenues)

#     structural_pnls = spread_revenues - hedge_costs

#     return {
#         "mean_final_pnl": float(np.mean(pnls)),
#         "std_final_pnl": float(np.std(pnls, ddof=1)),
#         "mean_inv": float(np.mean(invs)),
#         "mean_hedge_cost": float(np.mean(hedge_costs)),
#         "mean_spread_revenue": float(np.mean(spread_revenues)),
#         "mean_structural_pnl": float(np.mean(structural_pnls)),
#         "var_final_pnl": float(np.var(pnls, ddof=1)),
#         "structural_pnls_array": structural_pnls,
#         "final_pnls_array": pnls
#     }
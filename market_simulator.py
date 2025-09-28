import numpy as np
from market_maker_advanced import MarketMakerAdv

def generate_data(T=1000, seed=36, sigma=0.02, mu=0.0, S0=100):
    """Returns simulated price data using simple GBM model"""

    np.random.seed(seed)
    dt = 1
    S = [S0]
    for t in range(T):
        S.append(S[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.randn()))
    return np.array(S)

def simulate_market(data, mm, k, alpha, hedge_threshold, hedge_size):
    """Simulates market making, returns PnL and inventory std dev"""
    
    # Simulation params
    T = len(data)
    dt = 1              # time step size
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
    value, hedge_cost, hedge_volume, num_hedges, revenue, std_inv = mm.report_MM_stats(data)

    return {
        "net_pnl": value,
        "inv_vol": std_inv,
        "total_hedge_cost": hedge_cost,
        "total_spread_revenue": revenue,
        "num_hedges": num_hedges,
        "total_hedge_volume": hedge_volume
    }

def evaluate(params, all_data):
    k, alpha, hedge_threshold, hedge_size = params
    pnls, invs, hedge_costs = [], [], []
    for data in all_data:
        mm = MarketMakerAdv() # initialise with default params
        out = simulate_market(data, mm, k, alpha, hedge_threshold, hedge_size)
        pnls.append(out['net_pnl'])
        invs.append(out['inv_vol'])
        hedge_costs.append(out['total_hedge_cost'])
    return {
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "mean_inv": float(np.mean(invs)),
        "mean_hedge_cost": float(np.mean(hedge_costs))
    }

def evaluate_oos(params, all_data):
    k, alpha, hedge_threshold, hedge_size = params
    pnls, invs, hedge_costs = [], [], []
    for data in all_data:
        mm = MarketMakerAdv() # initialise with default params
        out = simulate_market(data, mm, k, alpha, hedge_threshold, hedge_size)
        pnls.append(out['net_pnl'])
        invs.append(out['inv_vol'])
        hedge_costs.append(out['total_hedge_cost'])
    return {
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "mean_inv": float(np.mean(invs)),
        "mean_hedge_cost": float(np.mean(hedge_costs))
    }, pnls
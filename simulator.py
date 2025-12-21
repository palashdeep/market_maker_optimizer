import numpy as np
from market_maker import MarketMaker

def generate_data(T=5000, seed=36, sigma=0.2, mu=0.0, S0=100, steps_per_year=25200):
    """
    Simulate price path using geometric Brownian motion.
    sigma, mu are annualized.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / steps_per_year

    S = np.empty(T+1)
    S[0] = S0

    for t in range(T):
        z = rng.standard_normal()
        S[t+1] = S[t] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    
    return S

def simulate_path(prices, mm: MarketMaker, k, alpha, hedge_threshold, hedge_size, order_prob = 0.5, max_order_size = 5, seed=None):
    """
    Simulates a single market making path
    Returns 
    - final cash 
    - final inventory
    - inventory time series
    - controlled PnL components
    """
    rng = np.random.default_rng(seed)
    
    inv, cash = 0, 0
    inv_path = [inv]

    for t in range(1, len(prices)):
        mid = prices[t]
        ret = np.log(prices[t]/prices[t-1])
        
        #update vol estimate and quote
        mm.update_volatility(ret)
        bid, ask = mm.quote(mid, k, inv, alpha)

        if rng.random() < order_prob:
            side = rng.choice(["buy", "sell"])
            size = rng.integers(1, max_order_size + 1)

            price = ask if side == "buy" else bid
            cash, inv = mm.execute_trade(side, price, size, inv, cash, mid)

        cash, inv = mm.hedge(mid, inv, cash, hedge_threshold, hedge_size)

        inv_path.append(inv)

    return {
        "final_cash": cash,
        "final_inventory": inv,
        "inventory_path": np.array(inv_path),
        "spread_revenue": mm.spread_revenue,
        "hedge_cost": mm.hedge_cost,
        "controlled_pnl": mm.controlled_PnL()
    }
import numpy as np

class MarketMaker():
    """
    Market maker with:
    - volatility-adaptive spreads
    - inventory-based quote skew
    - volatility-scaled inventory hedging
    """

    def __init__(self, lam=0.94, sigma=0.2, slippage=0.01, impact=0.0005, max_shift=2.0):
        # Volatility estimation
        self.lam = lam       
        self.sigma_ref = sigma 
        self.sigma2_ewma = sigma**2

        # Execution cost model
        self.slippage = slippage
        self.impact = impact      

        # Inventory control
        self.max_shift = max_shift

        # Accounting
        self.spread_revenue = 0.0
        self.hedge_cost = 0.0

    def update_volatility(self, ret):
        """Updates EWMA volatility estimate"""

        self.sigma2_ewma = self.lam * self.sigma2_ewma + (1 - self.lam) * (ret**2)
        return np.sqrt(self.sigma2_ewma)
    
    def quote(self, mid, k, inv, alpha):
        """
        Returns bid/ask quotes
        Spread depends on volatility; skew depends on inventory
        """

        sigma_ewma = np.sqrt(self.sigma2_ewma)
        spread = k * sigma_ewma

        skew = np.clip(-alpha * inv, -self.max_shift, self.max_shift)
        
        bid = mid - spread/2 + skew
        ask = mid + spread/2 + skew
        return bid, ask

    def execute_trade(self, side, price, size, inv, cash, mid):
        """Executes client trade and updates PnL"""

        if side == "buy":
            # client buys from MM at ask
            cash += price * size  
            inv -= size
            self.spread_revenue += (price - mid) * size
        else:
            # client sells to MM at bid
            cash -= price * size
            inv += size
            self.spread_revenue += (mid - price) * size
        
        return inv, cash
    
    def hedge(self, mid, inv, cash, base_threshold, hedge_size):
        """Volatility-scaled inventory hedging"""
        
        sigma = np.sqrt(self.sigma2_ewma)
        hedge_threshold = base_threshold * self.sigma_ref / max(sigma, 1e-8)

        if abs(inv) > hedge_threshold:
            units = -np.sign(inv) * min(abs(inv), hedge_size)
            exec_price = mid + np.sign(units) * (self.slippage + self.impact * abs(units))
            
            cash -= exec_price * units
            inv += units
            self.hedge_cost += abs(exec_price - mid) * abs(units)

        return inv, cash

    def controlled_PnL(self):
        """spread revenue minus hedge cost"""

        return self.spread_revenue - self.hedge_cost

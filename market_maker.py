import numpy as np

class MarketMakerAdv():
    def __init__(self, lam=0.94, sigma=0.2, slippage=0.01, eta=0.0005):
        self.lam = lam       # EWMA decay factor
        self.lambda_h = 2.0  # hedging intensity
        self.sigma_ref = sigma # reference volatility for hedging probability
        self.sigma2_ewma = sigma**2  # initial variance guess
        self.gamma = slippage     # slippage per unit hedged
        self.eta = eta           # market impact per unit hedged
        self.max_shift = 2.0      # max bid/ask shift due to inventory
        self.total_spread_revenue = 0.0    # revenue earned from client trades
        self.total_hedge_cost = 0.0        # cost incurred by external hedges
        self.total_hedge_volume = 0
        self.num_hedges = 0
        self.spreads = []
        self.inventory = []
        self.cash = []
        self.vols = []
        self.hedge_events = []  # list of (time, units, exec_price) for hedges
    
    def get_spread(self, k, ret):
        """Returns spread based on EWMA volatility"""

        self.sigma2_ewma = self.lam * self.sigma2_ewma + (1 - self.lam) * (ret**2)
        sigma_ewma = np.sqrt(self.sigma2_ewma)
        self.vols.append(sigma_ewma)
        spread = k * sigma_ewma
        
        self.spreads.append(spread)
        return spread

    def trade(self, mid, spread, side, size, alpha, inv, c):
        """Executes a trade and updates inventory and cash"""
        skew = np.clip(-alpha * inv, -self.max_shift, self.max_shift) # inventory penalty on price
        if side == "buy":
            # client buys from MM at ask
            c += (mid + spread/2 + skew) * size  
            inv -= size
            self.total_spread_revenue += (spread/2 + skew) * size
        else:
            # client sells to MM at bid
            c -= (mid - spread/2 + skew) * size
            inv += size
            self.total_spread_revenue += (spread/2 - skew) * size
        
        self.inventory.append(inv)
        self.cash.append(c)
        
        return c, inv
    
    def hedge(self, t, mid, inv, c, hedge_threshold, hedge_size):
        """Hedges inventory if above threshold, returns updated cash and inventory"""
        # Dynamic hedging probbility based on volatility
        hedge_prob = min(1.0, self.lambda_h * np.sqrt(self.sigma2_ewma) / self.sigma_ref)
        if np.random.rand() < hedge_prob and abs(inv) > hedge_threshold:
            hedge_units = -np.sign(inv) * min(hedge_size, abs(inv))  # don't over-hedge (some optimization cases when hedge_threshold < hedge_size)
            # execution price -> (mid + sign * (slippage + impact * size))
            exec_price = mid + np.sign(hedge_units) * (self.gamma + self.eta * abs(hedge_units))
            inv += hedge_units
            c -= exec_price * hedge_units
            # hedge cost = abs(exec_price - mid) * abs(hedge_units) + fee_component
            self.total_hedge_cost += abs(exec_price - mid) * abs(hedge_units)
            self.total_hedge_volume += abs(hedge_units)
            self.num_hedges += 1
            self.hedge_events.append((t, hedge_units, exec_price))
        
        self.inventory.append(inv)
        self.cash.append(c)
        
        return c, inv
    
    def report_MM_stats(self, mid_prices):
        """Returns final MM stats: portfolio value, hedges, hedge costs, std inventory"""

        inventory = np.array(self.inventory)
        cash = np.array(self.cash)
        mids = np.array(mid_prices)
        # Portfolio value
        value = cash[-1] + inventory[-1] * mids[-1]
        return value, self.total_hedge_cost, self.total_hedge_volume, self.num_hedges, self.total_spread_revenue, inventory

    def plot_stats(self, mid_prices):
        """Plots mid prices, spreads, inventory, cash, portfolio value"""

        inventory = np.array(self.inventory)
        cash = np.array(self.cash)
        mid_prices = np.array(mid_prices)
        spreads = np.array(self.spreads)
        
        # Portfolio value
        value = cash + inventory * mid_prices

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,10))

        plt.subplot(6,1,1)
        plt.plot(mid_prices, label="Mid Price")
        plt.ylabel("Price")
        plt.legend()

        plt.subplot(6,1,2)
        plt.plot(self.vols)
        plt.axhline(self.sigma_ref, linewidth=0.8)
        plt.title("EWMA Volatility (sigma_ewma) and Reference")
        plt.xlabel("Time step")
        plt.ylabel("Volatility")
        plt.grid(True)
        plt.show()

        plt.subplot(6,1,3)
        plt.plot(spreads, label="Spread (vol-adaptive)", color="purple")
        plt.ylabel("Spread")
        plt.legend()

        plt.subplot(6,1,4)
        plt.plot(self.inventory, label="Inventory", color="green")
        # mark hedge events
        if self.hedge_events:
            xs = [e[0] for e in self.hedge_events]
            ys = [self.inventory[e[0]] for e in self.hedge_events]
            plt.scatter(xs, ys)
        plt.title("Inventory Path (hedge events marked)")
        plt.xlabel("Time step")
        plt.ylabel("Inventory")
        plt.grid(True)

        plt.subplot(6,1,5)
        plt.plot(cash, label="Cash", color="orange")
        plt.ylabel("Cash")
        plt.legend()

        plt.subplot(6,1,6)
        plt.plot(value, label="Portfolio Value", color="red")
        plt.ylabel("Value")
        plt.legend()

        plt.tight_layout()
        plt.show()

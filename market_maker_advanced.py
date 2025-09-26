import numpy as np

class MarketMakerAdv():
    def __init__(self, lam=0.94, sigma=0.02, slippage=0.01, eta=0.0005):
        self.lam = lam       # EWMA decay factor
        self.sigma2_ewma = sigma**2  # initial variance guess
        self.gamma = slippage     # slippage per unit hedged
        self.eta = eta           # market impact per unit hedged
        self.total_spread_revenue = 0.0    # revenue earned from client trades
        self.total_hedge_cost = 0.0        # cost incurred by external hedges
        self.total_hedge_volume = 0
        self.num_hedges = 0
        self.spreads = []
        self.inventory = []
        self.cash = []
    
    def get_spread(self, k, ret):
        """Returns spread based on EWMA volatility"""

        self.sigma2_ewma = self.lam * self.sigma2_ewma + (1 - self.lam) * (ret**2)
        sigma_ewma = np.sqrt(self.sigma2_ewma)
        spread = k * sigma_ewma
        
        self.spreads.append(spread)
        return spread

    def trade(self, mid, spread, side, alpha, inv, c):
        """Executes a trade and updates inventory and cash"""

        if side == "buy":
            # client buys from MM at ask
            inv -= 1
            c = mid + spread/2 + alpha*inv  # inventory penalty on price
        else:
            # client sells to MM at bid
            inv += 1
            c = - (mid - spread/2 + alpha*inv)
        
        self.total_spread_revenue += (spread/2 + alpha*inv)
        self.inventory.append(inv)
        self.cash.append(c)
        
        return c, inv
    
    def hedge(self, t, mid, inv, c, hedge_threshold, hedge_size):
        """Hedges inventory if above threshold, returns updated cash and inventory"""

        if abs(inv) > hedge_threshold:
            hedge_units = -np.sign(inv) * min(hedge_size, abs(inv))  # don't over-hedge (some optimization cases when hedge_threshold < hedge_size)
            # execution price -> (mid + sign * (slippage + impact * size))
            exec_price = mid + np.sign(hedge_units) * (self.gamma + self.eta * abs(hedge_units))
            inv += hedge_units
            c =- exec_price * hedge_units
            # hedge cost = (exec_price - mid) * abs(hedge_units) + fee_component
            self.total_hedge_cost += (exec_price - mid) * abs(hedge_units)
            self.total_hedge_volume += abs(hedge_units)
            num_hedges += 1
        
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
        return value, self.total_hedge_cost, self.total_hedge_volume, self.num_hedges, self.total_spread_revenue, np.std(inventory)
    
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

        plt.subplot(5,1,1)
        plt.plot(mid_prices, label="Mid Price")
        plt.ylabel("Price")
        plt.legend()

        plt.subplot(5,1,2)
        plt.plot(spreads, label="Spread (vol-adaptive)", color="purple")
        plt.ylabel("Spread")
        plt.legend()

        plt.subplot(5,1,3)
        plt.plot(inventory, label="Inventory", color="green")
        plt.ylabel("Inventory")
        plt.legend()

        plt.subplot(5,1,4)
        plt.plot(cash, label="Cash", color="orange")
        plt.ylabel("Cash")
        plt.legend()

        plt.subplot(5,1,5)
        plt.plot(value, label="Portfolio Value", color="red")
        plt.ylabel("Value")
        plt.legend()

        plt.tight_layout()
        plt.show()
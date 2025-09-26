import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Run params
T = 500                 # time steps
dt = 1/100              # time step size
sigma = 0.02            # volatility
mid0 = 100.0            # initial mid price
spread = 0.1            # fixed spread
order_prob = 0.3        # prob of order arrival
hedge_threshold = 50    # when to hedge inventory
hedge_size = 20         # how much to hedge when threshold exceeded

# Inventory
mid_prices = [mid0]
inventory = [0]
cash = [0]
hedges = []

# Simulation params
mid = mid0
inv = 0
c = 0

for t in range(1, T):
    # GBM without drift
    mid *= np.exp(-0.5*sigma**2*dt + sigma*np.sqrt(dt)*np.random.randn())
    mid_prices.append(mid)
    
    # Probability order arrives at this step
    if np.random.rand() < order_prob:
        side = np.random.choice(["buy", "sell"])
        if side == "buy":
            # Client buys from MM
            inv -= 1
            c += mid + spread/2
        else:
            # Client sells to MM
            inv += 1
            c -= mid - spread/2
    
    # If inventory too large, MM hedges at mid price
    if abs(inv) > hedge_threshold:
        hedge_units = -np.sign(inv) * hedge_size
        inv += hedge_units
        c -= hedge_units * mid
        hedges.append((t, mid, hedge_units))
    else:
        hedges.append((t, mid, 0))
    
    inventory.append(inv)
    cash.append(c)

# Final MM stats
inventory = np.array(inventory)
cash = np.array(cash)
mid_prices = np.array(mid_prices)

# MM Portfolio = cash + (inv * mid)
value = cash + inventory * mid_prices

plt.figure(figsize=(12,6))

#Plotting the evolution of mid price, inventory, and PnL
plt.subplot(3,1,1)
plt.plot(mid_prices, label="Mid Price")
plt.ylabel("Price")
plt.legend()

plt.subplot(3,1,2)
plt.plot(inventory, label="Inventory", color="orange")
plt.axhline(hedge_threshold, color="red", ls="--")
plt.axhline(-hedge_threshold, color="red", ls="--")
plt.ylabel("Inventory")
plt.legend()

plt.subplot(3,1,3)
plt.plot(value, label="PnL", color="green")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.show()
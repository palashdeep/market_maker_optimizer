# Project: Market Making Tradeoff Simulator

## Overview

This project studies market maker's tradeoff between **spread capture**, **inventory risk** and **hedging cost** in a noisy and uncertain market environment

Rather than optimizing for raw PnL - which is dominated by mark-to-market noise - the focus is on **controlled PnL** and **robust decision-making across many market simulations.**

The core question I try to answer is:

> Which quoting and hedging behaviors remain profitable across many stochastic paths, not just a single realization?

## Model Summary

### Price Process

- Midprice follows a geometric Brownian motion
- Volatility is annualized and scaled by √Δt
- Common random numbers are used across parameter evaluations

### Order Flow

- Buy/Sell orders arrive randomly
- Order flow is exogenous

### Quoting

- Spread is proportional to estimated volatility
- Quotes are skewed are linearly with inventory
```ini
skew = −α × inventory
```

### Hedging

- Inventory is hedged externally when it exceeds a volatility-scaled threshold
- Hedge execution includes fixed slippage and linear market impact

Hedging decisions are solely driven by inventory risk; execution costs are accounted for ex-post rather than used as decision constraints

## PnL Decomposition

Total PnL is decomposed as:
```ini
Total_PnL = Controlled_PnL + Mark-to-Market_Noise
```
where:
- Controlled_PnL = spread revenue - headge cost
- Mark-to-Market_Noise = inventory x price movement

Only **controlled PnL** is used for optimization and evaluation, since MTM dominates variance and is not directly controllable by the strategy.

## Evaluation Methodology

- Each parameter set is evaluated across many independent price paths
- For each set, we compute:
    - mean controlled PnL
    - standard deviation of PnL
    - t-statistic
    - mean inventory volatility
    - mean hedge cost

Experiments are performed via a single runner script -  `run_experiment.py`.

A parameter set is considered statistically meaningful only if:
```perl
t-stat ≥ 2
```

Out of sample validation is performed by re-evaluating statistically singnificant parameter regimes on unseen random seeds

## Optimization Perspective

Rather than selecting a single "best" parameter set, we focus on **Pareto efficient tradeoffs** between:
- controlled PnL (↑)
- inventory risk (↓)

This reflects the reality of market making which involves choosing a position on the tradeoff surface, not maximizing a single objective.

## Baseline Comparison

A simple baseline strategy is included with:
- fixed spread
- no inventory skew
- static hedging threshold

Optimized startegies consistently outperform the baseling on controlled PnL while maintaining comparable inventory risk.

For reference, exploratory comparisons against simple baselines were conducted during development (see `research` branch)

## Key Findings

- Raw PnL is extremely noisy and misleading for optimization
- Controlled PnL reveals stable and repeatable structure
- Only a subset of parameter space (~10%) produces statistically significant performance
- Volatility adaptive hedging materially reduces tail inventory exposure

## Limitations

- Order flow is independent of quotes (no adverse selection)
- Hedge execution assumes immediate fills with simplified impact
- The model does not learn parameters online

These simplifications are intentional to isolate core tradeoffs

## Possible Extensions

- Quote-dependent fill probabilities
- Adverse selection modelling
- Online learning of inventory skew
- Regime-dependent volatility

## Repository Structure

```graphql
simulator.py      # price generation and single-path simulation
market_maker.py  # quoting, hedging, and controlled PnL logic
optimizer.py     # Monte-Carlo evaluation and Pareto analysis
```
Exploratory experiments and alternative implementations are preserved on the `research` branch

## Takeaway

This project emphasizes **robust reasoning under uncertainty**, not curve-fitting. The goal is not to find a single optimal parameter set, but to understand which behaviors survive noise.
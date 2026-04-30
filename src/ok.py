import numpy as np
from scipy.optimize import brentq
from math import comb

# --- Step 1: Market Data and Parameters ---
spot_rates = np.array([0.030, 0.031, 0.032, 0.033, 0.034,
                       0.035, 0.0355, 0.036, 0.0365, 0.037])
n = len(spot_rates)
Z_market = [100 / ((1 + r) ** (i + 1)) for i, r in enumerate(spot_rates)]

b = 0.05            # log volatility
tolerance = 1e-8
notional = 1_000_000
swap_rate = 0.039
expiry = 3
maturity = 10
delta = 1

# --- Step 2: Calibrate BDT Tree ---
a_list = []
bdt_tree = {}

def build_rate_row(a_i, b, i):
    return [a_i * np.exp(b * j) for j in range(i + 1)]

def price_zero_coupon(i, rate_row):
    bond_prices = [100.0] * (i + 1)
    for t in reversed(range(i + 1)):
        new_prices = []
        for j in range(t):
            disc = 1 + rate_row[j]
            value = 0.5 * (bond_prices[j] + bond_prices[j + 1]) / disc
            new_prices.append(value)
        bond_prices = new_prices
    return bond_prices[0]

for i in range(n):
    def objective(a_i):
        rate_row = build_rate_row(a_i, b, i)
        return price_zero_coupon(i, rate_row) - Z_market[i]

    lower = 0.0001
    upper = 0.5 if i == 0 else a_list[-1] * 1.5
    a_i = brentq(objective, lower, upper, xtol=tolerance)
    a_list.append(a_i)
    bdt_tree[i] = build_rate_row(a_i, b, i)

# --- Step 3: Helper Functions ---

def get_short_rate(t, j):
    return bdt_tree[t][j]

def binomial_prob(t, j):
    return comb(t, j) * (0.5 ** t)

def swap_value_at_node(t, j):
    value = 0.0
    for k in range(t + 1, maturity + 1):
        r = get_short_rate(k - 1, j)
        cf = notional * (r - swap_rate) * delta

        # Discount cash flow from k to t
        df = 1.0
        for m in range(t, k):
            df *= 1 / (1 + get_short_rate(m, j))
        value += cf * df
    return value

# --- Step 4: Price Payer Swaption at t = 0 ---

swaption_price = 0.0
for j in range(expiry + 1):
    prob = binomial_prob(expiry, j)
    payoff = swap_value_at_node(expiry, j)
    swaption_price += prob * payoff

# --- Step 5: Output Results ---

print("\nCalibrated a_i values:")
for i, a in enumerate(a_list):
    print(f"a_{i+1} = {a:.10f}")

print("\nCalibrated BDT Tree (in %):")
for i in range(n):
    print(f"Period {i+1}: {[round(100 * r, 4) for r in bdt_tree[i]]}")

print(f"\nPrice of payer swaption: ${swaption_price:,.2f}")

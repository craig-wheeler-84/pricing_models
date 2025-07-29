# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:40:42 2025

@author: Laptop
"""

#import os
#print(os.getcwd())

import numpy as np
import pandas as pd


sofr_curves = pd.read_csv("historical_sofr_curves.csv")
sofr_curves["Date"] = pd.to_datetime(sofr_curves["Date"])



class SwapPricer:
    def __init__(self, notional, fixed_rate, maturity, pay_freq=1):
    
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.maturity = maturity
        self.pay_freq = pay_freq
        self.payment_times = np.arange(1, maturity + 1 / pay_freq, 1 / pay_freq)

    def discount_factors(self, zero_curve):
        dfs = []
        for t in self.payment_times:
            # Use the closest tenor available
            tenor = f"{int(round(t))}Y"
            r = zero_curve.get(tenor, list(zero_curve.values())[-1])  # fallback to last rate
            dfs.append(np.exp(-r * t))
        #print(dfs)
        return np.array(dfs)

    def price(self, zero_curve):
        dfs = self.discount_factors(zero_curve)

        # Fixed leg: sum of fixed payments
        fixed_cashflows = self.fixed_rate * self.notional * (1 / self.pay_freq)
        pv_fixed = np.sum(fixed_cashflows * dfs)

        # Floating leg: approximate using par swap assumption
        pv_float = self.notional * (1 - dfs[-1])

        # NPV from payer's perspective = receive float - pay fixed
        npv = pv_float - pv_fixed
        return pv_fixed, pv_float, npv


swap_price = SwapPricer(10_000_000, 0.042, 5)

results = []
for _, row in sofr_curves.iterrows():
    date = row["Date"]
    zero_curve = (row.drop("Date")/100).astype(float).to_dict()
    npv = swap_price.price(zero_curve)[2]
    results.append((date, npv))

pricing_results = pd.DataFrame(results, columns=["Date", "NPV"])

#VaR Calculation

latest_row = sofr_curves.iloc[-1]

today_curve = latest_row.drop("Date") / 100

today_curve = today_curve.astype(float).to_dict()

# Price using today's curve
npv_today = swap_price.price(today_curve)[2]

#print(npv_today)

pnl_list = []

for i in range(len(sofr_curves) - 1):  # exclude today
    hist_row = sofr_curves.iloc[i]
    hist_curve = hist_row.drop("Date") / 100
    hist_curve = hist_curve.astype(float).to_dict()

    npv_hist = swap_price.price(hist_curve)[2]
    pnl = npv_today - npv_hist  # P&L from applying historical shock
    pnl_list.append((hist_row["Date"], pnl))
    
pnl_df = pd.DataFrame(pnl_list, columns=["Date", "PnL"])
#print(pnl_df.head())

var_99 = -np.percentile(pnl_df["PnL"], 1)

print(f"Full Reval 99% VaR Shocking all RF's: {var_99:,.2f}")

# ----- Individual VaRs (tenor-by-tenor) -----

tenors = list(today_curve.keys())  # e.g., ['1Y', '2Y', '3Y', '4Y', '5Y']
individual_pnls = {t: [] for t in tenors}

for i in range(len(sofr_curves) - 1):  # exclude today
    hist_row = sofr_curves.iloc[i]

    for t in tenors:
        shocked_curve = today_curve.copy()
        shocked_curve[t] = hist_row[t] / 100  # shock just this tenor

        npv_shocked = swap_price.price(shocked_curve)[2]
        pnl = npv_today - npv_shocked
        individual_pnls[t].append(pnl)

# Calculate 99% historical VaR for each tenor
individual_vars = {t: -np.percentile(pnls, 1) for t, pnls in individual_pnls.items()}

# Output results
print("\nIndividual 99% VaRs by tenor:")
for tenor, var in individual_vars.items():
    print(f"{tenor}: {var:,.2f}")
    
    
#Subset the relevant tenors (risk factors)
tenors = ["1Y", "2Y", "3Y", "4Y", "5Y"]
tenor_rates = sofr_curves[tenors] / 100  # Convert from % to decimal if needed

# Calculate daily changes (log or simple)
rate_changes = tenor_rates.diff().dropna()

# Correlation matrix
correlation_matrix = rate_changes.corr()

print("      ")

print("Correlation Matrix of 5 Risk Factors (Daily Changes):")
print(correlation_matrix)

# Convert individual_vars dict to numpy array aligned with 'tenors'
var_vector = np.array([individual_vars[t] for t in tenors])

# Convert correlation_matrix DataFrame to numpy array
corr_matrix = correlation_matrix.values

# Calculate variance-covariance matrix from correlation matrix and individual VaRs
cov_matrix = np.outer(var_vector, var_vector) * corr_matrix

# Calculate total portfolio VaR via variance-covariance formula
total_var = np.sqrt(var_vector @ corr_matrix @ var_vector)

print(f"\nTotal VaR from Variance-Covariance method: {total_var:,.2f}")

# Optional: print covariance matrix nicely
cov_df = pd.DataFrame(cov_matrix, index=tenors, columns=tenors)
#print("\nCovariance Matrix (million^2):")
#print(cov_df)

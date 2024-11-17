import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import seaborn as sns
from funcs import (
    load_local, zero_rate_curve, calculate_time_to_maturity,
    BlackScholesLognormalCall, BlackScholesLognormalPut,
    impliedVolatility
)

# Set style for plots using Seaborn
sns.set_theme(style="darkgrid")  # Choose your preferred Seaborn theme
sns.set_palette("husl")

# Load market data
spx, spy, rates = load_local()

# Constants
S0 = 3662.45 
EXPIRY = 20210115

def payoff_1(S_T):
    """Calculate first exotic payoff"""
    return S_T**(1/3) + 1.5 * np.log(S_T) + 10.0

def payoff_2(sigma, T):
    """Calculate model-free integrated variance"""
    return sigma**2 * T

def black_scholes_price(S0, K, r, sigma, T, payoff_func):
    """Price using Black-Scholes model"""
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    return np.exp(-r*T) * (
        payoff_func(K) * norm.cdf(-d2) + 
        payoff_func(S0) * norm.cdf(d1)
    )

def black_scholes_mc(S0, r, sigma, T, payoff_func, n_sims=100000):
    """Monte Carlo simulation for Black-Scholes model"""
    Z = np.random.normal(0, 1, n_sims)
    S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoffs = payoff_func(S_T)
    mc_price = np.exp(-r*T) * np.mean(payoffs)
    mc_std = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_sims)
    return mc_price, mc_std, payoffs

def bachelier_price(S0, K, r, sigma, T, payoff_func):
    """Price using Bachelier model"""
    d = (S0 - K)/(sigma*np.sqrt(T))
    
    return np.exp(-r*T) * (
        payoff_func(K) * norm.cdf(-d) +
        payoff_func(S0) * norm.cdf(d)
    )

def bachelier_mc(S0, r, sigma, T, payoff_func, n_sims=100000):
    """Monte Carlo simulation for Bachelier model"""
    Z = np.random.normal(0, 1, n_sims)
    S_T = S0 + r*T + sigma*np.sqrt(T)*Z
    payoffs = payoff_func(S_T)
    mc_price = np.exp(-r*T) * np.mean(payoffs)
    mc_std = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_sims)
    return mc_price, mc_std, payoffs

def h_prime(K):
    """First derivative of payoff function"""
    return (1/(3*K**(2/3))) + 1.5/K

def h_double_prime(K):
    """Second derivative of payoff function"""
    return (-2/(9*K**(5/3))) - 1.5/(K**2)

def static_replication(spx_data, S0, T, payoff_func):
    """Price using static replication with market data"""
    # Filter options for target expiry
    options = spx_data[spx_data['exdate'] == EXPIRY].copy()
    
    # Calculate time to maturity and rate
    T_days, days = calculate_time_to_maturity(EXPIRY)
    r = zero_rate_curve(days, rates)
    
    # Sort strikes and calculate weights
    strikes = np.sort(options['strike_price'].unique())
    
    # Initialize price
    price = np.exp(-r*T) * payoff_func(S0)  # First term
    
    # Add integral terms using market prices
    for K in strikes:
        # Get market prices
        call = options[
            (options['strike_price'] == K) & 
            (options['cp_flag'] == 'C')
        ]['best_offer'].iloc[0]
        
        put = options[
            (options['strike_price'] == K) & 
            (options['cp_flag'] == 'P')
        ]['best_offer'].iloc[0]
        
        # Add contribution from this strike
        price += np.exp(-r*T) * h_double_prime(K) * 0.5 * (call + put)
    
    return price

def plot_mc_convergence(mc_convergence_df, bs_price):
    plt.figure(figsize=(10, 6))
    plt.semilogx(mc_convergence_df['Simulations'], mc_convergence_df['Price'], 'b-', label='MC Price')
    plt.fill_between(mc_convergence_df['Simulations'], 
                     mc_convergence_df['CI_Lower'],
                     mc_convergence_df['CI_Upper'],
                     alpha=0.2)
    plt.axhline(y=bs_price, color='r', linestyle='--', label='Analytical Price')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Option Price')
    plt.title('Monte Carlo Price Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('mc_convergence.png')
    plt.close()

def plot_payoff_distributions(bs_payoffs, bach_payoffs):
    plt.figure(figsize=(10, 6))
    plt.hist(bs_payoffs, bins=50, alpha=0.5, label='Black-Scholes', density=True)
    plt.hist(bach_payoffs, bins=50, alpha=0.5, label='Bachelier', density=True)
    plt.xlabel('Payoff Value')
    plt.ylabel('Density')
    plt.title('Distribution of Payoffs')
    plt.legend()
    plt.grid(True)
    plt.savefig('payoff_distributions.png')
    plt.close()

def plot_butterfly_spreads(butterfly_df):
    if not butterfly_df.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(butterfly_df['K2'], butterfly_df['Butterfly_Value'])
        plt.xlabel('Middle Strike (K2)')
        plt.ylabel('Butterfly Spread Value')
        plt.title('Butterfly Spread Values')
        plt.grid(True)
        plt.savefig('butterfly_spreads.png')
        plt.close()

def main():
    # Fixed parameters
    EXPIRY = 20210115
    T_days, days = calculate_time_to_maturity(EXPIRY)
    T = T_days/365
    r = zero_rate_curve(days, rates)
    
    # Filter options for target expiry
    options = spx[spx['exdate'] == EXPIRY].copy()
    
    # Parameters from Part 2 calibration
    sigma_calibrated = 0.1668 #From SABR calibration
    beta = 0.6690
    F = S0 * np.exp(r * T)  # Forward price formula: F = S0 * e^(rT)
    
    print(f"\n=== Pricing Exotic Derivatives for 15-Jan-2021 ===")
    print(f"Time to maturity (years): {T:.4f}")
    print(f"Risk-free rate: {r:.4f}")
    print(f"Using calibrated σ: {sigma_calibrated:.4f}")
    print(f"Forward price (F): {F:.4f}")
    
    # 1. Black-Scholes Prices using calibrated sigma
    print("\n1. Black-Scholes Prices (using calibrated σ):")
    bs_price = black_scholes_price(S0, F, r, sigma_calibrated, T, payoff_1)
    bs_var = sigma_calibrated**2 * T
    bs_mc_price, bs_mc_std, bs_payoffs = black_scholes_mc(S0, r, sigma_calibrated, T, payoff_1)
    
    # Perform statistical test for Black-Scholes
    bs_mean = np.mean(bs_payoffs)
    bs_se = np.std(bs_payoffs) / np.sqrt(len(bs_payoffs))
    bs_tstat = (bs_mean - bs_price) / bs_se
    bs_pvalue = 2 * (1 - norm.cdf(abs(bs_tstat)))
    
    print(f"Exotic Payoff 1 (S^(1/3) + 1.5*log(S) + 10):")
    print(f"  Null Hypothesis: μ = {bs_mc_price:.4f}")
    print(f"  Alternative Hypothesis: μ ≠ {bs_mc_price:.4f}")
    print(f"  Standard Error: {bs_se:.4f}")
    print(f"  Test Statistic: {bs_tstat:.4f}")
    print(f"  p-value: {bs_pvalue:.4f}")
    print(f"  Analytical: {bs_price:.4f}")
    print(f"  Monte Carlo: {bs_mc_price:.4f} ± {1.96*bs_mc_std:.4f} (95% CI)")
    print(f"  Difference: {abs(bs_price - bs_mc_price):.4f} ({100*abs(bs_price - bs_mc_price)/bs_price:.2f}%)")
    print(f"  Null Hypothesis: Analytical mean equals Monte Carlo mean")
    print(f"  Alternative Hypothesis: Analytical mean differs from Monte Carlo mean")
    print(f"  Test Statistic: {bs_tstat:.4f}")
    print(f"  p-value: {bs_pvalue:.4f}")
    print(f"  Statistically significant difference: {'Yes' if bs_pvalue < 0.05 else 'No'}")
    print(f"Integrated Variance (σ²T): {bs_var:.4f}")
    
    # 2. Bachelier Prices using adjusted sigma
    sigma_normal = sigma_calibrated * S0 * np.exp(-r * T / 2)
    print("\n2. Bachelier Prices (using converted normal σ):")
    bach_price = bachelier_price(S0, F, r, sigma_normal, T, payoff_1)
    bach_var = (sigma_normal**2 * T) / (S0**2)
    bach_mc_price, bach_mc_std, bach_payoffs = bachelier_mc(S0, r, sigma_normal, T, payoff_1)
    
    # Perform statistical test for Bachelier
    bach_mean = np.mean(bach_payoffs)
    bach_se = np.std(bach_payoffs) / np.sqrt(len(bach_payoffs))
    bach_tstat = (bach_mean - bach_price) / bach_se
    bach_pvalue = 2 * (1 - norm.cdf(abs(bach_tstat)))
    
    print(f"Exotic Payoff 1 (S^(1/3) + 1.5*log(S) + 10):")
    print(f"  Null Hypothesis: μ = {bach_mc_price:.4f}")
    print(f"  Alternative Hypothesis: μ ≠ {bach_mc_price:.4f}")
    print(f"  Standard Error: {bach_se:.4f}")
    print(f"  Test Statistic: {bach_tstat:.4f}")
    print(f"  p-value: {bach_pvalue:.4f}")
    print(f"  Analytical: {bach_price:.4f}")
    print(f"  Monte Carlo: {bach_mc_price:.4f} ± {1.96*bach_mc_std:.4f} (95% CI)")
    print(f"  Difference: {abs(bach_price - bach_mc_price):.4f} ({100*abs(bach_price - bach_mc_price)/bach_price:.2f}%)")
    print(f"  Null Hypothesis: Analytical mean equals Monte Carlo mean")
    print(f"  Alternative Hypothesis: Analytical mean differs from Monte Carlo mean")
    print(f"  Test Statistic: {bach_tstat:.4f}")
    print(f"  p-value: {bach_pvalue:.4f}")
    print(f"  Statistically significant difference: {'Yes' if bach_pvalue < 0.05 else 'No'}")
    print(f"Integrated Variance (σ²T): {bach_var:.4f}")
    
    # 3. Static Replication using market data
    print("\n3. Static Replication Price (using market data):")
    static_price = static_replication(spx, S0, T, payoff_1)
    print(f"Exotic Payoff 1 (S^(1/3) + 1.5*log(S) + 10): {static_price:.4f}")
    
    # Sense check across all methods
    print("\nSense Check - Comparison across all methods:")
    print(f"Black-Scholes: {bs_price:.4f}")
    print(f"Bachelier: {bach_price:.4f}")
    print(f"Static Replication: {static_price:.4f}")
    print(f"Max difference between methods: {max(abs(bs_price - bach_price), abs(bs_price - static_price), abs(bach_price - static_price)):.4f}")
    
    # Additional Tests
    print("\n Validation Tests:")
    
    # 1. Monte Carlo Convergence Test DataFrame
    mc_data = []
    sim_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    for n in sim_sizes:
        bs_mc_price, bs_mc_std, _ = black_scholes_mc(S0, r, sigma_calibrated, T, payoff_1, n_sims=n)
        mc_data.append({
            'Simulations': n,
            'Price': bs_mc_price,
            'Std Error': bs_mc_std,
            'CI_Lower': bs_mc_price - 1.96*bs_mc_std,
            'CI_Upper': bs_mc_price + 1.96*bs_mc_std
        })
    mc_convergence_df = pd.DataFrame(mc_data)
    print("\nMonte Carlo Convergence Test Results:")
    print(mc_convergence_df.to_string(index=False))
    
    # Create plots
    plot_mc_convergence(mc_convergence_df, bs_price)
    plot_payoff_distributions(bs_payoffs, bach_payoffs)
    
    # 2. Arbitrage Tests
    # Sort options by strike price
    calls = options[options['cp_flag'] == 'C'].sort_values('strike_price')
    puts = options[options['cp_flag'] == 'P'].sort_values('strike_price')
    
    # Butterfly Test DataFrame
    butterfly_data = []
    for i in range(len(puts)-2):
        K1 = puts.iloc[i]['strike_price']
        K2 = puts.iloc[i+1]['strike_price']
        K3 = puts.iloc[i+2]['strike_price']
        P1 = puts.iloc[i]['best_offer']
        P2 = puts.iloc[i+1]['best_offer']
        P3 = puts.iloc[i+2]['best_offer']
        
        butterfly = P1 - 2*P2 + P3
        if butterfly < -0.01:  # Allow for small numerical errors
            butterfly_data.append({
                'K1': K1,
                'K2': K2,
                'K3': K3,
                'P1': P1,
                'P2': P2,
                'P3': P3,
                'Butterfly_Value': butterfly
            })
    
    butterfly_df = pd.DataFrame(butterfly_data)
    print("\nButterfly Test Results (Potential Arbitrage Opportunities):")
    if not butterfly_df.empty:
        print(butterfly_df.head(10).to_string(index=False))
    else:
        print("No butterfly arbitrage opportunities found")
    
    plot_butterfly_spreads(butterfly_df)
    
    # Box Spread Test DataFrame
    box_spreads = []
    for i in range(len(calls)-1):
        K1 = calls.iloc[i]['strike_price']
        K2 = calls.iloc[i+1]['strike_price']
        C1 = calls.iloc[i]['best_offer']
        C2 = calls.iloc[i+1]['best_offer']
        P1 = puts[puts['strike_price'] == K1]['best_offer'].iloc[0]
        P2 = puts[puts['strike_price'] == K2]['best_offer'].iloc[0]
        
        box_value = (K2 - K1) * np.exp(-r * T)
        box_cost = (C1 - C2) + (P2 - P1)
        diff = box_value - box_cost
        
        if abs(diff) > 1.0:  # Only include significant differences
            box_spreads.append({
                'K1': K1,
                'K2': K2,
                'Box_Value': box_value,
                'Box_Cost': box_cost,
                'Difference': diff
            })
    
    box_spread_df = pd.DataFrame(box_spreads)
    print("\nBox Spread Test Results (Significant Arbitrage Opportunities):")
    if not box_spread_df.empty:
        print(box_spread_df.head(1).to_string(index=False))
    else:
        print("No significant box spread arbitrage opportunities found")
    
    # 5. Cross-Model Correlation Test
    print("\n5. Cross-Model Correlation:")
    corr = np.corrcoef(bs_payoffs, bach_payoffs)[0,1]
    print(f"  BS-Bachelier Correlation: {corr:.4f}")
    
    # Additional visualization for model comparison
    plt.figure(figsize=(10, 6))
    methods = ['Black-Scholes', 'Bachelier', 'Static Replication']
    prices = [bs_price, bach_price, static_price]
    plt.bar(methods, prices)
    plt.ylabel('Price')
    plt.title('Price Comparison Across Methods')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()

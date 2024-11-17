from scipy.stats import norm, mannwhitneyu
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm 
import warnings
import scipy.stats as statsz
import pandas as pd 
from matplotlib.ticker import PercentFormatter
from concurrent.futures import ThreadPoolExecutor

MATURITY = 1 / 12
R = 0.05
S0 = 100.0
K = 100.0
SIGMA = 0.2
M = 1000
TRADING_DAYS = 252
warnings.filterwarnings("ignore")
np.set_printoptions(precision=3)
plt.style.use("ggplot")
plt.rcParams.update({
    "axes.grid": True,
    "grid.color": "grey",
    "grid.alpha": 0.25,
    "axes.facecolor": "white",
    "legend.fontsize": 10
})

def phi(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def psi_Bt(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -K * np.exp(-r * T) * norm.cdf(d2)

def simulate_Brownian_Motion(paths, steps, T):
    deltaT = T / steps
    t = np.linspace(0, T, steps + 1)
    X = np.random.randn(paths, steps)
    return t, np.cumsum(np.sqrt(deltaT) * np.column_stack((np.zeros(paths), X)), axis=1)

def calc_error(blackscholespath, T, maturity, S0, K, r, sigma, dt):
    stockhedge_errors = np.zeros(len(T))
    bondhedge_errors = np.zeros(len(T))
    
    prev_phi = prev_bond_pos = 0
    for i, (t, S_t) in enumerate(zip(T, blackscholespath)):
        stock_pos = phi(S_t, K, r, sigma, maturity - t) * S_t
        bond_pos = psi_Bt(S_t, K, r, sigma, maturity - t)
        if t != 0.0:
            stockhedge_errors[i] = prev_phi * S_t - stock_pos
            bondhedge_errors[i] = prev_bond_pos * np.exp(r * dt) - bond_pos
        prev_phi = phi(S_t, K, r, sigma, maturity - t)
        prev_bond_pos = bond_pos

    return stockhedge_errors.sum() + bondhedge_errors.sum()

def calculate_greeks(S, T, K, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

def analyze_greeks_evolution():
    times = np.linspace(0, MATURITY, TRADING_DAYS + 1)
    stock_prices = np.linspace(0.8 * S0, 1.2 * S0, 100)
    
    delta_surface = np.zeros((len(times), len(stock_prices)))
    gamma_surface = np.zeros_like(delta_surface)
    theta_surface = np.zeros_like(delta_surface)
    vega_surface = np.zeros_like(delta_surface)
    
    for i, t in enumerate(times):
        for j, s in enumerate(stock_prices):
            greeks = calculate_greeks(s, MATURITY - t, K, R, SIGMA)
            delta_surface[i, j] = greeks['delta']
            gamma_surface[i, j] = greeks['gamma']
            theta_surface[i, j] = greeks['theta']
            vega_surface[i, j] = greeks['vega']
    
    fig = plt.figure(figsize=(20, 15))
    X, Y = np.meshgrid(times, stock_prices)
    
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(X, Y, delta_surface.T, color='blue')
    ax1.set_title('Delta Surface')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Stock Price')
    ax1.set_zlabel('Delta')
    
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_surface(X, Y, gamma_surface.T, color='blue')
    ax2.set_title('Gamma Surface')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Stock Price')
    ax2.set_zlabel('Gamma')
    
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot_surface(X, Y, theta_surface.T, color='blue')
    ax3.set_title('Theta Surface')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Stock Price')
    ax3.set_zlabel('Theta')
    
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot_surface(X, Y, vega_surface.T, color='blue')
    ax4.set_title('Vega Surface')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Stock Price')
    ax4.set_zlabel('Vega')
    
    plt.tight_layout()
    plt.show()

def run_simulation(steps, total_error, model_type):
    dt = MATURITY / steps
    for _ in tqdm(range(50000), desc="Simulating for {steps} steps ({model_type} model)"): 
        T, W_T = simulate_Brownian_Motion(1, steps, MATURITY)
        blackscholespath = S0 * np.exp((R - SIGMA**2 / 2) * T + SIGMA * W_T[0])
        
        calc_error_value = calc_error(blackscholespath, T, MATURITY, S0, K, R, SIGMA, dt)
        
        total_error.append(calc_error_value)
        
        if len(total_error) > 1:
            total_error[-1] = (total_error[-1] + total_error[-2]) / 2

def create_box_plot(total_error_21, total_error_84):
    sigma_list_box = [0.1, 0.2, 0.3, 0.4, 0.5]
    errors_21 = []
    errors_84 = []
    
    labels_21 = [f'Sigma = {sigma}' for sigma in sigma_list_box] 
    
    for sigma in sigma_list_box:
        total_error_21_temp = []
        total_error_84_temp = []
        
        for _ in range(50000):
            T, W_T = simulate_Brownian_Motion(1, 21, MATURITY)
            blackscholespath_21 = S0 * np.exp((R - sigma**2 / 2) * T + sigma * W_T[0])
            blackscholespath_84 = S0 * np.exp((R - sigma**2 / 2) * T + sigma * W_T[0])
            total_error_21_temp.append(calc_error(blackscholespath_21, T, MATURITY, S0, K, R, sigma, MATURITY / 21))
            total_error_84_temp.append(calc_error(blackscholespath_84, T, MATURITY, S0, K, R, sigma, MATURITY / 84))
        
        errors_21.append(total_error_21_temp)
        errors_84.append(total_error_84_temp)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot(errors_21, labels=labels_21, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'), 
                medianprops=dict(color='red'))
    plt.title('Box Plot of Total Error (21 steps)')
    plt.xlabel('Volatility (Sigma)')
    plt.ylabel('Total Error')

    plt.subplot(1, 2, 2)
    plt.boxplot(errors_84, labels=labels_21, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'), 
                medianprops=dict(color='red'))
    plt.title('Box Plot of Total Error (84 steps)')
    plt.xlabel('Volatility (Sigma)')
    plt.ylabel('Total Error')

    plt.tight_layout()
    plt.show()

def main():
    total_error_21 = []
    total_error_84 = []
    
    ks_results = []
    mann_whitney_results = []

    with ThreadPoolExecutor() as executor:
        executor.submit(run_simulation, 21, total_error_21, "Standard")
        executor.submit(run_simulation, 84, total_error_84, "Standard")
        
    results = []

    def print_statistics(errors, label):
        mean_error = np.mean(errors)
        variance_error = np.var(errors)
        std_dev_error = np.std(errors)  # Calculate standard deviation
        kurtosis_error = np.sum((errors - mean_error)**4) / (len(errors) * (np.var(errors)**2)) - 3
        skewness_error = np.sum((errors - mean_error)**3) / (len(errors) * (np.std(errors)**3))
        print(f"{label} - Mean: {mean_error:.3f}, Variance: {variance_error:.3f}, Std Dev: {std_dev_error:.3f}, Kurtosis: {kurtosis_error:.3f}, Skewness: {skewness_error:.3f}")
        results.append({
            'Label': label,
            'Mean': mean_error,
            'Variance': variance_error,
            'Std Dev': std_dev_error,  # Add standard deviation to results
            'Kurtosis': kurtosis_error,
            'Skewness': skewness_error
        })

    print_statistics(total_error_21, "Total Error (Normal, 21 steps)")
    print_statistics(total_error_84, "Total Error (Normal, 84 steps)")
    results_df = pd.DataFrame(results)
    print(results_df) 

    ks_results = []
    mann_whitney_results = []
    plt.figure(figsize=(18, 10))

    plt.subplot(1, 2, 1)
    plt.hist(total_error_21, weights=np.ones(len(total_error_21)) / len(total_error_21),
             bins=np.linspace(-2, 2, 40), color="green")  
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Histogram of Total Error (Normal, 21 steps)', fontsize=16)
    plt.xlabel('Hedging Error', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    plt.subplot(1, 2, 2)
    plt.hist(total_error_84, weights=np.ones(len(total_error_84)) / len(total_error_84),
             bins=np.linspace(-2, 2, 40), color="green") 
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Histogram of Total Error (Normal, 84 steps)', fontsize=16)
    plt.xlabel('Hedging Error', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    plt.tight_layout()

    plt.tight_layout()
    plt.show()

    create_box_plot(total_error_21, total_error_84)
    
    return {
        'total_error_21': total_error_21,
        'total_error_84': total_error_84,
        'ks_results': ks_results,
        'mann_whitney_results': mann_whitney_results
    }

def mann_whitney_test(errors1, label1, errors2, label2):
    stat, p = mannwhitneyu(errors1, errors2, alternative='two-sided')
    alpha = 0.05
    significance = 'Yes' if p < alpha else 'No'
    hypothesis = 'Reject the null hypothesis.' if p < alpha else 'Fail to reject the null hypothesis.'
    
    result = {
        'Test': 'Mann-Whitney U',
        'Label1': label1,
        'Label2': label2,
        'Statistic': stat,
        'p-value': p,
        'Significant': significance,
        'Hypothesis': hypothesis
    }
    
    return result

def ks_two_sample_test(errors1, label1, errors2, label2):
    stat, p = statsz.ks_2samp(errors1, errors2)
    alpha = 0.05
    significance = 'Yes' if p < alpha else 'No'
    hypothesis = 'Reject the null hypothesis.' if p < alpha else 'Fail to reject the null hypothesis.'
    
    result = {
        'Test': 'KS',
        'Label1': label1,
        'Label2': label2,
        'Statistic': stat,
        'p-value': p,
        'Significant': significance,
        'Hypothesis': hypothesis
    }
    
    return result

def run_all_tests(errors):
    total_error_21 = errors['total_error_21']
    total_error_84 = errors['total_error_84']
    
    ks_results = errors['ks_results']
    mann_whitney_results = errors['mann_whitney_results']

    ks_results.append(ks_two_sample_test(total_error_21, "Normal 21", total_error_84, "Normal 84"))
    mann_whitney_results.append(mann_whitney_test(total_error_21, "Normal 21", total_error_84, "Normal 84"))

if __name__ == "__main__":
    errors = main()
    run_all_tests(errors)
    analyze_greeks_evolution()

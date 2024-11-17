from scipy.stats import norm
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm 
import warnings
from concurrent.futures import ThreadPoolExecutor
import scipy.stats as statsz
import seaborn as sns
from scipy.stats import mannwhitneyu
import pandas as pd 
from matplotlib.ticker import PercentFormatter

# Constants
MATURITY = 1 / 12
R = 0.05
S0 = 100.0
K = 100.0
SIGMA = 0.2
M = 1000
TRADING_DAYS = 252  # Number of trading days in a year
# Jump diffusion parameters
LAMBDA = 0.3  # Average jump rate
MU = 0.2      # Average jump size
SIGMA_JUMP = 0.3  # Volatility of jump size
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

def jump_diffusion_simulation(S0, r, sigma, T, N, lambda_, mu, sigma_jump, M):
    dt = T / N
    S_jump = np.zeros((M, N + 1))
    S_jump[:, 0] = S0

    for m in range(M):
        Z = np.random.normal(0, 1, (N,))  # For stock price
        jump_sizes = np.random.normal(mu, sigma_jump, (N,)) * (np.random.poisson(lambda_ * dt, (N,)) > 0)
        for i in range(1, N + 1):
            S_jump[m, i] = S_jump[m, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i-1] + jump_sizes[i-1])

    return S_jump

def calculate_greeks(S, T, K, r, sigma):
    """Calculate the Greeks for a European call option."""
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

def run_simulation(steps, total_error, total_error_antithetic, model_type):
    dt = MATURITY / steps
    for _ in tqdm(range(500), desc=f"Simulating for {steps} steps ({model_type} model)"): 
        T, W_T = simulate_Brownian_Motion(1, steps, MATURITY)
        blackscholespath = S0 * np.exp((R - SIGMA**2 / 2) * T + SIGMA * W_T[0])
        # Antithetic variate
        blackscholespath_antithetic = S0 * np.exp((R - SIGMA**2 / 2) * T + SIGMA * (-W_T[0]))
        calc_error_value = calc_error(blackscholespath, T, MATURITY, S0, K, R, SIGMA, dt)
        calc_error_value_antithetic = calc_error(blackscholespath_antithetic, T, MATURITY, S0, K, R, SIGMA, dt)
        total_error.append(calc_error_value)
        total_error_antithetic.append(calc_error_value_antithetic)

def main():
    total_error_21 = []
    total_error_21_antithetic = []
    total_error_84 = []
    total_error_84_antithetic = []
    jump_error_21 = []
    jump_error_84 = []
    
    # Initialize ks_results and mann_whitney_results
    ks_results = []
    mann_whitney_results = []

    # Jump diffusion simulation for 21 steps
    N1 = 21
    S_jump_paths_21 = jump_diffusion_simulation(S0, R, SIGMA, MATURITY, N1, LAMBDA, MU, SIGMA_JUMP, M)

    # Jump diffusion simulation for 84 steps
    N2 = 84
    S_jump_paths_84 = jump_diffusion_simulation(S0, R, SIGMA, MATURITY, N2, LAMBDA, MU, SIGMA_JUMP, M)

    # Calculate errors for jump diffusion model
    jump_error_21 = []
    jump_error_84 = []

    def calculate_jump_errors(steps, jump_errors, S_jump_paths):
        dt = MATURITY / steps 
        for _ in tqdm(range(500), desc=f"Calculating Jump errors for {steps} steps (Jump model)"):  
            T, W_T = simulate_Brownian_Motion(1, steps, MATURITY)
            m = np.random.randint(S_jump_paths.shape[0])  
            jump_path = S_jump_paths[m, :] 
            
            # Append the calculated error
            jump_errors.append(calc_error(jump_path, T, MATURITY, S0, K, R, SIGMA, dt))

    with ThreadPoolExecutor() as executor:
        executor.submit(run_simulation, 21, total_error_21, total_error_21_antithetic, "Standard")
        executor.submit(run_simulation, 21, total_error_21_antithetic, total_error_21_antithetic, "Antithetic")  
        executor.submit(run_simulation, 84, total_error_84, total_error_84_antithetic, "Standard")
        executor.submit(run_simulation, 84, total_error_84_antithetic, total_error_84_antithetic, "Antithetic") 
        executor.submit(calculate_jump_errors, 21, jump_error_21, S_jump_paths_21)
        executor.submit(calculate_jump_errors, 84, jump_error_84, S_jump_paths_84)
        
    results = []

    def print_statistics(errors, label):
        mean_error = np.mean(errors)
        variance_error = np.var(errors)
        kurtosis_error = np.sum((errors - mean_error)**4) / (len(errors) * (np.var(errors)**2)) - 3
        skewness_error = np.sum((errors - mean_error)**3) / (len(errors) * (np.std(errors)**3))
        print(f"{label} - Mean: {mean_error:.3f}, Variance: {variance_error:.3f}, Kurtosis: {kurtosis_error:.3f}, Skewness: {skewness_error:.3f}")
        results.append({
            'Label': label,
            'Mean': mean_error,
            'Variance': variance_error,
            'Kurtosis': kurtosis_error,
            'Skewness': skewness_error
        })

    print_statistics(total_error_21, "Total Error (Normal, 21 steps)")
    print_statistics(total_error_21_antithetic, "Total Error (Antithetic, 21 steps)")
    print_statistics(total_error_84, "Total Error (Normal, 84 steps)")
    print_statistics(total_error_84_antithetic, "Total Error (Antithetic, 84 steps)")
    print_statistics(jump_error_21, "Jump Error (21 steps)")
    print_statistics(jump_error_84, "Jump Error (84 steps)")
    results_df = pd.DataFrame(results)
    print(results_df) 


    ks_results = []
    mann_whitney_results = []



    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.hist(total_error_21, weights=np.ones(len(total_error_21)) / len(total_error_21),
             bins=np.linspace(-2, 2, 40), color="#68AC57")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Histogram of Total Error (Normal, 21 steps)')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 2)
    plt.hist(total_error_84, weights=np.ones(len(total_error_84)) / len(total_error_84),
             bins=np.linspace(-2, 2, 40), color="#68AC57")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Histogram of Total Error (Normal, 84 steps)')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 3)
    plt.hist(total_error_21_antithetic, weights=np.ones(len(total_error_21_antithetic)) / len(total_error_21_antithetic),
             bins=np.linspace(-2, 2, 40), color="#4682B4", alpha=0.5, label='Total Error (Antithetic)')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Histogram of Total Errors (Antithetic, 21 steps)')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.hist(total_error_84_antithetic, weights=np.ones(len(total_error_84_antithetic)) / len(total_error_84_antithetic),
             bins=np.linspace(-2, 2, 40), color="#4682B4", alpha=0.5, label='Total Error (Antithetic)')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Histogram of Total Errors (Antithetic, 84 steps)')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.hist(jump_error_21, weights=np.ones(len(jump_error_21)) / len(jump_error_21),
             bins=np.linspace(-2, 2, 40), color="#FF6347")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Histogram of Jump Error (21 steps)')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 6)
    plt.hist(jump_error_84, weights=np.ones(len(jump_error_84)) / len(jump_error_84),
             bins=np.linspace(-2, 2, 40), color="#FF6347")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Histogram of Jump Error (84 steps)')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    return {
        'total_error_21': total_error_21,
        'total_error_21_antithetic': total_error_21_antithetic,
        'total_error_84': total_error_84,
        'total_error_84_antithetic': total_error_84_antithetic,
        'jump_error_21': jump_error_21,
        'jump_error_84': jump_error_84,
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
    total_error_21_antithetic = errors['total_error_21_antithetic']
    total_error_84 = errors['total_error_84']
    total_error_84_antithetic = errors['total_error_84_antithetic']
    jump_error_21 = errors['jump_error_21']
    jump_error_84 = errors['jump_error_84']
    
    ks_results = errors['ks_results']
    mann_whitney_results = errors['mann_whitney_results']

    ks_results.append(ks_two_sample_test(total_error_21, "Normal 21", jump_error_21, "Jump 21"))
    ks_results.append(ks_two_sample_test(total_error_21, "Normal 21", total_error_21_antithetic, "Antithetic 21"))
    ks_results.append(ks_two_sample_test(jump_error_21, "Jump 21", total_error_21, "Normal 21"))
    ks_results.append(ks_two_sample_test(jump_error_21, "Jump 21", total_error_21_antithetic, "Antithetic 21"))
    ks_results.append(ks_two_sample_test(total_error_84, "Normal 84", jump_error_84, "Jump 84"))
    ks_results.append(ks_two_sample_test(total_error_84, "Normal 84", total_error_84_antithetic, "Antithetic 84"))
    ks_results.append(ks_two_sample_test(jump_error_84, "Jump 84", total_error_84, "Normal 84"))
    ks_results.append(ks_two_sample_test(jump_error_84, "Jump 84", total_error_84_antithetic, "Antithetic 84"))
    
    mann_whitney_results.append(mann_whitney_test(total_error_21, "Normal 21", total_error_21_antithetic, "Antithetic 21"))
    mann_whitney_results.append(mann_whitney_test(jump_error_21, "Jump 21", total_error_21, "Normal 21"))
    mann_whitney_results.append(mann_whitney_test(jump_error_21, "Jump 21", total_error_21_antithetic, "Antithetic 21"))
    mann_whitney_results.append(mann_whitney_test(total_error_84, "Normal 84", total_error_84_antithetic, "Antithetic 84"))
    mann_whitney_results.append(mann_whitney_test(jump_error_84, "Jump 84", total_error_84, "Normal 84"))
    mann_whitney_results.append(mann_whitney_test(jump_error_84, "Jump 84", total_error_84_antithetic, "Antithetic 84"))

if __name__ == "__main__":
    errors = main()
    run_all_tests(errors)
    analyze_greeks_evolution()
    # ks_df = pd.DataFrame(errors['ks_results'])
    # mw_df = pd.DataFrame(errors['mann_whitney_results'])
    # combined_tests_df = pd.concat([ks_df, mw_df], ignore_index=True)
    # combined_tests_df = combined_tests_df[['Test', 'Label1', 'Label2', 'Statistic', 'p-value', 'Significant', 'Hypothesis']]
    # print(combined_tests_df)

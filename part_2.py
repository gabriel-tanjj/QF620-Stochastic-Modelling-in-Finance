import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, least_squares
import matplotlib.pylab as plt
import math
from funcs import load_local

def ddcalibration(x, option_prices, ATM_iv, strikes, r, F, T) -> float:
    """Calculate total squared error between estimated and actual prices"""
    # x[0] is beta parameter
    # Transform strikes and volatility according to DDM
    F_dd = F / x[0]  # Displaced forward price
    K_dd = strikes + (F * (1 - x[0])) / x[0]  # Displaced strikes
    sig_dd = ATM_iv / x[0]  # Displaced volatility
    
    # Calculate model prices using displaced parameters
    est_prices = np.zeros_like(strikes)
    for i, K in enumerate(strikes):
        if K > F:
            est_prices[i] = BlackCall(F_dd, K_dd[i], r, sig_dd, T)
        else:
            est_prices[i] = BlackPut(F_dd, K_dd[i], r, sig_dd, T)
    
    # Return sum of squared errors
    return np.sum((option_prices - est_prices)**2)

def preprocess_options_data(file_path):
    """Preprocess options data"""
    df = pd.read_csv(file_path)
    df['mid'] = 0.5 * (df['best_bid'] + df['best_offer'])
    df['strike'] = df['strike_price'] * 0.001
    df['payoff'] = df['cp_flag'].map({'C': 'call', 'P': 'put'})
    return df

def sabrcalibration(x, strikes, vols, F, T) -> float:
    """Calculate total squared error between estimated and actual vols"""
    est_vols = np.array([SABR(F, K, T, x[0], beta, x[1], x[2]) for K in strikes])
    return np.sum((vols - est_vols)**2)

def impliedVolatility(S, K, r, price, T, payoff):
    try:
        if payoff.lower() == 'call':
            pricing_func = lambda x: price - BlackScholesLognormalCall(S, K, r, x, T)
        elif payoff.lower() == 'put':
            pricing_func = lambda x: price - BlackScholesLognormalPut(S, K, r, x, T)
        else:
            raise NameError('Payoff type not recognized')
            
        impliedVol = brentq(pricing_func, 1e-12, 10.0)
    except Exception:
        impliedVol = np.nan
    return impliedVol

def SABR(F, K, T, alpha, beta, rho, nu):
    """Calculate SABR implied volatility"""
    X = K
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        sabrsigma = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom
    return sabrsigma

def BlackScholesLognormalCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BlackScholesLognormalPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def BlackCall(F, K, r, sig, T):
    """Calculate Black model call price"""
    d1 = (np.log(F/K) + 0.5*(sig**2)*T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    return np.exp(-r*T) * (F*norm.cdf(d1) - K*norm.cdf(d2))

def BlackPut(F, K, r, sig, T):
    """Calculate Black model put price"""
    d1 = (np.log(F/K) + 0.5*(sig**2)*T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    return np.exp(-r*T) * (K*norm.cdf(-d2) - F*norm.cdf(-d1))

def DisplacedDiffusionPrice(beta, ATM_iv, strikes, r, S, T):
    """Calculate DD model prices"""
    F = S*np.exp(r*T)
    # Transform parameters according to DDM
    F_dd = F/beta  # Displaced forward
    K_dd = strikes + (F * (1-beta))/beta  # Displaced strikes 
    sig_dd = ATM_iv/beta  # Displaced volatility
    
    prices = np.zeros_like(strikes)
    for i, K in enumerate(strikes):
        if K > F:
            prices[i] = BlackCall(F_dd, K_dd[i], r, sig_dd, T)
        else:
            prices[i] = BlackPut(F_dd, K_dd[i], r, sig_dd, T)
            
    return prices

def calculate_displaced_diffusion_params(df, S, dict_ex_date_rate, impliedVolatility):
    dd_params = []
    
    fig, axes = plt.subplots(len(dict_ex_date_rate.items()), 1, figsize=(10, 4*len(dict_ex_date_rate.items())), tight_layout=True)
    
    for idx, (exdate, r) in enumerate(dict_ex_date_rate.items()):
        df_ex = df[df["exdate"] == exdate].copy()
        T = (pd.Timestamp(str(exdate)) - pd.Timestamp('2020-12-01')).days / 365
        r = r/100
        F = S * np.exp(r * T)

        df_ex['vols'] = df_ex.apply(lambda x: impliedVolatility(S, x['strike'], r, x['mid'], T, x['payoff']), axis=1)
        df_ex.dropna(inplace=True)
        
        strikes = df_ex[df_ex['payoff'] == 'put']['strike'].values
        df_iv = pd.DataFrame({
            'strike': strikes,
            'impliedvol': [df_ex[(df_ex['strike'] == K) & (df_ex['payoff'] == ('call' if K > S else 'put'))]['vols'].iloc[0] for K in strikes],
            'option_price': [df_ex[(df_ex['strike'] == K) & (df_ex['payoff'] == ('call' if K > S else 'put'))]['mid'].iloc[0] for K in strikes],
            'payoff': ['call' if K > S else 'put' for K in strikes]
        })

        closest_strike = df_iv.loc[(df_iv['strike'] - F).abs().idxmin(), 'strike']
        spx_ATM_implied_vol = df_iv[df_iv['strike'] == closest_strike]['impliedvol'].iloc[0]

        # Optimize beta parameter
        res = least_squares(lambda x: ddcalibration(x, df_iv['option_price'], spx_ATM_implied_vol, 
                                                  strikes, r, F, T), [0.5], bounds=([0.1], [0.9]))
        optimal_beta = res.x[0]
        
        # Calculate DD prices with optimal beta
        dd_prices = DisplacedDiffusionPrice(optimal_beta, spx_ATM_implied_vol, strikes, r, S, T)
        dd_vols = [impliedVolatility(S, K, r, p, T, pf) 
                  for K, p, pf in zip(strikes, dd_prices, df_iv['payoff'])]
        
        dd_params.append({
            'exdate': exdate,
            'dd_beta': optimal_beta,
            'sig_atm': spx_ATM_implied_vol,
            'dd_vols': dd_vols
        })
        
        axes[idx].set_title(f"{exdate} Market vs DD Implied Vol")
        axes[idx].plot(strikes, df_iv['impliedvol'], 'gs', label='Market Implied Vols')
        axes[idx].plot(strikes, dd_vols, 'r-', label=f'DD Implied Vols, beta={optimal_beta:.3f}')
        axes[idx].legend()

    plt.show()

    for params in dd_params:
        print(f"Exdate {params['exdate']}: beta = {params['dd_beta']:.5f}, sigma = {params['sig_atm']:.5f}")

    return dd_params

def calculate_sabr_params(df, S, dict_ex_date_rate, impliedVolatility, sabrcalibration, SABR, beta=0.7):
    """Calculate SABR parameters and plot results"""
    sabr_params = []
    
    for exdate, r in dict_ex_date_rate.items():
        df_ex = df[df["exdate"] == exdate].copy()
        T = (pd.Timestamp(str(exdate)) - pd.Timestamp('2020-12-01')).days / 365
        r = r/100
        F = S * np.exp(r * T)

        df_ex['vols'] = df_ex.apply(lambda x: impliedVolatility(S, x['strike'], r, x['mid'], T, x['payoff']), axis=1)
        df_ex.dropna(inplace=True)

        strikes = df_ex[df_ex['payoff'] == 'put']['strike'].values
        impliedvols = [df_ex[(df_ex['strike'] == K) & 
                            (df_ex['payoff'] == ('call' if K > S else 'put'))]['vols'].iloc[0] 
                      for K in strikes]
        
        df_iv = pd.DataFrame({'strike': strikes, 'impliedvol': impliedvols})
        
        res = least_squares(lambda x: sabrcalibration(x, df_iv['strike'], df_iv['impliedvol'], F, T),
                          [.02, .2, .1])
        alpha, rho, nu = res.x
        
        sabrvols = [SABR(F, K, T, alpha, beta, rho, nu) for K in strikes]
        sabr_params.append({'exdate': exdate, 'alpha': alpha, 'rho': rho, 'nu': nu, 'vols': sabrvols})
        
        plt.figure(tight_layout=True)
        plt.title(f"{exdate} Market vs SABR Implied Vol")
        plt.plot(strikes, df_iv['impliedvol'], 'gs', label='Market Implied Vols')
        plt.plot(strikes, sabrvols, 'm--', label='SABR Implied Vols')
        plt.legend()
        plt.show()

    for params in sabr_params:
        print(f"Exdate {params['exdate']}: alpha = {params['alpha']:.3f}, rho = {params['rho']:.3f}, nu = {params['nu']:.3f}")
        # Calculate and print SABR sigma for ATM strike
        atm_sigma = SABR(F, F, T, params['alpha'], beta, params['rho'], params['nu'])
        print(f"SABR ATM sigma for {params['exdate']}: {atm_sigma:.4f}")

    return sabr_params

def plot_sabr_params(df, S, dict_ex_date_rate, impliedVolatility, sabrcalibration, SABR, beta=0.7):
    """Plot SABR parameter sensitivity"""
    exdate = 20210115
    r = dict_ex_date_rate[exdate]/100
    T = (pd.Timestamp(str(exdate)) - pd.Timestamp('2020-12-01')).days / 365
    F = S * np.exp(r * T)

    df_ex = df[df["exdate"] == exdate].copy()
    df_ex['vols'] = df_ex.apply(lambda x: impliedVolatility(S, x['strike'], r, x['mid'], T, x['payoff']), axis=1)
    df_ex.dropna(inplace=True)

    strikes = df_ex[df_ex['payoff'] == 'put']['strike'].values
    impliedvols = [df_ex[(df_ex['strike'] == K) & 
                        (df_ex['payoff'] == ('call' if K > S else 'put'))]['vols'].iloc[0] 
                  for K in strikes]
    df_iv = pd.DataFrame({'strike': strikes, 'impliedvol': impliedvols})

    res = least_squares(lambda x: sabrcalibration(x, df_iv['strike'], df_iv['impliedvol'], F, T),
                       [.02, .2, .1])
    alpha, rho, nu = res.x

    ls_rho = [-.5, 0, .5]
    ls_nu = [nu*.8, nu*1.2, nu*1.4]
    colors = plt.cm.viridis(np.linspace(0, 1, len(ls_rho) + len(ls_nu) + 1))

    # Plot rho sensitivity
    plt.figure(tight_layout=True)
    plt.title(f"{exdate} Market vs SABR Implied Vol - Rho Sensitivity")
    plt.plot(strikes, df_iv['impliedvol'], 'gs', label='Market Implied Vols')
    for i, test_rho in enumerate(ls_rho):
        vols = [SABR(F, K, T, alpha, beta, test_rho, nu) for K in strikes]
        plt.plot(strikes, vols, '--', color=colors[i], 
                label=f'SABR Implied Vols rho={test_rho:.2f}')
    plt.legend()
    plt.show()

    # Plot nu sensitivity  
    plt.figure(tight_layout=True)
    plt.title(f"{exdate} Market vs SABR Implied Vol - Nu Sensitivity")
    plt.plot(strikes, df_iv['impliedvol'], 'gs', label='Market Implied Vols')
    for i, test_nu in enumerate(ls_nu):
        vols = [SABR(F, K, T, alpha, beta, rho, test_nu) for K in strikes]
        plt.plot(strikes, vols, '--', color=colors[i], 
                label=f'SABR Implied Vols nu={test_nu:.3f}')
    plt.legend()
    plt.show()

# Constants and data loading
beta = 0.7
S_spx = 3662.45 
S_spy = 366.02

df_rates = pd.read_csv("zero_rates_20201201.csv").sort_values('days')
df_spx = preprocess_options_data("SPX_options.csv")
df_spy = preprocess_options_data("SPY_options.csv")

exdates = sorted(df_spx['exdate'].unique())[:3]
days_to_expiry = [(pd.Timestamp(str(date)) - pd.Timestamp('2020-12-01')).days for date in exdates]
arr_rates = np.interp(days_to_expiry, df_rates['days'].values, df_rates['rate'].values)
dict_ex_date_rate = dict(zip(exdates, arr_rates))

calculate_displaced_diffusion_params(df_spx, S_spx, dict_ex_date_rate, impliedVolatility)
calculate_displaced_diffusion_params(df_spy, S_spy, dict_ex_date_rate, impliedVolatility)
calculate_sabr_params(df_spx, S_spx, dict_ex_date_rate, impliedVolatility, sabrcalibration, SABR, beta=0.7)
calculate_sabr_params(df_spy, S_spy, dict_ex_date_rate, impliedVolatility, sabrcalibration, SABR, beta=0.7)
plot_sabr_params(df_spx, S_spx, dict_ex_date_rate, impliedVolatility, sabrcalibration, SABR, beta=0.7)
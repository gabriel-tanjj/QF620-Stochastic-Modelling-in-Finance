import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import lzma
import dill as pickle

def save_local():
    spx = pd.read_csv("/Users/gabriel/Library/Mobile Documents/com~apple~CloudDocs/SMU Masters /QF620 Stochastic Modelling in Finance/Project/SPX_options.csv")
    spy = pd.read_csv("/Users/gabriel/Library/Mobile Documents/com~apple~CloudDocs/SMU Masters /QF620 Stochastic Modelling in Finance/Project/SPY_options.csv")
    rates = pd.read_csv("/Users/gabriel/Library/Mobile Documents/com~apple~CloudDocs/SMU Masters /QF620 Stochastic Modelling in Finance/Project/zero_rates_20201201.csv")
    save_pickle("spx.obj", spx)
    save_pickle("spy.obj", spy)
    save_pickle("rates.obj", rates)

def save_pickle(filepath, obj):
    with lzma.open(filepath, "wb") as fp:
        pickle.dump(obj, fp)
        
def load_pickle(filepath):
    with lzma.open(filepath, "rb") as fp:
        data = pickle.load(fp)
    return data

def load_local():
    """Load local data files and return paths"""
    spx = load_pickle("spx.obj")
    spy = load_pickle("spy.obj")
    rates = load_pickle("rates.obj")
    return spx, spy, rates

def zero_rate_curve(days, rates):
    """Create interpolated zero rate curve"""
    return interp1d(rates['days'], rates['rate'], fill_value="extrapolate")(days)

def calculate_time_to_maturity(exdate):
    """Calculate time to maturity in days"""
    exp_date = pd.to_datetime(exdate, format='%Y%m%d')
    curr_date = pd.Timestamp('2020-12-01')
    days_to_expiry = (exp_date - curr_date).days
    return max(days_to_expiry, 0), days_to_expiry

def BlackScholesLognormalCall(S, K, r, sigma, T):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def BlackScholesLognormalPut(S, K, r, sigma, T):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def impliedCallVolatility(S, K, r, price, T):
    """Calculate implied volatility for call options"""
    try:
        impliedVol = brentq(
            lambda x: price - BlackScholesLognormalCall(S, K, r, x, T),
            1e-12, 5.0
        )
    except Exception:
        impliedVol = np.nan
    return impliedVol

def impliedPutVolatility(S, K, r, price, T):
    """Calculate implied volatility for put options"""
    try:
        impliedVol = brentq(
            lambda x: price - BlackScholesLognormalPut(S, K, r, x, T),
            1e-12, 5.0
        )
    except Exception:
        impliedVol = np.nan
    return impliedVol

def impliedVolatility(S, K, r, price, T, payoff):
    """Calculate implied volatility based on option type"""
    if payoff.lower() == 'call':
        return impliedCallVolatility(S, K, r, price, T)
    elif payoff.lower() == 'put':
        return impliedPutVolatility(S, K, r, price, T)
    else:
        raise ValueError('Payoff type not recognized')

def SABR(F, K, T, alpha, beta, rho, nu):
    """Calculate SABR volatility"""
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta) ** 2) / 24) * alpha * alpha / (F ** (2 - 2 * beta))
        numer2 = 0.25 * rho * beta * nu * alpha / (F ** (1 - beta))
        numer3 = ((2 - 3 * rho * rho) / 24) * nu * nu
        VolAtm = alpha * (1 + (numer1 + numer2 + numer3) * T) / (F ** (1 - beta))
        return VolAtm
    
    # for other strikes
    z = (nu / alpha) * ((F * X) ** (0.5 * (1 - beta))) * np.log(F / X)
    inside_log = ((1 - 2 * rho * z + z * z) ** 0.5 + z - rho) / (1 - rho)
    
    if inside_log <= 0:
        raise ValueError("Invalid value for logarithm: value must be positive.")
    
    zhi = np.log(inside_log)
    numer1 = (((1 - beta) ** 2) / 24) * ((alpha * alpha) / ((F * X) ** (1 - beta)))
    numer2 = 0.25 * rho * beta * nu / ((F * X) ** ((1 - beta) / 2))
    numer3 = ((2 - 3 * rho * rho) / 24) * nu * nu
    numer = alpha * (1 + (numer1 + numer2 + numer3) * T) * z
    denom1 = ((1 - beta) ** 2 / 24) * (np.log(F / X)) ** 2
    denom2 = (((1 - beta) ** 4) / 1920) * ((np.log(F / X)) ** 4)
    denom = ((F * X) ** ((1 - beta) / 2)) * (1 + denom1 + denom2) * zhi
    
    return numer / denom

def sabrcalibration(x, strikes, vols, F, T):
    """Calibrate SABR model parameters"""
    alpha, rho, nu = x
    beta = 0.7  # Fixed beta
    
    if not (0 < alpha and -1 < rho < 1 and nu > 0):
        return np.full_like(vols, np.inf)
    
    err = 0.0
    for i, vol in enumerate(vols):
        try:
            model_vol = SABR(F, strikes[i], T, alpha, beta, rho, nu)
            err += (vol - model_vol) ** 2
        except ValueError:
            err += np.inf
    return err

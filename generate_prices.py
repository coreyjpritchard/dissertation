from run_simulations_numba import simulate_gbm_numba, simulate_ms_garch_numba, simulate_garch_numba
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import os
import pandas as pd
from keras.models import load_model
from deep_sims import reg_sim, class_sim
from  scipy.interpolate import interp1d

class RiskFreeCurve:
    """
    Interpolates a Treasury term‑structure keyed in *trading* days.
    - Stores log‑discount factors (monotone  arbitrage‑free for a normal curve).
    - Linear in log‑D is plenty for money‑market work; swap desks often use
      cubic splines – just swap `interp1d(..., kind="linear")` for CubicSpline.
    """
    TDAYS_PER_YEAR = 252                      
    
    def __init__(self, rf_dict: dict[int, float]):
        # --- pillar data -----------------------------------------------------
        self.t   = np.array(sorted(rf_dict))                        # days
        y        = np.array([rf_dict[d] for d in self.t])           # yields
        self._logD = np.log(np.exp(-y * self.t / self.TDAYS_PER_YEAR))
        # --- interpolator ----------------------------------------------------
        self._ilogD = interp1d(self.t, self._logD,
                               kind="cubic",          
                               fill_value="extrapolate",
                               assume_sorted=True)

    # ---------- user‑facing API ----------------------------------------------
    def discount(self, t: int | float) -> float:
        "DF for any (fractional) trading‑day t"
        return np.exp(self._ilogD(t))
    
    def zero_rate(self, t: int | float) -> float:
        "Continuously‑compounded zero yield"
        return -np.log(self.discount(t)) * self.TDAYS_PER_YEAR / t
    
    __call__ = zero_rate                     

ticker = "SPY"
timeframe = "1d"

ms_garch_params = pd.read_csv(f"ms_garch_params.csv", index_col=0)

garch_params = pd.read_csv(f"garch_params.csv", index_col=0)
garch_mu = garch_params.loc["mu"].values[0]
garch_alpha = garch_params.loc["alpha[1]"].values[0]
garch_beta = garch_params.loc["beta[1]"].values[0]
garch_omega = garch_params.loc["omega"].values[0]

yf_ticker = yf.Ticker(ticker)
yf_data = yf_ticker.history(period="max", interval=timeframe)
yf_data["Log Return"] = np.log(yf_data["Close"] / yf_data["Close"].shift(1))
yf_data = yf_data[1:]

S0 = yf_data["Close"].iloc[-1]

# rf_rates = { #https://tradingeconomics.com/united-states/government-bond-yield - we said we collected this straight from US treasure gov
#     30: 4.38 / 100, # 1 month treasury yield
#     60: 4.36 / 100, # 2 month treasury yield
#     90: 4.42 / 100, # 3 month treasury yield
#     180: 4.29 / 100, # 6 month treasury yield
#     365: 4.11 / 100, # 1 year treasury yield
#     730: 3.98 / 100, # 2 year treasury yield
#     1095: 3.97 / 100, # 3 year treasury yield
# }

rf_rates = { # num_trade_days = round(num_real_days * 252 / 365) 
    21 : 4.32 / 100,
    41 : 4.37 / 100,
    62 : 4.40 / 100,
    124: 4.25 / 100,
    252: 4.12 / 100,
    504: 3.99 / 100,
    756: 3.98 / 100,
    1260: 4.08 / 100,
}

curve = RiskFreeCurve(rf_rates)

def get_closest_rate(dte, rf_rates):
    # sorted_terms = sorted(rf_rates.keys())
    # for term in sorted_terms:
    #     if dte <= term:
    #         return rf_rates[term]
    return curve(dte)
    
#unique_dtes = market_prices_df['DTE'].unique()


def get_all_option_prices(expiration, ticker=yf_ticker):
    option_chain = ticker.option_chain(expiration)
    calls = option_chain.calls
    puts = option_chain.puts
    
    
    merged = pd.merge(calls[['strike', 'bid', 'ask']], 
                      puts[['strike', 'bid', 'ask']], 
                      on='strike', 
                      suffixes=('_call', '_put'))
    
    
    merged['Call_Market_Price'] = (merged['bid_call'] + merged['ask_call']) / 2
    merged['Put_Market_Price'] = (merged['bid_put'] + merged['ask_put']) / 2
    
    return merged[['strike', 'Call_Market_Price', 'Put_Market_Price']]


data = []
expiration_dates = yf_ticker.options

for exp in expiration_dates:
    option_prices = get_all_option_prices(exp)
    option_prices['Expiration'] = exp
    data.append(option_prices)

market_prices_df = pd.concat(data, ignore_index=True)
market_prices_df['Expiration'] = pd.to_datetime(market_prices_df['Expiration'])

today = datetime.today()
today = datetime(today.year, today.month, today.day) # set it to midnight

market_prices_df['DTE'] = np.ceil((market_prices_df['Expiration'] - today).dt.days * 252 / 365).astype(int)

unique_dtes = market_prices_df['DTE'].unique()
dte_risk_free_rates = {dte: get_closest_rate(dte, rf_rates) for dte in unique_dtes}

# drop row where both call and put prices are 0
#market_prices_df = market_prices_df[(market_prices_df['Call_Market_Price'] != 0) | (market_prices_df['Put_Market_Price'] != 0)]

# save
market_prices_df.to_csv("final prices/market_prices.csv", index=False)
with open("final prices/S0.txt", "w") as f:
    f.write(str(S0))

unique_dtes = market_prices_df['DTE'].unique()
# drop 0dte
unique_dtes = unique_dtes[unique_dtes > 0]

div = yf_data["Dividends"]
prices = yf_data["Close"]

#div_yield = div[div > 0] / prices[div > 0] # Calculate dividend yield for each dividend payment
div_yield = div / prices
q = div_yield.mean() * 252 # Annualize the dividend yield
#make it continuous
q = np.log(1 + q)

sigma = np.std(yf_data["Log Return"]) * np.sqrt(252)


def get_closest_rate(dte, rf_rates):
    # sorted_terms = sorted(rf_rates.keys())
    # for term in sorted_terms:
    #     if dte <= term:
    #         return rf_rates[term]
    return curve(dte)

dte_risk_free_rates = {dte: get_closest_rate(dte, rf_rates) for dte in unique_dtes}
mu_Q_dict = {dte: (rf - q) for dte, rf in dte_risk_free_rates.items()} 

yf_data["Log Return Sq"] = yf_data["Log Return"] ** 2

num_rolling = [15, 30, 60, 180]
for i in num_rolling:
    yf_data[f"Rolling Mean {i}"] = yf_data["Log Return"].rolling(i).mean()
    yf_data[f"Rolling Std {i}"] = yf_data["Log Return"].rolling(i).std()
    yf_data[f"Rolling Skew {i}"] = yf_data["Log Return"].rolling(i).skew()
    yf_data[f"Rolling Kurt {i}"] = yf_data["Log Return"].rolling(i).kurt()

    yf_data[f"Rolling Vol Mean {i}"] = yf_data["Log Return Sq"].rolling(i).mean()
    yf_data[f"Rolling Vol Std {i}"] = yf_data["Log Return Sq"].rolling(i).std()
    yf_data[f"Rolling Vol Skew {i}"] = yf_data["Log Return Sq"].rolling(i).skew()
    yf_data[f"Rolling Vol Kurt {i}"] = yf_data["Log Return Sq"].rolling(i).kurt()


# high_low = yf_data["High"] - yf_data["Low"]
# high_close = (yf_data["High"] - yf_data["Close"].shift()).abs()
# low_close = (yf_data["Low"]  - yf_data["Close"].shift()).abs()
# tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
# yf_data["ATR 14"] = tr.rolling(14).mean()

# mid = yf_data["Close"].rolling(20).mean()
# std = yf_data["Close"].rolling(20).std()
# yf_data["BB Width"] = (mid + 2*std - (mid - 2*std)) / mid

# delta = yf_data["Close"].diff()
# up = delta.clip(lower=0)
# down = -delta.clip(upper=0)
# ema_up = up.ewm(span=14).mean()
# ema_dn = down.ewm(span=14).mean()
# yf_data["RSI 14"] = 100 - (100 / (1 + ema_up/ema_dn))

# yf_data["Z Score"] = (yf_data["Close"] - yf_data["Close"].rolling(120).mean()) / yf_data["Close"].rolling(120).std()

yf_data = yf_data[max(num_rolling):]    

features = [
    "Log Return",
    "Log Return Sq",
    # "ATR 14",
    # "BB Width",
    # "RSI 14",
    # "Z Score"
]


features += [f"Rolling Mean {i}" for i in num_rolling]
features += [f"Rolling Std {i}" for i in num_rolling]
features += [f"Rolling Skew {i}" for i in num_rolling]
features += [f"Rolling Kurt {i}" for i in num_rolling]

features += [f"Rolling Vol Mean {i}" for i in num_rolling]
features += [f"Rolling Vol Std {i}" for i in num_rolling]
features += [f"Rolling Vol Skew {i}" for i in num_rolling]
features += [f"Rolling Vol Kurt {i}" for i in num_rolling]

# features += [f"VolVol {i}" for i in num_rolling]
# features += [f"Vol Mean {i}" for i in num_rolling]



# keras_models folder

reg_model = load_model("keras_models/reg.keras")
class_model = load_model("keras_models/class.keras")

lookback = 60
initial_data = yf_data[features].iloc[-lookback:]

vol_map = { # 5 day std deviation of log returns - mean value per class
    0: 0.005810	 * np.sqrt(252/5),
    1: 0.016162* np.sqrt(252/5),
}

num_simulations = 10000

import pickle
with open("scalers/reg_features.pkl", "rb") as f:
    reg_scaler = pickle.load(f)
with open("scalers/class_features.pkl", "rb") as f:
    class_scaler = pickle.load(f)

for dte in unique_dtes:
    print(dte)
    if dte not in [30, 96]:
        continue
    mu_Q = mu_Q_dict[dte]
    mu_Q = np.log(1 + mu_Q) 

    # num_simulations = 1000000
    
    # gbm_return_path = simulate_gbm_numba(mu_Q, sigma, dte, num_simulations, seed=42)
    # gbm_final_prices = S0 * np.exp(gbm_return_path[-1, :])
    # np.save(f"final prices/gbm_final_prices_{dte}.npy", gbm_final_prices)
    # del gbm_return_path, gbm_final_prices
    # gc.collect()

    # ms_garch_return_path = simulate_ms_garch_numba(mu_Q, ms_garch_params, dte, num_simulations, seed=42)
    # ms_garch_prices = S0 * np.exp(ms_garch_return_path[-1, :])
    # np.save(f"final prices/ms_garch_final_prices_{dte}.npy", ms_garch_prices)
    # del ms_garch_return_path, ms_garch_prices
    # gc.collect()

    # garch_return_path = simulate_garch_numba(mu_Q, garch_mu, garch_omega, garch_alpha, garch_beta, dte, num_simulations, seed=42)
    # garch_prices = S0 * np.exp(garch_return_path[-1, :])
    # np.save(f"final prices/garch_final_prices_{dte}.npy", garch_prices)
    # del garch_return_path, garch_prices
    # gc.collect()

    num_simulations = 10000

    reg_lr_paths = reg_sim(reg_model, reg_scaler, initial_data,mu_Q, dte, num_simulations, lookback=None, seed=None)
    reg_finals = S0 * np.exp(reg_lr_paths[-1,:])
    np.save(f"final prices/reg_final_prices_{dte}.npy", reg_finals)
    del reg_lr_paths, reg_finals
    gc.collect()

    class_lr_paths = class_sim(class_model, class_scaler, initial_data, vol_map, mu_Q, dte, num_simulations, lookback=None, seed=None)
    class_finals = S0 * np.exp(class_lr_paths[-1,:])
    np.save(f"final prices/class_final_prices_{dte}.npy", class_finals)
    del class_lr_paths, class_finals
    gc.collect()

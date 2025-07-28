import numba as nb
import numpy as np
import pandas as pd

# I gave my functions to GPT-o3 and asked it to optimize using numba

@nb.njit(parallel=True, fastmath=True)
def _simulate_gbm_core(mu_Q, sigma, n_days, n_paths):
    dt      = 1.0 / 252.0
    drift   = (mu_Q - 0.5 * sigma * sigma) * dt
    vol     = sigma * np.sqrt(dt)

    # allocate [n_days × n_paths]
    out = np.empty((n_days, n_paths), dtype=np.float64)

    # Generate all random shocks at once
    shocks = np.random.randn(n_days, n_paths)

    # first row
    out[0, :] = drift + vol * shocks[0, :]

    # remaining rows (cumulative sum)
    for t in range(1, n_days):
        out[t, :] = out[t - 1, :] + drift + vol * shocks[t, :]

    return out


def simulate_gbm_numba(mu_Q_annual, sigma_annual, n_days=252, n_paths=10_000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    log_paths = _simulate_gbm_core(mu_Q_annual, sigma_annual, n_days, n_paths)
    return log_paths
    # df = pd.DataFrame(log_paths, index=np.arange(1, n_days + 1))
    # df.columns = [f"Sim_{i+1}" for i in range(n_paths)]
    # return df



def _prep_params(params_df):
    return (params_df["Mean"].values.astype(np.float64),
            params_df["Omega"].values.astype(np.float64),
            params_df["Alpha"].values.astype(np.float64),
            params_df["Beta"].values.astype(np.float64),
            params_df["p"].values.astype(np.float64))


@nb.njit
def sample_discrete(p): #np.random.choice not supported
    u = np.random.rand()
    cumulative = 0.0
    for i in range(p.shape[0]):
        cumulative += p[i]
        if u < cumulative:
            return i
    return p.shape[0] - 1  # catch rounding errors


@nb.njit(parallel=True, fastmath=True)
def _simulate_ms_garch_core(mu_Q, mu, omegas, alphas, betas, p_stay,
                            n_days, n_paths):
    m = omegas.shape[0]
    dt = 1.0 / 252.0

    # -------- transition matrix -------------
    P = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        P[i, i] = p_stay[i]
        if m > 1:
            off = (1.0 - p_stay[i]) / (m - 1)
            for j in range(m):
                if i != j:
                    P[i, j] = off

    # -------- stationary distribution --------
    if m == 1:
        stationary = np.array([1.0], dtype=np.float64)
    else:
        stationary = np.ones(m, dtype=np.float64) / m
        for _ in range(50):                     # quick power‑iteration
            stationary = stationary @ P

    # -------- output array -------------------
    out = np.empty((n_days, n_paths), dtype=np.float64)


    # -------- main parallel loop -------------
    for path in nb.prange(n_paths):

        #state = np.random.choice(m, p=stationary)
        state = sample_discrete(stationary)

        # if alphas[state] + betas[state] < 1.0:
        #     prev_var = omegas[state] / (1.0 - alphas[state] - betas[state])
        # else:
        #     prev_var = omegas[state]

        prev_var = omegas[state] / (1.0 - alphas[state] - betas[state])

        prev_ret = 0.0
        cum_ret  = 0.0

        for t in range(n_days):
            if t > 0:
                #state = np.random.choice(m, p=P[state])
                state = sample_discrete(P[state])

            prev_eps = prev_ret - mu[state] # mu_Q * dt 
            var   = omegas[state] + alphas[state] * prev_eps**2 + betas[state] * prev_var
            drift = (mu_Q * dt) - 0.5 * var #* dt
            sd    = np.sqrt(var)

            r_t    = drift + sd * np.random.randn()
            cum_ret += r_t
            out[t, path] = cum_ret

            prev_ret = r_t
            prev_var = var

    return out

def simulate_ms_garch_numba(mu_Q_annual, params_df,
                            n_days=252, n_paths=10_000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mu, omegas, alphas, betas, p_stay = _prep_params(params_df)
    log_paths = _simulate_ms_garch_core(
        mu_Q_annual, mu, omegas, alphas, betas, p_stay,
        n_days, n_paths
    )
    return log_paths
    # df = pd.DataFrame(log_paths, index=np.arange(1, n_days + 1))
    # df.columns = [f"Sim_{i+1}" for i in range(n_paths)]
    # return df


import numpy as np
import pandas as pd
import numba as nb

# --------------------------------------------------------------------- #
#  NUMBA JIT KERNEL                                                     #
# --------------------------------------------------------------------- #
@nb.njit(parallel=True, fastmath=True)
def _simulate_garch_core(mu_Q_annual: float,
                         mu: float,
                         omega: float,
                         alpha: float,
                         beta:  float,
                         n_days: int,
                         n_paths: int) -> np.ndarray:
    """
    Fast Monte‑Carlo generator for a *single‑regime* GARCH(1,1) model
    under the risk‑neutral measure.

    Parameters
    ----------
    mu_Q_annual : float
        Annualised (r – q).  Only this term is re‑scaled by dt.
    omega, alpha, beta : float
        GARCH(1,1) parameters estimated on **daily** log‑returns.
    n_days : int
        Horizon in trading days.
    n_paths : int
        Number of independent paths.
    Returns
    -------
    np.ndarray  shape = (n_days, n_paths)
        Cumulative log‑return paths (no price conversion).
    """

    dt        = 1.0 / 252.0
    logpaths  = np.empty((n_days, n_paths), dtype=np.float64)

    # long‑run variance as starting point
    if alpha + beta < 1.0:
        var0 = omega / (1.0 - alpha - beta)
    else:                       # non‑stationary fallback
        var0 = omega

    # ------------------------------------------------------------------
    for p in nb.prange(n_paths):

        prev_var   = var0
        prev_r     = 0.0
        cum_return = 0.0

        for t in range(n_days):
            prev_eps = prev_r - mu #mu_Q_annual * dt#mu # 
            variance = omega + alpha * prev_eps * prev_eps + beta * prev_var
            sd       = np.sqrt(variance)              # DAILY σ
            drift    = mu_Q_annual * dt - 0.5 * variance   # DAILY drift adj.

            r_t = drift + sd * np.random.randn()
            cum_return += r_t
            logpaths[t, p] = cum_return

            prev_var = variance
            prev_r   = r_t

    return logpaths


# --------------------------------------------------------------------- #
#  USER‑FRIENDLY WRAPPER                                                #
# --------------------------------------------------------------------- #
def simulate_garch_numba(mu_Q_annual: float,
                         mu: float,
                         omega: float,
                         alpha: float,
                         beta: float,
                         n_days: int = 252,
                         n_paths: int = 100_000,
                         seed: int | None = None) -> pd.DataFrame:
    """
    Wrapper that seeds NumPy, calls the Numba kernel, and returns a DataFrame.

    Returns
    -------
    pd.DataFrame
        Index = 1…n_days,  columns = Sim_1 … Sim_n_paths
    """
    if seed is not None:
        np.random.seed(seed)

    logpaths = _simulate_garch_core(mu_Q_annual, mu, omega, alpha, beta,
                                    n_days, n_paths)

    return logpaths
    # df = pd.DataFrame(
    #     logpaths,
    #     index=np.arange(1, n_days + 1),
    #     columns=[f"Sim_{i+1}" for i in range(n_paths)]
    # )
    # return df

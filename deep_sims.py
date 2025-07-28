import numpy as np

def class_sim(model, scaler, initial_feature_window, 
                                        class_vol_mapping, mu_Q_annual, 
                                        n_days, n_paths, lookback=None, seed=None):
    """
    Monte Carlo simulation of daily log returns using a classification volatility model.
    
    Parameters:
        model (keras.Model): Trained Keras classification model that outputs either 
            class probabilities or a class label for volatility regime.
        scaler (object): Scaler for feature normalisation (same as used in model training).
        initial_feature_window (pd.DataFrame or np.ndarray): Last `lookback` days of features 
            (unscaled) for initialisation.
        class_vol_mapping (dict): Mapping from class index (int) to annualised volatility (float). 
            E.g., {0: 0.10, 1: 0.20, 2: 0.30} for 10%, 20%, 30% annual vol regimes.
        mu_Q_annual (float): Annualised risk-neutral drift.
        n_days (int): Number of days to simulate.
        n_paths (int): Number of simulation paths.
        lookback (int, optional): Length of feature window to use (infer from data if None).
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Simulated log returns, shape (n_days, n_paths).
    """
    rng = np.random.default_rng(seed)
    if lookback is None:
        lookback = initial_feature_window.shape[0]
    if initial_feature_window.shape[0] < lookback:
        raise ValueError(f"Initial feature window must have at least {lookback} rows")
    # Initialise feature window for all paths
    if isinstance(initial_feature_window, np.ndarray):
        initial_window = initial_feature_window[-lookback:].copy()
    else:
        initial_window = initial_feature_window.iloc[-lookback:].values.copy()
    n_features = initial_window.shape[1]
    simulated_returns = np.zeros((n_days, n_paths))
    windows = np.tile(initial_window[np.newaxis, :, :], (n_paths, 1, 1))
    daily_drift = mu_Q_annual * (1.0/252)
    
    for day in range(n_days):
        # Prepare inputs (scaled) for model prediction
        window_batch = windows
        if scaler is not None:
            orig_shape = window_batch.shape
            window_batch_scaled = scaler.transform(window_batch.reshape(-1, n_features))
            window_batch_scaled = window_batch_scaled.reshape(orig_shape)
        else:
            window_batch_scaled = window_batch
        # Predict volatility class for each path
        class_preds = model.predict(window_batch_scaled)  # shape (n_paths, n_classes) or (n_paths,) if already argmax
        # Determine class indices:
        if class_preds.ndim > 1 and class_preds.shape[1] > 1:
            # If we have probability distribution output
            class_indices = np.argmax(class_preds, axis=1)
        else:
            # If model gives a single output per sample (e.g., already an encoded class or probability of class 1 in binary case)
            class_indices = class_preds.reshape(-1).astype(int)
        # Map each class index to its annualised volatility and then to daily vol
        sigma_ann = np.array([class_vol_mapping.get(cls, 0.0) for cls in class_indices])
        sigma_daily = sigma_ann * np.sqrt(1.0/252.0)  # convert annual vol to one-day vol:contentReference[oaicite:11]{index=11}
        # Simulate returns: r = drift*dt + sigma_daily * Z  (drift*dt is already daily_drift)
        Z = rng.standard_normal(n_paths)
        returns_day = daily_drift + sigma_daily * Z * np.sqrt(1.0/252.0)
        # Note: Multiplying by sqrt(1/252) again here would double-count the sqrt(dt). We should use either:
        # returns_day = daily_drift + sigma_ann * np.sqrt(1.0/252.0) * Z 
        # OR as above: compute sigma_daily then returns_day = daily_drift + sigma_daily * Z.
        # To avoid confusion, let's do: returns_day = daily_drift + sigma_ann * np.sqrt(1.0/252.0) * Z.
        returns_day = daily_drift + sigma_ann * np.sqrt(1.0/252.0) * Z
        simulated_returns[day, :] = returns_day
        
        # Update features for each path
        try:
            columns = (initial_feature_window.columns if hasattr(initial_feature_window, 'columns') 
                       else [str(i) for i in range(n_features)])
        except Exception:
            columns = [str(i) for i in range(n_features)]
        # Identify return and volatility feature indices as before
        return_idx = None
        vol_feat_idx = None
        for idx, col in enumerate(columns):
            col_lower = str(col).lower()
            if return_idx is None and "return" in col_lower and "sq" not in col_lower:
                return_idx = idx
            if vol_feat_idx is None and "vol" in col_lower:
                # We take "vol" to indicate a volatility feature (could be previous day's vol or similar)
                vol_feat_idx = idx
        for p in range(n_paths):
            windows[p, :-1, :] = windows[p, 1:, :]
            new_row = windows[p, -1, :].copy()
            # Update previous return
            if return_idx is not None:
                new_row[return_idx] = returns_day[p]
            # Update volatility-related feature (if any) with the regime's daily volatility (or some representation of it)
            if vol_feat_idx is not None:
                new_row[vol_feat_idx] = sigma_daily[p]  # or sigma_ann[p] if features expect annual vol
            # Recompute rolling stats similarly to regression case:
            data_window = windows[p, :, :]
            if return_idx is not None:
                data_window[-1, return_idx] = returns_day[p]
            # (For classification, squared returns might not explicitly be a feature, but if they are, update similarly)
            # Compute rolling moments for specified window lengths
            window_lengths = [15, 30, 60, 180]
            window_lengths = [w for w in window_lengths if w <= data_window.shape[0]]
            return_series = data_window[:, return_idx] if return_idx is not None else None
            # (If squared return features exist, identify and update them as in regression function.)
            for w in window_lengths:
                if return_series is not None:
                    recent_returns = return_series[-w:]
                    r_mean = np.mean(recent_returns)
                    r_std = np.std(recent_returns, ddof=0)
                    if r_std == 0:
                        r_skew = 0.0
                        r_kurt = 0.0
                    else:
                        m3 = np.mean((recent_returns - r_mean)**3)
                        m2 = r_std**2
                        m4 = np.mean((recent_returns - r_mean)**4)
                        r_skew = m3 / (m2**1.5)
                        r_kurt = m4 / (m2**2)
                    for idx, col in enumerate(columns):
                        col_lower = str(col).lower()
                        if f"mean{w}" in col_lower:
                            new_row[idx] = r_mean
                        elif f"std{w}" in col_lower:
                            new_row[idx] = r_std
                        elif f"skew{w}" in col_lower:
                            new_row[idx] = r_skew
                        elif f"kurt{w}" in col_lower:
                            new_row[idx] = r_kurt
            # Update window with new features
            windows[p, -1, :] = new_row
    # End simulation loop
    return simulated_returns

import numpy as np
# Optionally import scipy.stats for skew and kurtosis if needed:
# from scipy.stats import skew, kurtosis

def reg_sim(model, scaler, initial_feature_window, 
                                    mu_Q_annual, n_days, n_paths, 
                                    lookback=None, seed=None):
    """
    Monte Carlo simulation of daily log returns using a regression volatility model.
    
    Parameters:
        model (keras.Model): Trained Keras regression model that outputs an annualised 
            5-day volatility estimate for the next day.
        scaler (sklearn.preprocessing.StandardScaler or similar): Scaler used to 
            normalise features for the model. The inverse transform is not needed 
            since we update features in original scale and then re-normalise for predictions.
        initial_feature_window (pd.DataFrame or np.ndarray): The latest `lookback` days 
            of features (unscaled) to initialize the rolling window. Its shape should be 
            (lookback, n_features). This includes recent rolling stats and other features.
        mu_Q_annual (float): Annualised drift under the risk-neutral measure Q (e.g. risk-free rate).
        n_days (int): Number of trading days to simulate.
        n_paths (int): Number of independent simulation paths.
        lookback (int, optional): Length of the feature window (if None, it will be 
            inferred from `initial_feature_window`).
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Simulated daily log returns of shape (n_days, n_paths).
        (Optional) np.ndarray: Full simulated returns history for each path including 
            the initial window, if needed for further feature computations.
    """
    rng = np.random.default_rng(seed)
    # Determine window length
    if lookback is None:
        lookback = initial_feature_window.shape[0]
    if initial_feature_window.shape[0] < lookback:
        raise ValueError(f"Initial feature window must have at least {lookback} rows")
    # Start with the provided feature window (use a copy to avoid altering original data)
    if isinstance(initial_feature_window, np.ndarray):
        initial_window = initial_feature_window[-lookback:].copy()
    else:
        initial_window = initial_feature_window.iloc[-lookback:].values.copy()
    n_features = initial_window.shape[1]
    # Set up storage for simulation results
    simulated_returns = np.zeros((n_days, n_paths))
    # Replicate the initial window for each path (3D array: path × time × features)
    windows = np.tile(initial_window[np.newaxis, :, :], (n_paths, 1, 1))
    # Pre-compute daily drift
    daily_drift = mu_Q_annual * (1.0/252)  # = mu_Q_annual * dt
    
    # If we want to store full return histories (initial real + simulated) per path:
    # returns_history = np.zeros((n_paths, lookback + n_days))
    # If initial_window includes a column for actual returns, we can initialise returns_history[:, :lookback] with that.
    
    for day in range(n_days):
        # Prepare model inputs: normalise each path's current feature window
        window_batch = windows  # shape (n_paths, lookback, n_features)
        # Flatten for scaler transform (if provided), then reshape back
        if scaler is not None:
            orig_shape = window_batch.shape
            window_batch_scaled = scaler.transform(window_batch.reshape(-1, n_features))
            window_batch_scaled = window_batch_scaled.reshape(orig_shape)
        else:
            window_batch_scaled = window_batch
        # Model prediction: forecast annualised 5-day volatility for each path
        vol_pred_ann5 = model.predict(window_batch_scaled)  # shape (n_paths, 1) or (n_paths,) depending on model
        vol_pred_ann5 = vol_pred_ann5.reshape(-1)  # ensure shape (n_paths,)
        # Convert annualised 5-day vol to daily vol (one-day std dev of returns)
        sigma_daily = vol_pred_ann5 / np.sqrt(252.0/5.0)  # using √time scaling:contentReference[oaicite:5]{index=5}
        # Simulate returns for all paths for this day
        Z = rng.standard_normal(n_paths)  # independent N(0,1) shocks
        # If sigma_daily is the actual daily std, we can use: return = daily_drift + sigma_daily * Z
        # (daily_drift already incorporates dt, and sigma_daily is one-day vol)
        returns_day = daily_drift + sigma_daily * Z
        simulated_returns[day, :] = returns_day
        
        # Update the feature windows for each path with the new day's data
        # First, maintain a history of recent returns for rolling calculations:
        # We identify which column in the features (if any) corresponds to raw returns and squared returns.
        # For robustness, we can compute rolling stats directly from a returns history that we maintain.
        # Here, we will derive the needed rolling statistics from the simulated returns and possibly initial real returns.
        
        # Option 1: If we know the index of return and squared return in feature vector:
        # Find columns that contain 'return' (for raw log return) and 'sq_return' or similar for squared returns.
        # We will assume there is at least one column for raw returns in the feature set.
        try:
            columns = (initial_feature_window.columns if hasattr(initial_feature_window, 'columns') 
                       else [str(i) for i in range(n_features)])
        except Exception:
            columns = [str(i) for i in range(n_features)]
        
        # Identify the index of the raw return and squared return in the feature vector, if present
        return_idx = None
        sq_return_idx = None
        for idx, col in enumerate(columns):
            col_lower = str(col).lower()
            if return_idx is None and "return" in col_lower and "sq" not in col_lower:
                return_idx = idx
            if sq_return_idx is None and ("sq_return" in col_lower or "squared" in col_lower):
                sq_return_idx = idx
        
        # If we have not been maintaining a separate returns history array, we can extract the needed past returns
        # from the feature window itself if the first feature is the log return.
        # Let's assume `return_idx` gives the position of log return in the feature vector.
        
        for p in range(n_paths):
            # Shift window left (drop oldest day)
            windows[p, :-1, :] = windows[p, 1:, :]
            # Prepare new feature row (initialised as copy of previous last row to carry over unchanged features)
            new_row = windows[p, -1, :].copy()
            
            # Update return-based features
            if return_idx is not None:
                new_row[return_idx] = returns_day[p]
            if sq_return_idx is not None:
                new_row[sq_return_idx] = returns_day[p]**2
            
            # Now recompute rolling statistics for each window length of interest
            # We assume that the feature vector contains rolling stats for various windows (e.g., 15, 30, 60, 180 days).
            # We will recompute those by using the recent returns from the window (including the new return).
            # To do this properly, we might need the actual past returns beyond just the last 'lookback-1' days contained in window after shift.
            # If lookback is large enough to cover the largest rolling window (e.g., 180), we can use the data in `windows[p]`.
            data_window = windows[p, :, :]  # after shifting, this still contains the old last row in last position, which we've copied to new_row
            # First, temporarily set the last position's return to the new return (so we can compute stats including it)
            if return_idx is not None:
                data_window[-1, return_idx] = returns_day[p]
            if sq_return_idx is not None:
                data_window[-1, sq_return_idx] = returns_day[p]**2
            
            # Now compute rolling moments for returns and squared returns for each relevant window length
            # Define window lengths of interest (ensure they are <= lookback)
            window_lengths = [15, 30, 60, 180]
            window_lengths = [w for w in window_lengths if w <= data_window.shape[0]]
            # Extract the return series from the data window
            return_series = data_window[:, return_idx] if return_idx is not None else None
            # Compute rolling stats for each window length
            for w in window_lengths:
                if return_series is not None:
                    recent_returns = return_series[-w:]  # last w returns including new one
                    # Compute mean, std, skew, kurtosis of recent_returns
                    r_mean = np.mean(recent_returns)
                    r_std = np.std(recent_returns, ddof=0)
                    # Skewness and kurtosis (population versions, bias=True)
                    if r_std == 0:
                        r_skew = 0.0
                        r_kurt = 0.0
                    else:
                        m3 = np.mean((recent_returns - r_mean)**3)
                        m2 = r_std**2  # variance
                        m4 = np.mean((recent_returns - r_mean)**4)
                        r_skew = m3 / (m2**1.5)  # Fisher-Pearson coefficient:contentReference[oaicite:6]{index=6}
                        r_kurt = m4 / (m2**2)    # Kurtosis including 3 for normal:contentReference[oaicite:7]{index=7}
                    # Now assign these values to the appropriate feature indices if we know them.
                    # We assume feature names encode the window length and stat, e.g., "mean_15", "std_15", "skew_15", "kurt_15".
                    for idx, col in enumerate(columns):
                        col_lower = str(col).lower()
                        if f"mean{w}" in col_lower or f"mean_{w}" in col_lower:
                            new_row[idx] = r_mean
                        elif f"std{w}" in col_lower or f"std_{w}" in col_lower:
                            new_row[idx] = r_std
                        elif f"skew{w}" in col_lower or f"skew_{w}" in col_lower:
                            new_row[idx] = r_skew
                        elif f"kurt{w}" in col_lower or f"kurt_{w}" in col_lower:
                            new_row[idx] = r_kurt
                # Similarly, if squared-return features exist (other than just mean and std of squared returns):
                if sq_return_idx is not None:
                    recent_sq_returns = data_window[-w:, sq_return_idx]
                    sq_mean = np.mean(recent_sq_returns)
                    sq_std = np.std(recent_sq_returns, ddof=0)
                    for idx, col in enumerate(columns):
                        col_lower = str(col).lower()
                        if f"sq_mean{w}" in col_lower or f"squared_mean_{w}" in col_lower:
                            new_row[idx] = sq_mean
                        elif f"sq_std{w}" in col_lower or f"squared_std_{w}" in col_lower:
                            new_row[idx] = sq_std
            # End for each window length
            
            # Assign the fully updated new_row to the window (last position)
            windows[p, -1, :] = new_row
            # (If we maintained returns_history, we would append returns_day[p] to it here as well)
        # End for each path
    # End for each day
    
    return simulated_returns  # shape (n_days, n_paths)
    # If needed: also return returns_history or final windows for analysis

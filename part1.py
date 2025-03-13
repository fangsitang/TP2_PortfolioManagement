import pandas as pd

# DATA CLEANING
# region

# read returns file
df_rets = pd.read_csv('10_Industry_Portfolios.csv', index_col=0)
df_rets.index = df_rets.index.astype(str)
df_rets.index = df_rets.index.str[:4] + '-' + df_rets.index.str[4:]

# read risk-free rate file
df_rates = pd.read_csv('F-F_Research_Data_Factors.csv',index_col=0)
df_rates.index = df_rates.index.astype(str)
df_rates.index = df_rates.index.str[:4] + df_rets.index.str[4:]

df_rets = df_rets.merge(df_rates[['Mkt-RF']], left_index=True, right_index=True, how='left')
df_rets = df_rets.loc["1931-06":] #
# endregion


# PART A

# 1--MAX SHARPE RATIO
# region
import numpy as np
import pandas as pd

# Compute excess returns
excess_returns = df_rets.iloc[:, :-1].sub(df_rets["Mkt-RF"], axis=0)  # Subtract risk-free rate


# Function to compute max Sharpe portfolio weights
def max_sharpe_weights(returns):
    mean_ret = returns.mean()
    cov_matrix = returns.cov()
    inv_cov = np.linalg.inv(cov_matrix)

    # Compute tangency portfolio weights (unconstrained Markowitz solution)
    ones = np.ones(len(mean_ret))
    w_tangency = inv_cov @ mean_ret / (ones @ inv_cov @ mean_ret)

    return w_tangency


# Rolling window (5 years = 60 months)
window = 60
weights_list = []

for i in range(window, len(excess_returns)):
    rolling_data = excess_returns.iloc[i - window:i]  # Select 60 months of data
    weights = max_sharpe_weights(rolling_data)  # Compute optimal weights
    weights_list.append(weights)

# Convert list to DataFrame
dates = excess_returns.index[window:]
weights_df = pd.DataFrame(weights_list, index=dates, columns=excess_returns.columns)

print(weights_df.head())

# endregion

# 2--MAX SHARPE RATIO, SHORT-SALE CONSTRAINED
# region
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Compute excess returns
excess_returns = df_rets.iloc[:, :-1].sub(df_rets["Mkt-RF"], axis=0)  # Subtract risk-free rate


# Function to compute negative Sharpe ratio (for minimization)
def neg_sharpe(weights, mean_ret, cov_matrix, rf):
    port_return = np.dot(weights, mean_ret)  # Expected portfolio return
    port_vol = np.sqrt(weights @ cov_matrix @ weights.T)  # Portfolio standard deviation
    sharpe_ratio = (port_return - rf) / port_vol
    return -sharpe_ratio  # Negative Sharpe to minimize


# Rolling window (5 years = 60 months)
window = 60
weights_list = []

for i in range(window, len(excess_returns)):
    rolling_data = excess_returns.iloc[i - window:i]  # Select 60 months of data
    mean_ret = rolling_data.mean()
    cov_matrix = rolling_data.cov()
    rf = df_rets["Mkt-RF"].iloc[i]  # Risk-free rate of current month

    # Initial guess (equal weights)
    num_assets = len(mean_ret)
    init_guess = np.ones(num_assets) / num_assets

    # Define bounds (no short-selling)
    bounds = [(0, 1) for _ in range(num_assets)]  # Weights between 0 and 1

    # Constraint: Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Optimize using SLSQP (local search)
    result = minimize(neg_sharpe, init_guess, args=(mean_ret, cov_matrix, rf),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    weights_list.append(result.x)  # Store optimized weights

# Convert list to DataFrame
dates = excess_returns.index[window:]
weights_df = pd.DataFrame(weights_list, index=dates, columns=excess_returns.columns)

# Display sample results
print(weights_df.head())

# endregion

# 3--ASSET INVERSELY RELATED TO VARIANCE
# region
import numpy as np
import pandas as pd

# Compute excess returns (subtract risk-free rate)
excess_returns = df_rets.iloc[:, :-1].sub(df_rets["Mkt-RF"], axis=0)

# Rolling window (5 years = 60 months)
window = 60
weights_list = []

for i in range(window, len(excess_returns)):
    rolling_data = excess_returns.iloc[i - window:i]  # Select past 60 months

    # Compute variances of each asset
    variances = rolling_data.var()

    # Compute inverse variance weights
    inv_var_weights = 1 / variances
    inv_var_weights /= inv_var_weights.sum()  # Normalize to sum to 1

    weights_list.append(inv_var_weights.values)  # Store weights

# Convert list to DataFrame
dates = excess_returns.index[window:]
weights_df = pd.DataFrame(weights_list, index=dates, columns=excess_returns.columns)

print(weights_df.head())

# endregion

#4--WEIGHTS INVERSELY RELATED TO VOLATILITY
# region
import numpy as np
import pandas as pd

# Compute excess returns (subtract risk-free rate)
excess_returns = df_rets.iloc[:, :-1].sub(df_rets["Mkt-RF"], axis=0)

# Rolling window (5 years = 60 months)
window = 60
weights_list = []

for i in range(window, len(excess_returns)):
    rolling_data = excess_returns.iloc[i - window:i]  # Select past 60 months

    # Compute standard deviations of each asset
    volatilities = rolling_data.std()

    # Compute inverse volatility weights
    inv_vol_weights = 1 / volatilities
    inv_vol_weights /= inv_vol_weights.sum()  # Normalize to sum to 1

    weights_list.append(inv_vol_weights.values)  # Store weights

# Convert list to DataFrame
dates = excess_returns.index[window:]
weights_df = pd.DataFrame(weights_list, index=dates, columns=excess_returns.columns)

# Display sample results
print(weights_df.head())

# endregion

#5--ASSETS HAVE SAME WEIGHT
# All assets 10%.

#6--WEIGHTS LINEARLY RELATED TO MARKET CAP


#7--MINIMUM VARIANCE PF
# region
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Initialize a list to store the minimum variance portfolio weights
min_variance_weights = []


# Function to calculate the portfolio variance given the weights and covariance matrix
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights


# Iterate through each month starting from the 5th year (since it's a 5-year rolling window)
for month in df_rets.index[60:]:  # Start from 5 years after June 1931
    # Define the rolling window (5 years = 60 months)
    start_date = str(int(month[:4]) - 5) + month[4:]  # Adjust the year for the rolling window

    # Select the relevant window of returns (5 years of monthly returns)
    window_rets = df_rets.loc[start_date:month]

    # Calculate the covariance matrix for the returns over the window
    cov_matrix = window_rets.cov()

    # Initial guess for portfolio weights (equal weights)
    initial_weights = np.ones(len(window_rets.columns)) / len(window_rets.columns)

    # Constraint that the sum of weights must equal 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds for weights: weights should be between 0 and 1
    bounds = [(0, 1) for _ in range(len(window_rets.columns))]

    # Minimize the portfolio variance
    result = minimize(portfolio_variance, initial_weights, args=(cov_matrix,), bounds=bounds, constraints=constraints)

    # Store the weights of the minimum variance portfolio for this month
    min_variance_weights.append(result.x)

# Convert the minimum variance portfolio weights to DataFrame for easier interpretation
df_min_variance_weights = pd.DataFrame(min_variance_weights, index=df_rets.index[60:], columns=df_rets.columns)

# Print or return the minimum variance portfolio weights
print(df_min_variance_weights)

# endregion




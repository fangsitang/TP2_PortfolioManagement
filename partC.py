import pandas as pd
import numpy as np
from ccxt.static_dependencies.lark.parsers.cyk import print_parse

# DATA CLEANING
# region

# read 48 industry file
file_path = '48_industry_portfolios.xlsx'

# Read each sheet into a separate DataFrame
df_number_firms = pd.read_excel(file_path, sheet_name='number_firms', index_col=0)
df_firm_size = pd.read_excel(file_path, sheet_name='firm_size', index_col=0)
df_sum_BE = pd.read_excel(file_path, sheet_name='sum_BE', index_col=0)
df_avg_ind_rets = pd.read_excel(file_path, sheet_name='avg_ind_rets', index_col=0)

# Convert index to string format YYYY-MM
for df in [df_number_firms, df_firm_size,df_sum_BE, df_avg_ind_rets]:
    df.index = df.index.astype(str)
    df.index = df.index.str[:4] + '-' + df.index.str[4:]
    df.replace(-99.99, 0, inplace=True)

# Convert index to string format YYYY
df_sum_BE
df_sum_BE.index = df_sum_BE.index.astype(str)
df_sum_BE.index = df_sum_BE.index.str[:4]
df_sum_BE.replace(-99.99, 0, inplace=True)

# Display the first few rows of each DataFrame
print(df_number_firms.head())
print(df_firm_size.head())
print(df_sum_BE.head())
print(df_avg_ind_rets.head())


# endregion

# 1--MARKET CAP
# region
df_market_cap = df_firm_size * df_number_firms
print(df_market_cap.head())

# endregion

# 2--BOOK TO MARKET RATIO
# region

# Expand annual Book-to-Market ratio (Sum BE / Sum ME) to monthly values
bm_monthly = []
for year in df_sum_BE.index.astype(int):
    for month in range(7, 13):  # July to December of year s
        bm_monthly.append([f"{year}-{month:02d}"] + list(df_sum_BE.loc[str(year)]))
    for month in range(1, 7):  # January to June of year s+1
        bm_monthly.append([f"{year+1}-{month:02d}"] + list(df_sum_BE.loc[str(year)]))

# Convert to DataFrame with YYYY-MM index
df_BM_monthly = pd.DataFrame(bm_monthly, columns=["Date"] + list(df_sum_BE.columns))
df_BM_monthly.set_index("Date", inplace=True)

# Ensure df_BM_monthly index matches df_market_cap index
df_BM_monthly = df_BM_monthly.loc[df_market_cap.index]

# Compute monthly Book-to-Market ratio for each industry
df_book_to_market = df_BM_monthly * df_market_cap  # Element-wise multiplication

print(df_book_to_market.head())

# endregion


# 3--MOMENTUM
# region
# Compute momentum for each industry
df_momentum = pd.DataFrame(index=df_avg_ind_rets.index, columns=df_avg_ind_rets.columns)

# Iterate over each month and industry to compute momentum
for date in df_avg_ind_rets.index:
    # Get the last 12 months' data up to the current month
    momentum_data = df_avg_ind_rets.loc[df_avg_ind_rets.index <= date].tail(12)

    # Calculate the average of the last 12 months for each industry
    df_momentum.loc[date] = momentum_data.mean(axis=0)

print(df_momentum.head())
# endregion


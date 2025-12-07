import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ----------------------------------------------------
# Project paths
# ----------------------------------------------------

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
weekly_returns_file_path = PROJECT_ROOT / "outputs" / "complete_lumped_daily_returns.csv"
output_dir = PROJECT_ROOT / "outputs" / "max_correlation_matrices"

os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

# Convert stringified lists back into lists
converters = {
    col: eval
    for col in pd.read_csv(weekly_returns_file_path, nrows=1).columns
    if col != "Date"
}

weekly_returns_df = pd.read_csv(
    weekly_returns_file_path,
    index_col="Date",
    converters=converters,
    parse_dates=True,
)

# ----------------------------------------------------
# Correlation + lag computation
# ----------------------------------------------------

def generate_weekly_correlations_with_lags(weekly_returns_df, max_lag):
    dates = weekly_returns_df.index

    for i, current_date in enumerate(dates):
        week_data = weekly_returns_df.loc[current_date]

        correlation_matrix = pd.DataFrame(
            index=weekly_returns_df.columns,
            columns=weekly_returns_df.columns,
            dtype=float,
        )
        lag_matrix = pd.DataFrame(
            index=weekly_returns_df.columns,
            columns=weekly_returns_df.columns,
            dtype=int,
        )
        p_value_matrix = pd.DataFrame(
            index=weekly_returns_df.columns,
            columns=weekly_returns_df.columns,
            dtype=float,
        )

        np.fill_diagonal(correlation_matrix.values, 1.0)
        np.fill_diagonal(p_value_matrix.values, 0.0)

        for stock1 in weekly_returns_df.columns:
            for stock2 in weekly_returns_df.columns:
                if stock1 == stock2:
                    continue

                max_corr = -np.inf
                best_lag = 0
                best_p_value = np.nan

                for lag in range(-max_lag, max_lag + 1):
                    lag_index = i + lag
                    if 0 <= lag_index < len(dates):
                        series1 = week_data[stock1]
                        series2 = weekly_returns_df.iloc[lag_index][stock2]

                        if len(series1) >= 2 and len(series2) >= 2:
                            n = min(len(series1), len(series2))
                            corr, p_val = pearsonr(series1[:n], series2[:n])

                            if corr > max_corr:
                                max_corr = corr
                                best_lag = lag
                                best_p_value = p_val

                correlation_matrix.at[stock1, stock2] = max_corr
                lag_matrix.at[stock1, stock2] = best_lag
                p_value_matrix.at[stock1, stock2] = best_p_value

        yield current_date, correlation_matrix, lag_matrix, p_value_matrix


def save_correlation_matrices(output_dir, max_lag):
    for date, corr_df, lag_df, pval_df in generate_weekly_correlations_with_lags(
        weekly_returns_df, max_lag
    ):
        date_str = date.strftime("%Y-%m-%d")

        corr_df.to_csv(output_dir / f"max_correlation_matrix_{date_str}.csv")
        lag_df.to_csv(output_dir / f"optimal_lag_matrix_{date_str}.csv")
        pval_df.to_csv(output_dir / f"p_value_matrix_{date_str}.csv")

        print(f"Saved matrices for {date_str}")


# ----------------------------------------------------
# Run
# ----------------------------------------------------

max_lag = 4
save_correlation_matrices(output_dir, max_lag)

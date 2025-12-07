import pandas as pd

def process_data(input_file, output_file, top_200_output_file):
    # Read the combined data and keep only the necessary columns
    combined_data = pd.read_csv(input_file, usecols=['Date', 'Adj Close', 'Volume', 'Stock Ticker'],
                                parse_dates=['Date'])

    # Calculate average volume for each company
    avg_volume = combined_data.groupby('Stock Ticker')['Volume'].mean()

    # Select companies by average volume
    sorted_companies = avg_volume.sort_values(ascending=False).index

    # Filter data for the sorted companies
    filtered_data = combined_data[combined_data['Stock Ticker'].isin(sorted_companies)]

    # Pivot the DataFrame to get the final data for these companies
    returns_df = filtered_data.pivot_table(index='Date', columns='Stock Ticker', values='Adj Close')

    # Initialize an empty list to hold the selected companies
    selected_companies = []
    for company in sorted_companies:
        if len(selected_companies) >= 50:
            break
        company_data = returns_df[company]
        if company_data.isna().sum() == 0:  # No NaN values
            selected_companies.append(company)

    # Filter the returns DataFrame to include only the selected companies
    final_returns_df = returns_df[selected_companies]

    # Save the top 200 companies to a CSV file
    top_200_companies_df = avg_volume.loc[selected_companies].reset_index()
    top_200_companies_df.columns = ['Stock Ticker', 'Average Volume']
    top_200_companies_df.to_csv(top_200_output_file, index=False)
    print(f"Top 200 companies by average volume saved to {top_200_output_file}")

    # Resample daily returns into weekly groups (Thursday to Wednesday)
    def lump_daily_returns(group):
        return group.values.tolist()

    lumped_daily_returns = final_returns_df.resample('W-THU').apply(lump_daily_returns)
    lumped_daily_returns = lumped_daily_returns.dropna(how='all')  # Drop weeks with all NaN values

    # Save the lumped daily returns to a CSV file
    lumped_daily_returns.to_csv(output_file)
    print(f"Lumped daily returns saved to {output_file}")

if __name__ == "__main__":
    input_file = 'complete_filtered_data.csv'
    output_file = 'complete_lumped_daily_returns.csv'
    top_200_output_file = 'top_200_companies.csv'
    process_data(input_file, output_file, top_200_output_file)

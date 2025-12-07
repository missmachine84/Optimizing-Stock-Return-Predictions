import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

import os
from pathlib import Path

# Get project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output directories
input_directory_path = PROJECT_ROOT / "outputs" / "max_correlation_matrices"
output_directory_path = PROJECT_ROOT / "outputs"

output_file_path = output_directory_path / "combined_centrality_measures.csv"
graph_output_directory = output_directory_path / "graphs"
gephi_output_directory = output_directory_path / "gephi_files"

# Create output directories if they don't exist
os.makedirs(output_directory_path, exist_ok=True)
os.makedirs(graph_output_directory, exist_ok=True)
os.makedirs(gephi_output_directory, exist_ok=True)

# Get all files in the input directory
files = os.listdir(input_directory_path)

# Extract dates from filenames
dates = set()
for file in files:
    if file.endswith('.csv'):
        date = file.split('_')[-1].split('.')[0]
        dates.add(date)

# Initialize a list to collect dataframes
all_weeks_centrality_df_list = []

# Process each date
for date in sorted(dates):
    max_corr_file = os.path.join(input_directory_path, f'max_correlation_matrix_{date}.csv')
    p_values_file = os.path.join(input_directory_path, f'p_value_matrix_{date}.csv')
    optimal_lag_file = os.path.join(input_directory_path, f'optimal_lag_matrix_{date}.csv')

    # Ensure all necessary files exist for the date
    if not (os.path.exists(max_corr_file) and os.path.exists(p_values_file) and os.path.exists(optimal_lag_file)):
        print(f"Missing one or more required files for date {date}")
        continue

    # Load the data
    max_corr_df = pd.read_csv(max_corr_file, index_col=0)
    p_values_df = pd.read_csv(p_values_file, index_col=0)
    optimal_lag_df = pd.read_csv(optimal_lag_file, index_col=0)

    # Thresholds for correlation and p-values
    correlation_threshold = 0.8
    p_value_threshold = 0.05

    # Create two graphs: one directed, one undirected
    G_directed = nx.DiGraph()
    G_undirected = nx.Graph()

    # Adding edges based on correlation, p-value, and lag interpretation
    edge_labels = {}  # Dictionary to store edge labels
    for stock1 in max_corr_df.columns:
        for stock2 in max_corr_df.index:
            if stock1 != stock2:
                corr = max_corr_df.loc[stock1, stock2]
                p_value = p_values_df.loc[stock1, stock2]
                lag = optimal_lag_df.loc[stock1, stock2]

                if corr >= correlation_threshold and p_value < p_value_threshold:
                    label = f"{corr:.2f}, lag={lag}"
                    if lag > 0:  # stock2 is lagging, stock1 is leading
                        G_directed.add_edge(stock1, stock2, weight=corr)
                        edge_labels[(stock1, stock2)] = label
                    elif lag < 0:  # stock1 is lagging, stock2 is leading
                        G_directed.add_edge(stock2, stock1, weight=corr)
                        edge_labels[(stock2, stock1)] = label
                    elif lag == 0:  # No lag, add undirected edge
                        G_undirected.add_edge(stock1, stock2, weight=corr)
                        edge_labels[(stock1, stock2)] = label

    # Combine the directed and undirected graphs for visualization and analysis
    G = nx.compose(G_directed, G_undirected)

    # Calculate centrality measures
    centrality_measures = {
        'degree_centrality': nx.degree_centrality(G),
        'betweenness_centrality': nx.betweenness_centrality(G, weight='weight'),
        'closeness_centrality': nx.closeness_centrality(G, distance='weight'),
        'pagerank': nx.pagerank(G, weight='weight')
    }

    # Try calculating eigenvector centrality and handle errors
    try:
        centrality_measures['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(G, weight='weight')
    except Exception as e:
        print(f"Skipping eigenvector centrality for {date} due to error: {e}")

    # Combine all centrality measures into one DataFrame
    combined_centrality_df = pd.DataFrame(centrality_measures)

    # Add the 'Stock' column
    combined_centrality_df.reset_index(inplace=True)
    combined_centrality_df.rename(columns={'index': 'Stock'}, inplace=True)

    # Add the 'Week' column
    combined_centrality_df['Week'] = date

    # Collect the DataFrame into the list
    all_weeks_centrality_df_list.append(combined_centrality_df)

    # Save graph for NVDA for the first and last week
    if 'NVDA' in max_corr_df.columns and (date == sorted(dates)[0] or date == sorted(dates)[-1]):
        pos = nx.spring_layout(G, k=0.7)  # Increase the k value to spread out the nodes more
        plt.figure(figsize=(20, 15))  # Increase the figure size

        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')  # Reduce node size
        nx.draw_networkx_edges(G, pos, edgelist=G_directed.edges(), edge_color='gray', arrowstyle='-|>', arrowsize=20, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_edges(G, pos, edgelist=G_undirected.edges(), style='solid', edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='darkblue')  # Reduce font size
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)  # Further reduce edge label font size

        plt.title(f'Stock Relationships for {date} Based on Lead-Lag Analysis')
        plt.axis("off")  # Hide axis
        graph_filename = f'nvda_graph_{date}.png'
        plt.savefig(os.path.join(graph_output_directory, graph_filename))
        plt.close()

        # Save the graph to a file compatible with Gephi
        gephi_filename = f'nvda_graph_{date}.gexf'
        nx.write_gexf(G, os.path.join(gephi_output_directory, gephi_filename))

# Concatenate all DataFrames into one
final_combined_df = pd.concat(all_weeks_centrality_df_list, ignore_index=True)

# Convert 'Week' to datetime
final_combined_df['Week'] = pd.to_datetime(final_combined_df['Week'])

# Save the final combined DataFrame to a single CSV file
final_combined_df.to_csv(output_file_path, index=False)
print(f"All weeks combined centrality measures saved to {output_file_path}")

# Visualization of how centrality measures vary over time
centrality_output_directory = os.path.join(output_directory_path, 'centrality_measures')
os.makedirs(centrality_output_directory, exist_ok=True)

centrality_measures_over_time = final_combined_df.melt(id_vars=['Stock', 'Week'], var_name='Centrality Measure', value_name='Value')
centrality_stocks = centrality_measures_over_time['Stock'].unique()

# List of specific dates for x-axis ticks
specific_dates = ['2019-01-04', '2020-01-04', '2021-01-04', '2022-01-04', '2023-01-04']
specific_dates_datetime = pd.to_datetime(specific_dates)

for stock in centrality_stocks:
    stock_centrality = centrality_measures_over_time[centrality_measures_over_time['Stock'] == stock]
    plt.figure(figsize=(14, 7))
    for measure in stock_centrality['Centrality Measure'].unique():
        measure_data = stock_centrality[stock_centrality['Centrality Measure'] == measure]
        plt.plot(measure_data['Week'], measure_data['Value'], label=measure)

    plt.title(f'Centrality Measures Over Time for {stock}')
    plt.xlabel('Date')
    plt.ylabel('Centrality Measure Value')
    plt.legend()

    # Explicitly set x-axis tick labels
    plt.gca().set_xticks(specific_dates_datetime)
    plt.gca().set_xticklabels(specific_dates, rotation=45)
    plt.tight_layout()

    centrality_filename = f'centrality_measures_{stock}.png'
    plt.savefig(os.path.join(centrality_output_directory, centrality_filename))
    plt.close()

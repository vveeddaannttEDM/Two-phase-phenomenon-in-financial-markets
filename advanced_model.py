#I'll implement Moving Average & Bollinger Bands and Monte Carlo Simulations first. Let me add them to the model! 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, kurtosis, skew
import argparse


def load_data(file_path=None):
    """
    Load financial time series data from a CSV file or generate synthetic data.
    """
    if file_path:
        data = pd.read_csv(file_path, parse_dates=['Timestamp'])
        data = data.sort_values('Timestamp')
    else:
        # Generate synthetic financial time series data
        np.random.seed(42)
        timestamps = pd.date_range(start='2020-01-01', periods=1000, freq='T')
        index_values = np.cumsum(np.random.randn(1000) * 10 + 10000)
        data = pd.DataFrame({'Timestamp': timestamps, 'Index Value': index_values})
    
    return data


def calculate_absolute_increment(data):
    """
    Calculate absolute increments of the financial index.
    """
    data['Absolute Increment'] = data['Index Value'].diff().abs()
    data = data.dropna()
    return data


def fit_power_law(data):
    """
    Fit a power-law distribution to the absolute increments and return the exponent.
    """
    increments = data['Absolute Increment'].values
    increments = increments[increments > 0]  # Remove zeros for log transformation

    # Log-log transformation
    log_increments = np.log(increments)
    log_ranks = np.log(np.arange(1, len(increments) + 1))

    # Linear regression on log-log data
    slope, intercept, r_value, p_value, std_err = linregress(log_increments, log_ranks)
    alpha = -slope

    print(f"Estimated Power-Law Exponent (ζ_I): {alpha}")
    return alpha, log_increments, log_ranks


def plot_distribution(data, log_increments, log_ranks):
    """
    Plot the empirical and fitted power-law distribution.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(np.exp(log_increments), np.exp(log_ranks), color='b', label='Empirical Data')
    plt.plot(np.exp(log_increments), np.exp(log_increments) ** -alpha, 'r--', label=f'Power-law Fit (α = {alpha:.2f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Absolute Increment')
    plt.ylabel('Rank')
    plt.title('Power-law Fit to Absolute Increment')
    plt.legend()
    plt.grid(True)
    plt.show()


def detect_bifurcation(data):
    """
    Detect bifurcation by analyzing the conditional probability distribution of returns.
    """
    data['Return'] = data['Index Value'].diff()
    data['Fluctuation'] = data['Return'].rolling(window=5).std()
    
    # Group by fluctuation levels
    bins = pd.qcut(data['Fluctuation'].dropna(), q=4, labels=False)
    grouped = data.groupby(bins)

    plt.figure(figsize=(10, 6))
    for name, group in grouped:
        plt.hist(group['Return'], bins=50, alpha=0.5, label=f'Fluctuation Level {name}')
    
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.title('Conditional Probability Distribution of Returns')
    plt.legend()
    plt.grid(True)
    plt.show()


def statistical_analysis(data):
    """
    Perform statistical analysis on the returns including skewness, kurtosis, and volatility.
    """
    data['Return'] = data['Index Value'].diff().dropna()
    returns = data['Return'].dropna()

    mean_return = returns.mean()
    volatility = returns.std()
    skewness_value = skew(returns)
    kurtosis_value = kurtosis(returns)

    print(f"Mean Return: {mean_return:.4f}")
    print(f"Volatility (Standard Deviation): {volatility:.4f}")
    print(f"Skewness: {skewness_value:.4f}")
    print(f"Kurtosis: {kurtosis_value:.4f}")

    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.7, color='c', edgecolor='k')
    plt.axvline(mean_return, color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.title('Histogram of Returns with Statistical Indicators')
    plt.legend()
    plt.grid(True)
    plt.show()


def moving_average_bollinger(data):
    """
    Plot Moving Average and Bollinger Bands.
    """
    data['Moving Average'] = data['Index Value'].rolling(window=20).mean()
    data['Bollinger High'] = data['Moving Average'] + 2 * data['Index Value'].rolling(window=20).std()
    data['Bollinger Low'] = data['Moving Average'] - 2 * data['Index Value'].rolling(window=20).std()

    plt.figure(figsize=(12, 6))
    plt.plot(data['Timestamp'], data['Index Value'], label='Index Value', color='blue')
    plt.plot(data['Timestamp'], data['Moving Average'], label='20-Period Moving Average', color='orange')
    plt.fill_between(data['Timestamp'], data['Bollinger High'], data['Bollinger Low'], color='gray', alpha=0.3, label='Bollinger Bands')
    plt.xlabel('Timestamp')
    plt.ylabel('Index Value')
    plt.title('Moving Average and Bollinger Bands')
    plt.legend()
    plt.grid(True)
    plt.show()


def monte_carlo_simulation(data, num_simulations=1000, num_days=252):
    """
    Monte Carlo simulation for future price forecasting using Geometric Brownian Motion.
    """
    returns = data['Index Value'].pct_change().dropna()
    last_price = data['Index Value'].iloc[-1]
    
    mean_return = returns.mean()
    volatility = returns.std()

    simulation_results = np.zeros((num_days, num_simulations))

    for i in range(num_simulations):
        price_series = [last_price]
        for _ in range(num_days):
            price_change = price_series[-1] * (mean_return + volatility * np.random.normal())
            price_series.append(price_series[-1] + price_change)
        simulation_results[:, i] = price_series[1:]

    plt.figure(figsize=(12, 6))
    plt.plot(simulation_results, color='gray', alpha=0.1)
    plt.title('Monte Carlo Simulation of Future Prices')
    plt.xlabel('Days')
    plt.ylabel('Simulated Price')
    plt.grid(True)
    plt.show()


def main(args):
    data = load_data(args.data_path)
    data = calculate_absolute_increment(data)
    
    global alpha  # To use in plot function
    alpha, log_increments, log_ranks = fit_power_law(data)
    plot_distribution(data, log_increments, log_ranks)
    detect_bifurcation(data)
    statistical_analysis(data)
    moving_average_bollinger(data)
    monte_carlo_simulation(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Two-Phase Phenomena in Financial Markets')
    parser.add_argument('--data_path', type=str, default=None, help='Path to the CSV file containing time series data')
    args = parser.parse_args()
    
    main(args)

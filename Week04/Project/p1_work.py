import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_prices(returns, initial_price, method="classical_brownian", print_calc = True):
    #initial price
    prices = [initial_price]

    for i in range(len(returns)):
        r_t = returns.iloc[i]

        if method == "classical_brownian":
            # Classical Brownian Motion: P_t = P_{t-1} + r_t
            p_t = prices[i] + r_t
        elif method == "arithmetic_return":
            # Arithmetic Return System: P_t = P_{t-1}(r_t + 1)
            p_t = prices[i] * (1 + r_t)
        elif method == "geometric_brownian":
            # Log Return or Geometric Brownian Motion: P_t = P_{t-1}*e^{r_t}
            p_t = prices[i] * np.exp(r_t)
        else:
            raise ValueError("Invalid method. Supported methods are 'classical_brownian', 'arithmetic_return', and 'geometric_brownian'.")

        prices.append(p_t)

    expected_value = np.mean(prices)
    std_deviation = np.std(prices)
    if print_calc == True:
        print(f"Expected value of {method}: {expected_value}")
        print(f"Standard Deviation of {method}: {std_deviation}\n")

    return prices, expected_value, std_deviation

# Calculate and compare the expected value and standard deviation of price at time t(P_t)
# Assume r_t ~ N(0, σ^2)
prices = pd.read_csv("Week04/DailyPrices.csv").drop("Date", axis=1)
returns = pd.read_csv("Week04/DailyReturn.csv").drop("Date", axis=1)


best_col = None
best_avg = 1
for col in returns.columns:
    avg = returns[col].mean()
    if abs(avg) < best_avg:
        best_col = col
        best_avg = abs(avg)
print(f"Stock with mean closest to 0 is {best_col} with mean of {best_avg:.6f}\n")

print(f"True Expected Value of {best_col}: {np.mean(prices[best_col])}")
print(f"True Standard Deviation of {best_col}: {np.std(prices[best_col])}\n")

# Performing calculations only on the best col for simplicity of analysis
initial_price = prices[best_col].iloc[0]
test_returns = returns[best_col]

# Calculate with classical brownian formula
classical_prices, classical_mean, classical_std = calculate_prices(test_returns, initial_price, method="classical_brownian")

# Calculate with arithmetic return formula
arithmetic_prices, arithmetic_mean, arithmetic_std = calculate_prices(test_returns, initial_price, method="arithmetic_return")

# Calculate with geometric brownian formula
geometric_prices, geometric_mean, geometric_std = calculate_prices(test_returns, initial_price, method="geometric_brownian")

def simulate_price_return_formulas(std_dev_returns, initial_price, method_form, num_sims = 1000):

    num_samples = len(returns[best_col])
    # Initialize arrays to contain simulation results
    sim_means = np.zeros(num_sims)
    sim_stds = np.zeros(num_sims)

    for i in range(num_sims):
        # Simulate a random returns based on the std dev of returns for this column
        sim_returns = np.random.normal(0, std_dev_returns, num_samples)
        sim_returns = pd.Series(sim_returns)
        sim_prices, sim_mean, sim_std = calculate_prices(sim_returns, initial_price, method=method_form, print_calc = False)
        sim_means[i] = sim_mean
        sim_stds[i] = sim_std


    # Show the mean and std dev match your expectations
    print(f"Simulated Expected Value of {method_form}: {np.mean(sim_means)}")
    print(f"Simulated Standard Deviation of {method_form}: {np.mean(sim_stds)}\n")
    return np.mean(sim_means), np.mean(sim_stds)

# Simulate each return equation using r_t ~ N(0, σ^2)
std_dev_returns = returns[best_col].std()

# Simulate using classical brownian
sim_classical_mean, sim_classical_std = simulate_price_return_formulas(std_dev_returns, initial_price, method_form="classical_brownian")

# Simulate using Arthimetic Returns
sim_arithmetic_mean, sim_arithmetic_std = simulate_price_return_formulas(std_dev_returns, initial_price, method_form="arithmetic_return")

# Simulate using Geometric Brownian
sim_geometric_mean, sim_geometric_std = simulate_price_return_formulas(std_dev_returns, initial_price, method_form="geometric_brownian")


# Define method names and their corresponding means and standard deviations
methods = ["Actual", "Classical Brownian", "Arithmetic Returns", "Geometric Brownian"]
means = [np.mean(prices[best_col]), classical_mean, arithmetic_mean, geometric_mean]
stds = [np.std(prices[best_col]), classical_std, arithmetic_std, geometric_std]

# Simulated means and standard deviations
simulated_means = [means[0], sim_classical_mean, sim_arithmetic_mean, sim_geometric_mean]
simulated_stds = [stds[0], sim_classical_std, sim_arithmetic_std, sim_geometric_std]

# Define the width of the bars
bar_width = 0.35

# Create an array for the x-coordinates of the bars
x = np.arange(len(methods))

# Define distinct colors for actual and simulated bars
actual_color = 'skyblue'
simulated_color = 'salmon'

# Create subplots for means and standard deviations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot means
ax1.bar(x - bar_width/2, means, label='Actual', alpha=0.7, color=actual_color, width=bar_width)
ax1.bar(x + bar_width/2, simulated_means, label='Simulated', alpha=0.7, color=simulated_color, width=bar_width)
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_title('Means')
ax1.legend(loc = 'lower right')

# Plot standard deviations
ax2.bar(x - bar_width/2, stds, label='Actual', alpha=0.7, color=actual_color, width=bar_width)
ax2.bar(x + bar_width/2, simulated_stds, label='Simulated', alpha=0.7, color=simulated_color, width=bar_width)
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.set_title('Standard Deviations')
ax2.legend(loc = 'lower right')

plt.tight_layout()
plt.show()
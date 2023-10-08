import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import pandas as pd
from scipy.stats import t, norm
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

# Use data in problem1.csv
data = pd.read_csv("Week05/Project/problem1.csv")

# Fit Normal Distribution to data
norm_mean, norm_std = fin.mle_normal_distribution_one_input(data)

print(f"Normal Optimized Mean: {norm_mean}")
print(f"Normal Optimized Std: {norm_std}")
# Fit Generalized T Distribution to this data
t_mean, t_std, t_df = fin.mle_t_distribution_one_input(data.values)
#t_mean, t_std, t_df = t.fit(data.values)

print(f"T Optimized Mean: {t_mean}")
print(f"T Optimized Std: {t_std}")
print(f"T Optimized DF: {t_df}")

# Calculate VaR for Normal
var_normal = -1*fin.calc_var_normal(norm_mean, norm_std)
print(f"VaR Normal Dist: {var_normal}")

# Calculate VaR for Generalized T
var_t = -1*fin.calc_var_t_dist(t_mean, t_std, t_df)
print(f"VaR T Dist: {var_t}")

# Calculate ES for Normal

es_norm = -1*fin.calc_expected_shortfall_normal(norm_mean, norm_std, alpha=0.05)
print(f"Expected Shortfall for Norm Dist: {es_norm}")

# Calculate ES for Generalized T

es_t = fin.calc_expected_shortfall_t(t_mean, t_std, t_df)
print(f"Expected Shortfall for T Dist: {es_t}")
# Overlay graphs of PDFs, VaR, ES

# Generate a range of x values for plotting
x = np.linspace(-0.5, 0.5, 1000)

# PDF for Normal Distribution
pdf_normal = norm.pdf(x, loc=norm_mean, scale=norm_std)

# PDF for Generalized T Distribution
pdf_t = t.pdf(x, t_df, loc=t_mean, scale=t_std)

# Plot the PDFs
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_normal, label='Normal PDF', color='blue')
plt.plot(x, pdf_t, label='Generalized T PDF', color='orange')

# VaR lines
plt.axvline(x=var_normal, color='blue', linestyle='--', label='VaR Normal')
plt.axvline(x=var_t, color='orange', linestyle='--', label='VaR Generalized T')

# ES lines
plt.axvline(x=es_norm, color='blue', linestyle='-.', label='ES Normal')
plt.axvline(x=es_t, color='orange', linestyle='-.', label='ES Generalized T')

plt.xlabel('X')
plt.ylabel('PDF')
plt.legend()
plt.title('PDFs, VaR, and ES')
plt.grid(True)
plt.show()
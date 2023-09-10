import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t

# Import data and separate into x and y components
data = pd.read_csv('Week02/Project/problem2.csv')
X = data['x']
y = data['y']

print("Performing OLS")
print("------------------------------------------------------------")
fin_package.perform_ols(X, y, True)

print("Performing MLE for Normal Distribution")
print("------------------------------------------------------------")
fin_package.optimize_normal_distribution(X, y, True)

print("Performing MLE for t Distribution")
print("------------------------------------------------------------")
fin_package.optimize_t_distribution(X, y, True)
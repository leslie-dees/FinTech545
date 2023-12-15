import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin

test_data = pd.read_csv("testfiles/data/test7_3.csv")
testout_data = pd.read_csv("testfiles/data/testout7_3.csv")

# get data
df_y = test_data[['y']]
df_x = test_data.drop(columns=['y'])



mu, sigma, nu, alpha, *beta = fin.fit_regression_t(df_x, df_y)
print(mu)
print(sigma)
print(nu)
print(alpha)
print(beta)

print(testout_data)
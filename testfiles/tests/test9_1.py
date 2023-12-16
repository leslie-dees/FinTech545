import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin

test_portfolio = pd.read_csv("testfiles/data/test9_1_portfolio.csv")
test_returns = pd.read_csv("testfiles/data/test9_1_returns.csv")
testout_data = pd.read_csv("testfiles/data/testout9_1.csv")

print(test_portfolio)
print(test_returns)

print(testout_data)
import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np

test_data = pd.read_csv("testfiles/data/test6.csv")
#testout_data = pd.read_csv("testfiles/data/testout_6.1.csv")

# Arithmetic returns
arithmetic_returns = fin.return_calculate(test_data, method="DISCRETE", date_column="Date")
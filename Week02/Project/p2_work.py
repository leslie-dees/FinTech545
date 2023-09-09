import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package
import pandas as pd

# Import data and separate into x and y components
data = pd.read_csv('Week02/Project/problem2.csv')
X = data['x']
Y = data['y']
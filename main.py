
from LinReg import LinearRegression as lr
import pandas as pd


datafile = input("Input file path for dataset (MoreHouseData.csv): ")
data = pd.read_csv(datafile)
m = len(data)

model = lr()

print(list(data))
x_col = list((input('Enter X features (type 3/4, delimit by space -- Beds, Baths, Sqft, Price): ')).split())
y_col = input('Enter Y Column name (the one not selected above): ')


model.matrix_maker(x_col, y_col, data, m)
weights = model.fit()

cost = model.cost_function(m)
print(f"Cost Function: {cost[0][0]}")
print(f"Weights (w1): \n {weights}")


x_pred = list(map(float, input("Enter X values separated by spaces: ").split()))
x_pred = [1] + x_pred #padding
y_pred = model.predict(x_pred)
print(f"\n\nPrediction for {x_pred} {x_col}:  {y_pred}  {y_col}")






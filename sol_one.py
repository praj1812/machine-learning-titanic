import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('train head:\n', train.head())
print('\ntrain description:\n', train.describe())

with open('sol_one.csv', 'w') as f:
    with open('test.csv', 'r') as g:
        f.write(g.read())
        f.close()
        g.close()

sol_one = pd.read_csv('sol_one.csv')

sol_one["Survived"] = 0
sol_one["Survived"][sol_one["Sex"] == "female"] = 1
print(sol_one["Survived"])

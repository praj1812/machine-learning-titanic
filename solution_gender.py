import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('train head:\n', train.head())
print('\ntrain description:\n', train.describe())

test_one = pd.read_csv('test.csv')

test_one['Survived'] = 0
test_one['Survived'][test_one['Sex'] == 'female'] = 1
print(test_one['Survived'])

cols = ['PassengerId', 'Survived']
solution_gender = test_one[cols]
solution_gender.to_csv("solution_gender.csv", index=False)

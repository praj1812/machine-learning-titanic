# solution_1498888965.63413.csv score = 0.70813
# 0.75598 from kaggle
# solution_1498889109.5971751.csv score = 0.70335 - without embarked

import pandas as pd
import numpy as np
from sklearn import tree
import time


def impute(data):
    data['Sex'][data['Sex'] == 'male'] = 0
    data['Sex'][data['Sex'] == 'female'] = 1

    data['Embarked'][data['Embarked'] == 'S'] = 0
    data['Embarked'][data['Embarked'] == 'C'] = 1
    data['Embarked'][data['Embarked'] == 'Q'] = 2
    # median = 0
    data.Embarked = data.Embarked.fillna(data.Embarked.median())

    # median = 28.0
    # mean = 29.36158249158249
    data.Age = data.Age.fillna(data.Age.mean())

    data.Fare = data.Fare.fillna(data.Fare.mean())


train = pd.read_csv('train.csv')
impute(train)

train_target = train['Survived'].values

train['family size'] = train['SibSp'] + train['Parch'] + 1
train_features = train[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'family size']].values

my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(train_features, train_target)

# Predict test data:
test = pd.read_csv('test.csv')
impute(test)

test['family size'] = test['SibSp'] + test['Parch'] + 1
test_features = test[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'family size']].values

prediction = my_tree.predict(test_features)

PassengerId = np.array(test['PassengerId']).astype(int)
solution = pd.DataFrame(prediction, PassengerId, columns=['Survived'])
print(solution)

file_name = 'solution_' + str(time.time()) + '.csv'

solution.to_csv(file_name, index_label=['PassengerId'])

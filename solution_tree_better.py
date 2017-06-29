# solution_tree_better_1498739483.6288395.csv score = 0.76077 -- using median
# solution_tree_better_1498739483.6288395.csv score = 0.76077 -- using mean

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
train_features = train[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']].values

my_tree = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5)
my_tree = my_tree.fit(train_features, train_target)

# Predict test data:
test = pd.read_csv('test.csv')
impute(test)

test_features = test[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']].values

prediction = my_tree.predict(test_features)

PassengerId = np.array(test['PassengerId']).astype(int)
solution = pd.DataFrame(prediction, PassengerId, columns=['Survived'])
print(solution)

file_name = 'solution_tree_better_' + str(time.time()) + '.csv'

solution.to_csv(file_name, index_label=['PassengerId'])

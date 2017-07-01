# solution_forest_1498890651.2149243.csv score = 0.74641
#   with features 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'family size'
# solution_forest_1498890759.1753356.csv score = 0.75598
#   with features 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'family size', 'Embarked'
# solution_forest_1498890880.1388474.csv score = 0.75598
#   with features 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked'


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
train_features = train[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'family size', 'Embarked']].values

my_forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
my_forest = my_forest.fit(train_features, train_target)

# Predict test data:
test = pd.read_csv('test.csv')
impute(test)

test['family size'] = test['SibSp'] + test['Parch'] + 1
test_features = test[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'family size', 'Embarked']].values

prediction = my_forest.predict(test_features)

PassengerId = np.array(test['PassengerId']).astype(int)
solution = pd.DataFrame(prediction, PassengerId, columns=['Survived'])
print(solution)

file_name = 'solution_forest_' + str(time.time()) + '.csv'

solution.to_csv(file_name, index_label=['PassengerId'])

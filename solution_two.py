# solution tree score = 0.72727 with features Pclass, Sex, Age, Fare ???
# solution_tree_four score = 0.67943 with features Pclass, Sex, Age, Fare, Embarked, SibSp and Parch
# solution_tree_five score = 0.72249 with features Pclass, Sex, Age, Fare
# solution_tree_six score = 0.69378 with features Sex, Age, Fare
# solution_tree_1498635726.175787.csv score = 0.72249 with features Pclass, Sex, Age, Fare

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
    # mean = 29.36158249158249 : TODO - try using mean
    data.Age = data.Age.fillna(data.Age.median())

    data.Fare = data.Fare.fillna(data.Fare.median())


train = pd.read_csv('train.csv')
impute(train)

train_target = train['Survived'].values
train_features = train[['Pclass', 'Sex', 'Age', 'Fare']].values

my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(train_features, train_target)

# Predict test data:
test = pd.read_csv('test.csv')
impute(test)

test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values

prediction = my_tree.predict(test_features)

PassengerId = np.array(test['PassengerId']).astype(int)
solution = pd.DataFrame(prediction, PassengerId, columns=['Survived'])
print(solution)

file_name = 'solution_tree_' + str(time.time()) + '.csv'

solution.to_csv(file_name, index_label=['PassengerId'])














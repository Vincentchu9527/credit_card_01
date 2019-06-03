# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

plt.rcParams['font.sans-serif'] = ['SimHei']

# 数据探索
data = pd.read_csv('./UCI_Credit_Card.csv')
print(data.info())
# print(data.describe())
label_counts = data['default.payment.next.month'].value_counts()
print(label_counts)
label_counts.plot(kind='bar')
plt.title('信用卡违约率客户\n (违约：1，守约：0)')
plt.show()

# 训练集测试集分离
labels = data['default.payment.next.month'].values
columns_ = data.columns[1: -2]
features = data[columns_].values
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=.3, stratify=labels, random_state=33)

# 构造各种分类器
classifiers = [
    SVC(random_state=1, kernel='rbf'),
    DecisionTreeClassifier(random_state=1, criterion='gini'),
    RandomForestClassifier(random_state=1, criterion='gini'),
    KNeighborsClassifier(metric='minkowski'),
    AdaBoostClassifier(base_estimator=None),
]
# 分类器名称
classifier_names = [
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'kneighborsclassifier',
    'adaboost',
]
# 分类器参数
classifier_param_grid = [
    {'svc__C': [1], 'svc__gamma': [0.01]},
    {'decisiontreeclassifier__max_depth': [6, 9, 11]},
    {'randomforestclassifier__n_estimators': [3, 5, 6]},
    {'kneighborsclassifier__n_neighbors': [4, 6, 8]},
    {'adaboost__n_estimators': [10, 50, 100]},
]


# 使用 GridSearchCV 进行参数调优
def GridSearchCV_work(model_name, pipeline, train_x, train_y, test_x, test_y, param_grid, score='accuracy'):
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score, cv=3)
    # 寻找最优的参数 和最优的准确率分数
    search = gridsearch.fit(train_x, train_y)
    print(model_name + ':')
    print("GridSearch最优参数：", search.best_params_)
    print("GridSearch最优分数： %0.4lf" % search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print("测试集准确率 %0.4lf" % accuracy_score(test_y, predict_y))


# 通过 for 循环调用函数
for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (model_name, model)
    ])
    GridSearchCV_work(model_name, pipeline, train_x, train_y, test_x, test_y, model_param_grid, score='accuracy')

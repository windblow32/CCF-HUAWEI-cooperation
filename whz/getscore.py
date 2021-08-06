import pandas as pd
from sklearn.datasets import load_iris

import random
from knn_class import KNN
from decision_tree_class import decision_tree
from rf_class import rf
from muti_class import muti_reg
from elastic_class import elastic_reg
from rbf_class import rbf
from GradientBoostingRegressor_class import gbr
from extraforest_class import etr
from linear_class import linear

# from sklearn.datasets import make_regression
# # X为样本特征，y为样本输出， coef 为回归系数，共1000个样本，每个样本1个特征
# X, y = make_regression(n_samples=500, n_features=2, coef= False)

data = pd.read_csv('./exponential_s.csv', header=None, engine='python')
X = []
y = []
global TOTAL_NUMBER
TOTAL_NUMBER = data.shape[0]
print(TOTAL_NUMBER)
for i in range(data.shape[0]):
    X.append(data.loc[i, 0])
    y.append(data.loc[i, 1])

# # 导入IRIS数据集
# iris = load_iris()
# # 特征矩阵
# X = iris.data
# # 目标向量
# y = iris.target
i = 0

# 文件输出
# f = open('log.txt','w')
# for i in range(100):
#     print(str(i), file=f)
# f.close()

for seed in range(10,50,2):
    print("***************")
    print("epoch " + str(i))
    i += 1
    score1 = KNN(X, y, seed).getscore()
    print(score1)

    score2 = decision_tree(X, y, seed).getscore()
    print(score2)

    score3 = rf(X, y, seed).getscore()
    print(score3)

    score4 = muti_reg(X, y, seed).getscore()
    print(score4)

    score5 = elastic_reg(X, y, seed).getscore()
    print(score5)

    score6 = rbf(X, y, seed).getscore()
    print(score6)

    score7 = gbr(X, y, seed).getscore()
    print(score7)

    score8 = etr(X, y, seed).getscore()
    print(score8)

    score9 = linear(X, y, seed).getscore()
    print(score9)

    score = [score1, score2, score3, score4, score5, score6, score7, score8, score9]
    name = ["KNN Regression", "Decision Tree Regression", "Random Forest Regression", "Multi Regression", "Elastic Regression", "Radial Basis Function Regression", "Gradient Boosting Regressor", "ExtraTree Regression", "Linear Regression" ]

    # #无decision tree 和 extratree
    # score = [score3, score5, score6, score9]
    # name = ["Random Forest Regression",
    #         "Elastic Regression", "Radial Basis Function Regression",
    #         "Linear Regression"]



    mininum = min(score)
    min_index = score.index(mininum)


    # 数组下标加一是模型序号
    print(name[min_index])


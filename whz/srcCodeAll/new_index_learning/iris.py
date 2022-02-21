from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics

from new_index_learning.produce_data import TestData
from train import ModelMark

# 交叉验证
from sklearn.model_selection import StratifiedKFold
# 网格搜索
from sklearn.model_selection import GridSearchCV

# 指标
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from numpy import random
import csv

# # 导入IRIS数据集
# iris = load_iris()
# # 特征矩阵
# X = iris.data
# # 目标向量
# y = iris.target

# 产生更真实的数据
data = TestData(1000, 1)
data.create(data)
csv_data = data.read(data, 0)
x_df = csv_data.iloc[:, 0]
y_df = csv_data.iloc[:, 1]

X = np.array(x_df)
X = X.reshape(100000, 1)
y = np.array(y_df)

# fixme readline
readline = 10000
s = np.shape(X)[0]
print("原有X大小 : ", s)
# size是修正后X的大小
size = int(s / readline) * readline
print("现有X大小 : ", size)

X = X[:size]
y = y[:size]
print("原X为 : \n", X)
# X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20)
test_size = 0.8
# fixme:test_size*size为整数，readline的倍数！
# 分割点
cutDot = int(test_size * size)
X1_train = X[:cutDot]
X1_test = X[cutDot:]
y1_train = y[:cutDot]
y1_test = y[cutDot:]

mk = ModelMark(X, y)
mk.train(readline, size)
z = mk.getLabel()
z = z[:size]
# X_train = X[:mk.getSize()*0.8]
# X_test = X[mk.getSize()*0.8:]
# z_train = z[:mk.getSize()*0.8]
# z_test = z[mk.getSize()*0.8:]
# 现在X1_train和z直接相连

# X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.20)
X_train = X[:cutDot]
X_test = X[cutDot:]
z_train = z[:cutDot]
z_test = z[cutDot:]
# 按道理，X_train和X1_train,以及X_test和X1_test应该相同

# print("X_train : \n", X_train)
# print("X1_train : \n", X1_train)
# 保证对应关系
assert X_train.all() == X1_train.all()
assert X_test.all() == X1_test.all()
# 调参
# n_estimators=100, *, criterion='mse',
# max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
# max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
# verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None

regressor = RandomForestClassifier(n_estimators=150, random_state=1, max_depth=6, min_samples_split=3,
                                   min_impurity_decrease=0.1, warm_start=True)
# 交叉验证
cross_score = cross_val_score(regressor, X, z, cv=10, scoring="neg_mean_squared_error")

# 查看所有模型评估(打分)的列表

print("所有模型评估(打分)的列表:\n", sorted(sklearn.metrics.SCORERS.keys()))
scoring = 'neg_mean_squared_error'
regressor.fit(X, z)
# 转化成int，带入到model得到输出，和开始的y比较
# fixme: 普通的int()只是取了整数部分，我们需要四舍五入！
z_pred = regressor.predict(X_test)

# 看分类器效果
print("predict : \n", z_pred)
print("true : \n", z_test)
# 计算RMSE(均方根误差，标准误差)
print("RMSE:", np.sqrt(metrics.mean_squared_error(z_test, z_pred)))
# 接近1拟合优度高
print("R2_score", r2_score(z_test, z_pred))

# median_absolute_error
print("Median absolute error : ", median_absolute_error(z_test, z_pred))

# mean_absolute_error
print("mean_absolute_error : ", mean_absolute_error(z_test, z_pred))

# mean_squared_log_error
print("mean_squared_log_error : ", mean_squared_log_error(z_test, z_pred))

# explained_variance_score
print("explained_variance_score : ", explained_variance_score(z_test, z_pred))

name = ["KNN Regression", "Decision Tree Regression", "Random Forest Regression", "Multi Regression",
        "Elastic Regression", "Radial Basis Function Regression", "Gradient Boosting Regressor",
        "ExtraTree Regression", "Linear Regression"]
from knn_class import KNN
from decision_tree_class import decision_tree
from rf_class import rf
from muti_class import muti_reg
from elastic_class import elastic_reg
from rbf_class import rbf
from GradientBoostingRegressor_class import gbr
from extraforest_class import etr
from linear_class import linear

error = 0.5


def correct(array):
    pos = 0
    for i in array:
        if i < 0:
            array[pos] = 0
        if (i - int(i)) > error:
            array[pos] = int(i) + 1
        else:
            array[pos] = int(i)
        pos += 1


# 最大可忍受
adaptable = 0.4
# 如果RMSE输出较大，说明拟合效果不好，选择的模型，有问题，应该缩小train部分，选择模型时候的error阈值
for model_i in z_pred:
    print("******************************************************")
    model_i = int(round(model_i))
    print("所选模型为 : ", name[model_i])
    print("真实下标为 : ", y1_test)
    # fixme : 最后是否为空
    predict = []
    RMSEValue = -1
    if model_i == 0:
        # 查看是否和y对应位置匹配,X_test对应训练结果是X的位置，对应了y_test
        predict = KNN.test(X1_test)
        print("预测下标为 : \n", predict)
        # -fixme : get y_pred
        correct(predict)
        print("修正整数后为 : \n", predict)
        RMSEValue = np.sqrt(metrics.mean_squared_error(predict, y1_test))
        print("RMSE : ", RMSEValue)
    if model_i == 1:
        # 查看是否和y对应位置匹配,X_test对应训练结果是X的位置，对应了y_test
        predict = decision_tree.test(X1_test)
        print("预测下标为 : \n", predict)
        # -fixme : get y_pred
        correct(predict)
        print("修正整数后为 : \n", predict)
        RMSEValue = np.sqrt(metrics.mean_squared_error(predict, y1_test))
        print("RMSE : ", RMSEValue)
    if model_i == 2:
        # 查看是否和y对应位置匹配,X_test对应训练结果是X的位置，对应了y_test
        predict = rf.test(X1_test)
        print("预测下标为 : \n", predict)
        # -fixme : get y_pred
        correct(predict)
        print("修正整数后为 : \n", predict)
        RMSEValue = np.sqrt(metrics.mean_squared_error(predict, y1_test))
        print("RMSE : ", RMSEValue)
    if model_i == 3:
        # 查看是否和y对应位置匹配,X_test对应训练结果是X的位置，对应了y_test
        predict = muti_reg.test(X1_test)
        print("预测下标为 : \n", predict)
        # -fixme : get y_pred
        correct(predict)
        print("修正整数后为 : \n", predict)
        RMSEValue = np.sqrt(metrics.mean_squared_error(predict, y1_test))
        print("RMSE : ", RMSEValue)
    if model_i == 4:
        # 查看是否和y对应位置匹配,X_test对应训练结果是X的位置，对应了y_test
        predict = elastic_reg.test(X1_test)
        print("预测下标为 : \n", predict)
        # -fixme : get y_pred
        correct(predict)
        print("修正整数后为 : \n", predict)
        RMSEValue = np.sqrt(metrics.mean_squared_error(predict, y1_test))
        print("RMSE : ", RMSEValue)
    if model_i == 5:
        # 查看是否和y对应位置匹配,X_test对应训练结果是X的位置，对应了y_test
        predict = rbf.test(X1_test)
        print("预测下标为 : \n", predict)
        # -fixme : get y_pred
        correct(predict)
        print("修正整数后为 : \n", predict)
        RMSEValue = np.sqrt(metrics.mean_squared_error(predict, y1_test))
        print("RMSE : ", RMSEValue)
    if model_i == 6:
        # 查看是否和y对应位置匹配,X_test对应训练结果是X的位置，对应了y_test
        predict = gbr.test(X1_test)
        print("预测下标为 : \n", predict)
        # -fixme : get y_pred
        correct(predict)
        print("修正整数后为 : \n", predict)
        RMSEValue = np.sqrt(metrics.mean_squared_error(predict, y1_test))
        print("RMSE : ", RMSEValue)
    if model_i == 7:
        # 查看是否和y对应位置匹配,X_test对应训练结果是X的位置，对应了y_test
        predict = etr.test(X1_test)
        print("预测下标为 : \n", predict)
        # -fixme : get y_pred
        correct(predict)
        print("修正整数后为 : \n", predict)
        RMSEValue = np.sqrt(metrics.mean_squared_error(predict, y1_test))
        print("RMSE : ", RMSEValue)
    if model_i == 8:
        # 查看是否和y对应位置匹配,X_test对应训练结果是X的位置，对应了y_test
        predict = linear.test(X1_test)
        print("预测下标为 : \n", predict)
        # -fixme : get y_pred
        correct(predict)
        print("修正整数后为 : \n", predict)
        RMSEValue = np.sqrt(metrics.mean_squared_error(predict, y1_test))
        print("RMSE : ", RMSEValue)

    if RMSEValue > 0.0:
        print("****************************************")
        print("need search to repair!")
        # fixme: searchArea : 出错后查找的范围
        searchArea = 3
        maxBound = int(np.shape(predict)[0])
        # 说明下标找错了
        for p in range(0, np.shape(predict)[0]):
            # fixme : X支持多维度输入！
            # 原版对于多维的y，后面有对应的[0:1]
            if predict[p:p + 1] != y1_test[p:p + 1]:
                # 说明预测出错,给出上下界
                print("error position located in : ", p)
                lowBound = max(p - searchArea, 0)
                upBound = min(p + searchArea, maxBound)
                for b in range(lowBound, upBound):
                    # 由于此处X数据多维，现在取0维假设是查询
                    if int(predict[b:b + 1]) == y1_test[p:p + 1]:
                        # 在此范围内找到了正确下标
                        continue
                    else:
                        while not (
                                predict[max(p - searchArea, 0):max(p - searchArea, 0) + 1] == y1_test[p:p + 1]
                                or predict[min(p + searchArea, maxBound):min(p + searchArea, maxBound) + 1]
                                == y1_test[p:p + 1]):
                            searchArea += 1
                            if np.isnan(searchArea) or searchArea > maxBound:
                                print("不存在对应的索引,range is NaN")
                                exit(0)
        print("查询范围searchArea : ", searchArea)
    if RMSEValue < 0:
        print("未进入模型选择阶段")
        exit(-1)

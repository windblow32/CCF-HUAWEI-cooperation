# 数据集
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 评估
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn import metrics

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from score_config import config
import joblib

class gbr:
    def __init__(self, X, y, seed):
        self.X = X
        self.y = y
        self.seed = seed

    def getscore(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=self.seed)

        # Fit regression model, 调参
        # subsample在0.5到0.8之间, n_estimators过大会过拟合。
        gbr = GradientBoostingRegressor(loss='quantile',n_estimators=150,learning_rate=0.95,subsample=0.7)

        # fit 拟合
        gbr.fit(self.X, self.y)

        # store
        joblib.dump(gbr, r'E:\whz\new_index_learning\software_newspaper\trainedModel\gbr.pkl',compress=3)

        # Predict

        y_pred = gbr.predict(X_test)

        # 计算RMSE(均方根误差，标准误差)
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        R2_score = r2_score(y_test, y_pred)

        # median_absolute_error
        Median_absolute_error = median_absolute_error(y_test, y_pred)

        # mean_absolute_error
        Mean_absolute_error = mean_absolute_error(y_test, y_pred)

        # # mean_squared_log_error 数据中有负值时无法使用
        # Mean_squared_log_error = mean_squared_log_error(y_test,y_pred))

        # explained_variance_score
        Explained_variance_score = explained_variance_score(y_test, y_pred)

        # 打分调参

        cfg = config()
        s_RMSE, s_Median_AE, s_Mean_AE = cfg.load()
        # score
        score = RMSE * s_RMSE + Median_absolute_error * s_Median_AE + Mean_absolute_error * s_Mean_AE

        return score
    def getname(self):
        print("Gradient Boosting Regressor\n")

    def getdataset(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=1)
        print(X_train)

    @classmethod
    def test(self, testSet):
        gbr = joblib.load(r'E:\whz\new_index_learning\software_newspaper\trainedModel\gbr.pkl')
        test_predict = gbr.predict(testSet)
        # print(test_predict)
        return test_predict

    @classmethod
    def testWithTime(self, testSet):
        start = time.time_ns()
        etr = joblib.load(r'E:\whz\new_index_learning\software_newspaper\trainedModel\gbr.pkl')
        end = time.time_ns()
        ioTime = end - start
        test_predict = etr.predict(testSet)
        # print(test_predict)
        return ioTime, test_predict

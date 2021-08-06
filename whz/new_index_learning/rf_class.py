from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics

# 交叉验证
from sklearn.model_selection import StratifiedKFold
# 网格搜索
from sklearn.model_selection import GridSearchCV

# 指标
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from score_config import config
import joblib

class rf:
    def __init__(self, X, y, seed):
        self.X = X
        self.y = y
        self.seed = seed

    def getscore(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=self.seed)

        # 调参
        # n_estimators=100, *, criterion='mse',
        # max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        # max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
        # min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
        # verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None

        regressor = RandomForestRegressor(n_estimators=150, random_state=1, max_depth=6, min_samples_split=3,
                                          min_impurity_decrease=0.1, warm_start=True)
        # 交叉验证
        cross_score = cross_val_score(regressor, self.X, self.y, cv=10, scoring="neg_mean_squared_error")

        regressor.fit(X_train, y_train)
        # 保存模型
        joblib.dump(regressor, r'E:\whz\new_index_learning\software_newspaper\trainedModel\rf.pkl',compress=3)


        y_pred = regressor.predict(X_test)

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
        print("Random Forest Regression\n")

    def getdataset(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=1)
        print(X_train)

    @classmethod
    def test(self, testSet):
        rf = joblib.load(r'E:\whz\new_index_learning\software_newspaper\trainedModel\rf.pkl')
        test_predict = rf.predict(testSet)
        # print(test_predict)
        return test_predict





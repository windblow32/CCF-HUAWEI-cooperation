import re
import gc
import time
import wrapt
from line_profiler import LineProfiler
import pyodbc
import numpy as np
import btree
import struct
import csv
from db import Pdb
from RMI import RMI

lp = LineProfiler()


def lp_wrapper():
    @wrapt.decorators
    def wrapper(func, instance, args, kwargs):
        global lp
        lp_wrapper = lp(func)
        res = lp_wrapper(*args, **kwargs)
        lp.print_stats()
        return res

    return wrapper


if __name__ == "__main__":
    #     ODBC = 'DRIVER={GaussMPP};SERVER=127.0.0.1;DATABASE=postgres;UID=omm;PWD=gauss_123'
    #     while True:
    #         # SERVER = input('Please input SERVER(default 127.0.0.1):')
    #         # DATABASE = input('Please input DATABASE(default postgres):')
    #         # UID = input('Please input USERID(default omm):')
    #         # PWD = input('Please input password(default gauss_123)')
    #         sql = input("sql# ").strip()
    #         if sql == "exit":
    #             break
    #         elif len(sql) == 0:
    #             continue
    #         elif re.match(r"^-[r,R][m,M][i,I]", sql):
    #             print('RMI mode:\n')
    #             sql = (sql[4:]).strip().lower()
    #             if re.match(r"^[s,S][e,E]", sql):
    #                 db = pyodbc.connect(ODBC)
    #                 cursor = db.cursor()
    #                 cursor.execute(sql)
    #                 data = cursor.fetchall()
    #                 data = np.array(data).reshape(-1)
    #                 cursor.close()
    #                 li = RMI()
    #                 li.train(data)
    #                 sql = sql.split(' ')
    #                 s = eval(sql[sql.index("where")+3])
    #                 pos = li.search(s)
    #                 if pos:
    #                     print(s, "found! pos: ", pos)
    #                 else:
    #                     print(s, "not found!")
    #                 err = np.mean(np.abs(li.error_bound))
    #                 print("mean error: \n", err)
    # data = np.hstack((np.random.randint(25000000, size=2500000), np.random.normal(250000, 10, size=3000000),
    #                   np.random.uniform(1, 25000000, size=3000000), np.random.poisson(250000, size=2500000),
    #                   np.random.exponential(20000000, size=3000000)))
    data = np.random.randint(0, 100000, 1000)
    data = np.sort(data).reshape(-1)
    # file = np.savetxt("data.csv", data, delimiter=",")
    print("______________start BTree______________")
    bt = btree.BTree(2)
    start = time.time()
    set_x = np.arange(len(data))
    set_y = data.copy()
    bt.build(set_x, set_y)
    bt.search(data)
    end = time.time()
    print("B Tree build time: ", end - start)
    print("=======================================")
#                 err = 0
#                 print("____________calculate error____________")
#                 start = time.time()
#                 for ind in range(len(set_x)):
#                     pre = bt.predict(set_x[ind])
#                     err += abs(pre - set_y[ind])
#                     if err != 0:
#                         flag = 1
#                         pos = pre
#                         off = 1
#                         while pos != set_y[ind]:
#                             pos += flag * off
#                             flag = -flag
#                             off += 1
#                 end = time.time()
#                 search = (end - start) / len(set_x)
#                 print("B Tree search time: ", search)
#                 mean_error = err * 1.0 / len(set_x)
#                 print("mean error = ", mean_error)
#                 print("=======================================")
#                 del bt
#                 del db
#                 gc.collect()
#         else:
# db = Pdb(ODBC, sql)
# if re.match(r'^[c,C]', sql):
#     db.crtb()
# elif re.match(r'^[i,I]', sql):
#     db.insr()
# elif re.match(r'^[d,D][e,E]', sql):
#     db.deld()
# elif re.match(r'^[d,D][r,R]', sql):
#     db.dptb()
# elif re.match(r'[s,S][e,E]', sql):
#     db.quey()
# elif re.match(r'^[u,U][p,P]', sql):
#     db.updt()
# else:
#     print('尚不支持\n')
# del db

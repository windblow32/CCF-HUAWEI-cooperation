# 连接gaussdb说明文档

## 官方文档

GaussDB T参考： [Python连接GaussDB T数据库并查询SQL](https://bbs.huaweicloud.com/blogs/147120)

GaussDB DWS参考：[使用Python连接GaussDB（DWS）](https://bbs.huaweicloud.com/blogs/detail/227290)

## 个人封装

方法同使用pyodbc连接PostgreSQL相同

```python
'''
下面是封装pyodbc连接参考
'''
import pyodbc


class Pdb:
    def __init__(self, odbc, psql):
        self.odbc = odbc
        self.psql = psql

    # 创建数据库表
    def crtb(self):
        db = pyodbc.connect(self.odbc)
        cursor = db.cursor()
        try:
            cursor.execute(self.psql)
            # 提交到数据库执行
            db.commit()
            print('CREATE TABLE SUCCESS!')
        # 捕获与数据库相关的错误
        except pyodbc.Error as err:
            print('CREATE TABLE FAILED, REASON:{}'.format(err))
            # 如果发生错误就回滚
            db.rollback()
        finally:
            # 关闭数据库连接
            db.close()

    # 删除数据库表
    def dptb(self):
        db = pyodbc.connect(self.odbc)
        cursor = db.cursor()
        try:
            cursor.execute(self.psql)
            db.commit()
            print('DROP TABLE SUCCESS.')
        # 捕获与数据库相关的错误
        except pyodbc.Error as err:
            print('DROP TABLE FAILED, REASON:{}'.format(err))
            # 如果发生错误就回滚
            db.rollback()
        finally:
            # 关闭数据库连接
            db.close()

    # 数据库插入
    def insr(self):
        db = pyodbc.connect(self.odbc)
        cursor = db.cursor()
        try:
            cursor.execute(self.psql)
            db.commit()
            print('INSERT SUCCESS.')
        except pyodbc.Error as err:
            print('INSERT FAILED, REASON:{}'.format(err))
            db.rollback()
        finally:
            db.close()

    # 数据库删除
    def deld(self):
        db = pyodbc.connect(self.odbc)
        cursor = db.cursor()
        try:
            cursor.execute(self.psql)
            db.commit()
            print('DELETE SUCCESS.')
        # 捕获与数据库相关的错误
        except pyodbc.Error as err:
            print('DELETE FAILED, REASON:{}'.format(err))
            db.rollback()
        finally:
            db.close()

    # 数据库更新
    def updt(self):
        db = pyodbc.connect(self.odbc)
        cursor = db.cursor()
        try:
            cursor.execute(self.psql)
            db.commit()
            print('UPDATE SUCCESS.')
        # 捕获与数据库相关的错误
        except pyodbc.Error as err:
            print('UPDATE FAILED, REASON:{}'.format(err))
            # 如果发生错误就回滚
            db.rollback()
        finally:
            db.close()

    # 数据库查询
    def quey(self):
        db = pyodbc.connect(self.odbc)
        cursor = db.cursor()
        try:
            cursor.execute(self.psql)
            '''
            fetchone()：该方法获取下一个查询结果集，结果集是一个对象。
            fetchall()：接收全部返回结果行。
            rowcount：返回执行 execute（）方法后影响的行数
            '''
            # 获取所有记录列表
            data = cursor.fetchall()
            for r in data:
                print(r)
        # 捕获与数据库相关的错误
        except pyodbc.Error as err:
            print('QUERY MySQL table FAILED, REASON:{}'.format(err))
        finally:
            db.close()

```

下面在主函数中调用:

```python
from db import Pdb
if __name__ == "__main__":
ODBC = 'DRIVER={GaussMPP};SERVER=127.0.0.1;DATABASE=postgres;UID=(omm自行修改);PWD=(gauss_123自行修改)'
        while True:
            # SERVER = input('Please input SERVER(default 127.0.0.1):')
            # DATABASE = input('Please input DATABASE(default postgres):')
            # UID = input('Please input USERID(default omm):')
            # PWD = input('Please input password(default gauss_123)')
            sql = input("sql# ").strip()
            if sql == "exit":
                break
            elif len(sql) == 0:
                continue
            elif re.match(r"^-[r,R][m,M][i,I]", sql):
                print('RMI mode:\n')
                sql = (sql[4:]).strip().lower()
                if re.match(r"^[s,S][e,E]", sql):
                    db = pyodbc.connect(ODBC)
                    cursor = db.cursor()
                    cursor.execute(sql)
                    data = cursor.fetchall()
                    data = np.array(data).reshape(-1)
                    cursor.close()
                    sql = sql.split(' ')
                    s = eval(sql[sql.index("where")+3])
				         db = Pdb(ODBC, sql)
                if re.match(r'^[c,C]', sql):
                    db.crtb()
                elif re.match(r'^[i,I]', sql):
                    db.insr()
                elif re.match(r'^[d,D][e,E]', sql):
                    db.deld()
                elif re.match(r'^[d,D][r,R]', sql):
                    db.dptb()
                elif re.match(r'[s,S][e,E]', sql):
                    db.quey()
                elif re.match(r'^[u,U][p,P]', sql):
                    db.updt()
                else:
                    print('尚不支持\n')
                del db

```

## 目前learnedIndex所用数据集格式

分为两列，暂定为第一列是索引，数据要求是整型；第二列是对应索引的数据，data要求是数值类型，double 或 long都被允许，多表联合查询暂时没有尝试

| index |    data    |
| :---: | :--------: |
|   0   |   0.5123   |
|   1   |    123     |
|   2   | 4584541312 |
|  ……   |     ……     |


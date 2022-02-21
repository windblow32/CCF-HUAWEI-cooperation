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

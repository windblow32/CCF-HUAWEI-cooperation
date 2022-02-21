import numpy as np
import pyodbc


def connect_gaussdb():
    [x for x in pyodbc.drivers() if x.startswith('GaussMPP')]
    cnxn = pyodbc.connect(
        'DRIVER={GaussMPP};SERVER=localhost;PORT=26000;DATABASE=postgres;UID=omm;PWD=gauss@333'
    )
    gdbc = cnxn.cursor()
    '''
    这里根据需要封装,例如下面将一个txt文件转化为一张表
    '''
    name = 'x.txt'
    if name == 'x.txt':
        gdbc.execute("drop table if exists x")
        gdbc.execute(
            "create table x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float,dim7 float,"
            "dim8 float, "
            "dim9 float,dim10 float,dim11 float,dim12 float,dim13 float,dim14 float,dim15 float,dim16 float,"
            "dim17 float, "
            "dim18 float,dim19 float,dim20 float,dim21 float,dim22 float,dim23 float,dim24 float,dim25 float,"
            "dim26 float, "
            "dim27 float,dim28 float,dim29 float,dim30 float)")
        with open(name, 'r') as f:
            lines = f.readlines()
            total = []
            for a in lines:
                s = []
                ta = a.split(',')
                for j in range(len(ta) - 1):
                    if j > 1:
                        s.append(float(eval(ta[j])))
                    else:
                        s.append(ta[j])
                total.append(s)
        for i in range(len(total)):
            gdbc.execute(
                f"insert into x values({total[i][0]},{total[i][1]},{total[i][2]},{total[i][3]},{total[i][4]},{total[i][5]},{total[i][6]},{total[i][7]},{total[i][8]},{total[i][9]},{total[i][10]},{total[i][11]}"
                f",{total[i][12]},{total[i][13]},{total[i][14]},{total[i][15]},{total[i][16]},{total[i][17]},"
                f"{total[i][18]},{total[i][19]},{total[i][20]},{total[i][21]},{total[i][22]},{total[i][23]},{total[i][24]},{total[i][25]},{total[i][26]},"
                f"{total[i][27]},{total[i][28]},{total[i][29]})")
    gdbc.commit()
    gdbc.close()
    cnxn.close()

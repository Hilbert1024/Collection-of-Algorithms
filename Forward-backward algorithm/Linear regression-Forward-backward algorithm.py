#王行健 18210180088 数学科学学院

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

#向前向后逐步回归算法

def ols(X,y):#X为设计矩阵
    return (X.T*X).I*X.T*y

def SS_Res(X,y):#X为数据矩阵
    n=X.shape[0]
    one=np.matrix([1]*n).T
    Xd=np.hstack((one,X))
    beta=ols(Xd,y)
    return float((y-Xd*beta).T*(y-Xd*beta))

def SS_T(y):#X为数据矩阵
    one=np.matrix([1]*len(y)).T
    y_bar=np.matrix(np.mean(y)*one)
    return float((y-y_bar).T*(y-y_bar))

def F_test(X,y):#X为数据矩阵
    n,p=X.shape #p为自变量个数
    one=np.matrix([1]*n).T
    Xd=np.hstack((one,X))
    A=(Xd.T*Xd).I
    beta=ols(Xd,y)
    sigma2=SS_Res(X,y)/(n-p-1)
    F=[]
    for i in range(p+1):
        F.append(float(beta[i]**2/(sigma2*A[i,i])))
    return F

def dic_min_key_value(dic):
    value_list=list(dic.values())
    key_list=list(dic.keys())
    max_value=max(value_list)
    min_value=min(value_list)
    max_value_index=value_list.index(max(value_list))
    min_value_index=value_list.index(min(value_list))
    max_key=key_list[max_value_index]
    min_key=key_list[min_value_index]
    dics={max_key:max_value,min_key:min_value}
    return dics

def All_Possible_Regressions(X,y):#X为数据矩阵
    n,p=X.shape #p为自变量个数
    SS_Res_fullmodel=SS_Res(X,y)
    sigma=SS_Res_fullmodel/(n-p-1)
    SS_T_fullmodel=SS_T(y)
    mat_dic={i:X[:,i] for i in range(p)}#自变量从0开始计数
    k=0
    MS_Res,Cp,Rp={},{},{}
    while(k<p):
        gen=itertools.combinations(range(p),k+1)
        for tuples in gen:
            q=len(tuples)+1
            temp=[]
            for i in tuples:
                temp.append(mat_dic[i])
            Xm=np.hstack(tuple(temp))
            MS_Res[str(tuples)]=SS_Res(Xm,y)/(n-q)
            Cp[str(tuples)]=SS_Res(Xm,y)/sigma-(n-2*q)
            Rp[str(tuples)]=1-SS_Res(Xm,y)/SS_T_fullmodel
        k+=1
    return MS_Res,Cp,Rp 

def forward_Regressions(X,y,F_test_value):
    '''
    X为数据矩阵
    F_test_value为一列F检验值，需先查表得.
    F_test_value[k]=F(1,n-k-1)
    '''
    n,p=X.shape #p为自变量个数
    mat_dic={i:X[:,i] for i in range(p)}
    k=0
    lists=list(range(p))
    temp=[]
    while(k<p):
        F=[]
        mat=[]
        for j in temp:
            mat.append(mat_dic[j])
        q=len(temp)+1
        for i in lists:
            mat.append(mat_dic[i])
            Xm=np.hstack(tuple(mat))
            F.append(F_test(Xm,y)[-1])
            mat.pop()
        print('step %d'%(k+1))
        print('F test of variables',lists,'is',F)
        max_value=max(F)
        if max_value<F_test_value[q-1]:
            print('No variables can be added in.')
            print('================================')
            break
        max_index=F.index(max_value)
        print('variable %d is added in.'%(lists[max_index]))
        print('================================')
        temp.append(lists[max_index])
        lists.pop(max_index)
        k+=1
    mat=[]
    for j in temp:
        mat.append(mat_dic[j])
    Xm=np.hstack(tuple(mat))
    one=np.matrix([1]*n).T
    Xd=np.hstack((one,Xm))
    beta=ols(Xd,y)
    print('The finnal variable is',temp)
    print('Corresponding coefficient is',[float(b) for b in beta],'where the first one is content term')
    return beta,temp


def backward_Regressions(X,y,F_test_value):
    '''
    X为数据矩阵
    F_test_value为一列F检验值，需先查表得.
    F_test_value[k]=F(1,n-k-1)
    '''
    n,p=X.shape #p为自变量个数
    mat_dic={i:X[:,i] for i in range(p)}
    k=0
    temp=list(range(p))
    while(k<p):
        F=[]
        mat=[]
        q=len(temp)+1
        for i in temp:
            mat.append(mat_dic[i])
            Xm=np.hstack(tuple(mat))
            F=F_test(Xm,y)[1:]
        print('step %d'%(k+1))
        print('F test of variables',temp,'is',F)
        min_value=min(F)
        if min_value>F_test_value[q-1]:
            print('No variables can be rejected.')
            print('================================')
            break
        min_index=F.index(min_value)
        print('variable %d is rejected.'%(temp[min_index]))
        print('================================')
        temp.pop(min_index)
        k+=1
    mat=[]
    for j in temp:
        mat.append(mat_dic[j])
    Xm=np.hstack(tuple(mat))
    one=np.matrix([1]*n).T
    Xd=np.hstack((one,Xm))
    beta=ols(Xd,y)
    print('The finnal variable is',temp)
    print('Corresponding coefficient is',[float(b) for b in beta],'where the first one is content term')
    return beta,temp

def stepward_Regressions(X,y,F_test_value):
    n,p=X.shape #p为自变量个数
    mat_dic={i:X[:,i] for i in range(p)}
    k=0
    lists=list(range(p))#待选变量
    temp=[]#输出变量
    ADD=True
    while(ADD):
   	    #引入变量模块
        F=[]
        mat=[]
        for j in temp:
            mat.append(mat_dic[j])
        q=len(temp)+1
        for i in lists:
            mat.append(mat_dic[i])
            Xm=np.hstack(tuple(mat))
            F.append(F_test(Xm,y)[-1])
            mat.pop()
        print('step %d'%(k+1))
        print('F test of variables',lists,'is',F)
        max_value=max(F)
        if max_value<F_test_value[q-1]:
            print('No variables can be added in.')
            print('================================')
            ADD=False
        if ADD:
            max_index=F.index(max_value)
            print('variable %d is added in.'%(lists[max_index]))
            print('================================')
            temp.append(lists[max_index])
            lists.pop(max_index)
        k+=1
        #剔除变量模块
        if(len(temp)<3):#当只有两个变量时不剔除变量
            continue
        print('******************************')
        print('Reject Variables')
        mat=[]
        for j in temp:
            mat.append(mat_dic[j])
        Xm=np.hstack(tuple(mat))
        variable=backward_Regressions(Xm,y,F_test_value)[1]
        temp_now=[temp[i] for i in variable]
        if set(temp_now)!=set(temp):
            ADD=True
            for v in temp:
                if v not in temp_now:
                    lists.append(v)
            temp=temp_now
        print('******************************')
    mat=[]
    for j in temp:
        mat.append(mat_dic[j])
    Xm=np.hstack(tuple(mat))
    one=np.matrix([1]*n).T
    Xd=np.hstack((one,Xm))
    beta=ols(Xd,y)
    print('================================')
    print('The finnal variable is',temp,)
    print('================================')
    print('Corresponding coefficient is',[float(b) for b in beta],'where the first one is content term')
    return beta,temp

def confidence_interval(X,y,t_value):#X为数据矩阵
    n,p=X.shape
    p+=1
    one=np.matrix([1]*n).T
    Xd=np.hstack((one,X))
    sigma=SS_Res(X,y)/(n-p)
    beta=ols(Xd,y)
    return [[float(Xd[i]*beta-t_value*np.sqrt(sigma*Xd[i]*(Xd.T*Xd).I*Xd[i].T)),float(Xd[i]*beta+t_value*np.sqrt(sigma*Xd[i]*(Xd.T*Xd).I*Xd[i].T))] for i in range(n)]



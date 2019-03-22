# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型,非线性分类问题,软间隔,kernel method
    """

    def __init__(self,sigma,epsilon,C):
    	self.maxiter = 100 # 最大循环次数
    	self.alpha = 0
    	self.bias = 0
    	self.sigma = sigma # Gaussian kernel参数
    	self.epsilon = epsilon # 精度
    	self.C = C # 惩罚参数
    	self.K = 0 # Kernel
    	self.n = 0
    	self.x = 0
    	self.y = 0

    def Gaussian_kernel(self, x):
        n = x.shape[0]
        self.K = np.ones([n,n])
        for i in range(n):
        	for j in range(i+1,n):
        		self.K[i][j] = self.K[j][i] = np.exp(- (sum((x[i] - x[j]) ** 2)) / self.sigma)

    def update(self,train_data):
        self.n = data_train.shape[0]
        self.x = data_train[:, :2]
        self.y = data_train[:,2]
    	

    def SMO(self, data_train):
        """
        Sequential minimal optimization,求解凸二次规划问题
        """

        def gf(i):
        	return sum(self.alpha*y*np.array([K[i][j] for j in range(n)])) + self.bias

        def Ef(i):
        	return gf(i) - y[i]

        def KKT_condition(i):
        	if self.alpha[i] == 0:
        		if y[i]*g[i] > 1 + self.epsilon:
        			return False
        	elif self.alpha[i] >0 and self.alpha[i] < self.C:
        		if abs(y[i]*g[i]-1) > self.epsilon:
        			return False
        	else:
        		if y[i]*g[i] < 1 - self .epsilon:
        			return False
        	return True

        def firstv(): #返回索引
        	lessC = self.alpha[self.alpha < self.C]
        	lessC_index = np.array([np.where(self.alpha == x)[0][0] for x in lessC])
        	if len(lessC) != 0:
        		for index in lessC_index:
        			if not KKT_condition(index):
        				return index
        	for index in range(n):
        		if not KKT_condition(index):
        			return index
        	return -1

        def secondv(i): #返回索引
        	if E[i] > 0:
        		return E.index(min(E))
        	else:
        		return E.index(max(E))

        def optimize_alpha(i,j): # 输入索引
        	alpha1 = self.alpha[i]
        	alpha2 = self.alpha[j]
        	eta = K[i][i] + K[j][j] - 2*K[i][j]
        	alpha2_unc = alpha2 + y[j] * (E[i] - E[j]) / eta
        	if y[i] != y[j]:
        		L = max(0,alpha2 - alpha1)
        		H = min(self.C ,self.C + alpha2 - alpha1)
        	else:
        		L = max(0,alpha2 + alpha1 - self.C)
        		H = min(self.C ,alpha2 + alpha1)
        	if alpha2_unc > H:
        		alpha2_new = H
        	elif alpha2_unc < L:
        		alpha2_new = L
        	else:
        		alpha2_new = alpha2_unc
        	alpha1_new = alpha1 + y[i]*y[j]*(alpha2 - alpha2_new)
        	self.alpha[i],self.alpha[j] = alpha1_new,alpha2_new # 更新alpha1,alpha2
        	b = self.bias
        	b1_new = -E[i] - y[i]*K[i][i]*(alpha1_new - alpha1) - y[j]*K[i][j]*(alpha2_new - alpha2) + b
        	b2_new = -E[j] - y[i]*K[i][j]*(alpha1_new - alpha1) - y[j]*K[j][j]*(alpha2_new - alpha2) + b
        	b_new = (b1_new + b2_new)/2
        	self.bias = b_new # 更新B
        	supportvec_index = np.where(self.alpha > 0)[0]
        	E[i] = sum([y[k]*self.alpha[k]*K[i][k] for k in supportvec_index]) + self.bias - y[i] # 更新E
        	E[j] = sum([y[k]*self.alpha[k]*K[j][k] for k in supportvec_index]) + self.bias - y[j] 

        def Quit():
        	if sum(self.alpha*y) != 0:
        		return False
        	else:
        		for i in range(n):
        			if self.alpha[i] < 0 or self.alpha[i] > self.C or (not KKT_condition(i)):
        				return False
        	return True

        self.update(data_train)
        n = self.n
        x = self.x
        y = self.y
        self.Gaussian_kernel(x)
        K = self.K
        self.alpha = np.zeros(n)
        g = [gf(i) for i in range(n)]
        E = [Ef(i) for i in range(n)] # E的列表
        i = firstv()
        j = secondv(i)
        optimize_alpha(i,j)
        i = firstv()
        j = secondv(i)
        while(not Quit() and self.maxiter > 0):
         	i = firstv()
         	j = secondv(i)
         	optimize_alpha(i,j)
         	self.maxiter -= 1

        return self.alpha


    def train(self, data_train):
        """
        训练模型。
        """
        self.alpha = self.SMO(data_train)
        n = self.n
        x = self.x
        y = self.y
        K = self.K
        alpha_lessC = self.alpha[self.alpha < self.C]
        alpha_star = alpha_lessC[alpha_lessC > 0][0]
        as_index = np.where(self.alpha == alpha_star)[0][0]
        self.bias = y[as_index] - sum([self.alpha[i] * y[i] * K[i][as_index] for i in range(n)])

    def predict(self, x):
        """
        预测标签。
        """
        pred = []
        for vec in x:
        	pred.append(np.sign(sum([self.alpha[i] * self.y[i] * np.exp(-sum((vec-self.x[i])**2) / self.sigma) for i in range(self.n)])))
        return pred
        	
if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_kernel.txt'
    test_file = 'data/test_kernel.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM(10,0.01,5)  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))

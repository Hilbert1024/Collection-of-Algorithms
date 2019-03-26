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


class LR_GD_multi():
    """
    Logistic Regression模型,采用梯度下降法
    """

    def __init__(self,learning_rate,epsilon):
        self.maxiter = 5000
        self.weight = 0
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        

    def train(self, data_train):
        """
        训练模型。
        """
        n = data_train.shape[0]
        x_temp = data_train[:, :2]
        x = np.array([np.append(x,1) for x in x_temp])
        y = data_train[:, 2] # y in {-1,0,1}
        # 似然函数 pi_1(x_i) ^ (y_i * (y_i + 1) / 2) + pi_2(x_i)^(1 - y_i**2) + (1 - pi_1(x_i) - pi_2(x_i)) ^  (y_i * (y_i - 1) / 2)
        self.weight = np.zeros((2,len(x[0])))
        def g(w):
        	gradient = np.zeros((2,len(x[0])))
        	for i in range(n):
        		gradient[0] += (1 / (1 + np.exp(-np.dot(x[i], w[0])) + np.exp(np.dot(x[i], w[1] - w[0]))) - (y[i] * (y[i] + 1)) / 2) * x[i] 
        		gradient[1] += (1 / (1 + np.exp(-np.dot(x[i], w[1])) + np.exp(np.dot(x[i], w[0] - w[1]))) - (y[i] * (y[i] - 1)) / 2) * x[i] 
        	return gradient
        while(sum(sum(g(self.weight)**2)) > self.epsilon and self.maxiter > 0):
        	self.weight -= self.learning_rate*g(self.weight)
        	self.maxiter -= 1
        print(self.weight,self.maxiter)



    def predict(self, x):
        """
        预测标签。
        """
        pred = []
        alpha = [1,-1,0]
        for vec in x:
        	vec = np.append(vec,1)
        	temp_1 = np.exp(np.dot(vec, self.weight[0]))
        	temp_2 = np.exp(np.dot(vec, self.weight[1]))
        	temp = [temp_1,temp_2,1]
        	pred.append(alpha[temp.index(max(temp))])
        print(pred)
        return pred
        


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_multi.txt'
    test_file = 'data/test_multi.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    LR = LR_GD_multi(0.001,0.1)  # 初始化模型
    LR.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = LR.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = LR.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))

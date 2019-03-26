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


class LR_GD():
    """
    Logistic Regression模型,采用梯度下降法
    """

    def __init__(self,learning_rate,epsilon):
        self.maxiter = 10000
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
        y = data_train[:, 2] # y in {-1,1}
        # 似然函数 pi(x_i)^((y_i+1)/2)+(1-pi(x_i))^((1-y_i)/2)
        self.weight = np.zeros(len(x[0]))
        def g(w):
        	gradient = np.zeros(len(x[0]))
        	for i in range(n):
        		gradient += (1 / (1 + np.exp(-np.dot(x[i], w))) - (y[i] + 1) / 2) * x[i]
        	return gradient 
        	# i = np.random.randint(n)
        	# return (y[i] - 1 / (1 + np.exp(-np.dot(x[i], w)))) * x[i]
        while(sum(g(self.weight)**2) > self.epsilon and self.maxiter > 0):
        	self.weight -= self.learning_rate*g(self.weight)
        	self.maxiter -= 1
        print(self.weight,self.maxiter)



    def predict(self, x):
        """
        预测标签。
        """
        pred = []
        for vec in x:
        	vec = np.append(vec,1)
        	temp = np.exp(np.dot(vec, self.weight)) 
        	pred.append(1 if temp > 1 else -1)
        return pred

        


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    LR = LR_GD(0.001,0.1)  # 初始化模型
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

# python: 3.6.5
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt

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


class Perceptron():
    """
    感知机模型
    对线性可分数据，当没有误分类点时，循环结束
    对于线性不可分数据，我们只能设置最大迭代次数来终止循环
    """

    def __init__(self, dimension):
        self.maxIter = 10000
        self.weight = np.zeros(dimension) #初始化参数
        self.bias = 0
        self.learning_rate = 1


    def train(self, data_train):
        """
        训练模型。
        """

        while(self.maxIter>0):
            Flag = True
            for vec in data_train:
                x = vec[:2]
                y = vec[2]
                if y*(np.dot(self.weight,x)+self.bias) <= 0:
                    self.weight += self.learning_rate*y*x
                    self.bias += self.learning_rate*y
                    Flag = False
            if Flag:
                break
            self.maxIter -=1
        # print(self.maxIter,self.weight,self.bias) #查看训练参数可取消注释


    def predict(self, x):
        """
        预测标签。
        """
        return np.array(list(map(lambda x:np.sign(np.dot(self.weight,x)+self.bias),x)))


class Perceptron_Pocket():
    """
    感知机模型口袋算法(Pocket Algorithm)
    对于线性不可分数据,可每次找到一个误分点更新参数,若更新后的参数作用到整个训练集上得到的误分点少,则更新参数,否则不更新
    口袋算法为了处理非线性可分数据,它保证每一次改变都是最好的线段,但必须人为限制迭代次数
    """

    def __init__(self, dimension):
        self.maxIter = 5000
        self.weight = np.zeros(dimension) #初始化参数
        self.bias = 0
        self.learning_rate = 0.1


    def train(self, data_train):
        """
        训练模型。
        """

        while(self.maxIter>0):
            for vec in data_train:
                x = vec[:2]
                y = vec[2]
                if y*(np.dot(self.weight,x)+self.bias) <= 0: #找到误分点
                    temp_weight = self.weight + self.learning_rate*y*x
                    temp_bias = self.bias + self.learning_rate*y
                    new = np.array(list(map(lambda x:np.dot(temp_weight,x)+temp_bias,data_train[:,:2])))*data_train[:,2]
                    old = np.array(list(map(lambda x:np.dot(self.weight,x)+self.bias,data_train[:,:2])))*data_train[:,2]
                    if len(new[new<=0])<len(old[old<=0]):
                        self.weight = temp_weight
                        self.bias = temp_bias
            self.maxIter -=1
        # print(self.maxIter,self.weight,self.bias) #查看训练参数可取消注释


    def predict(self, x):
        """
        预测标签。
        """
        return np.array(list(map(lambda x:np.sign(np.dot(self.weight,x)+self.bias),x)))



if __name__ == '__main__':
    # 载入数据
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    #使用训练集训练感知机模型
    Perceptron = Perceptron(2)
    Perceptron_Pocket = Perceptron_Pocket(2)
    Perceptron.train(data_train)
    Perceptron_Pocket.train(data_train)

    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = Perceptron.predict(x_train)  # 预测标签
    t_train_pred_pocket = Perceptron_Pocket.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = Perceptron.predict(x_test)
    t_test_pred_pocket = Perceptron_Pocket.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    acc_train_pocket = eval_acc(t_train, t_train_pred_pocket)
    acc_test_pocket = eval_acc(t_test, t_test_pred_pocket)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
    print("train accuracy(Pocket Algorithm): {:.1f}%".format(acc_train_pocket * 100))
    print("test accuracy(Pocket Algorithm): {:.1f}%".format(acc_test_pocket * 100))

import pandas as pd
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

dataset = datasets.load_boston()  # 导入数据集
data = dataset['data']  # 特征数据
m = data.shape[0]  # 得到样本个数
label = dataset['target']  # 数据对应的标签
y = np.array(label)  # 转化为数组
y = y.reshape((m, 1))  # (506,1)

feature = dataset['feature_names']  # 特征的名称
df = pd.DataFrame(data, columns=feature)  # 将特征数据转化为DF
df.insert(0, 'x0', 1)  # 插入常数列
x = np.array(df)  # (506,14)
theta = np.zeros((x.shape[1], 1))  # (14, 1)




class Linear:
    def __init__(self, feature, label):
        self.x = feature
        self.y = label
        self.m = self.x.shape[0]
        self.alpha = 0.01
        self.zx = 0

    def least_squares(self):
        w = self.x.T.dot(self.x)
        w = np.linalg.pinv(w)  # 无逆 求伪逆
        w = w.dot(self.x.T)
        w = w.dot(self.y)
        return w

    def bgradient_descent(self):
        x = self.x
        y = self.y
        alpha = self.alpha
        x = self.zscore()  # 标准化处理
        theta = np.zeros((x.shape[1], 1))  # 各特征参数
        temp = np.zeros((x.shape[1], 1))  # 储存运算后结果 不影响参数
        grad = np.ones((x.shape[1], 1))  # 梯度
        x = np.matrix(x)  # 转化为矩阵便于运算
        y = np.matrix(y)
        theta = np.matrix(theta)
        temp = np.matrix(temp)
        grad = np.matrix(grad)

        while np.sometrue(abs(grad) > 0.00001):
            temp = theta - (alpha / m) * (x.T * (x * theta - y))  # 参数迭代
            grad = x.T * (x * temp - y)  # 计算当前的梯度
            theta = temp  # 改变参数
        return theta

    def zscore(self):
        zx = pd.DataFrame(self.x)
        # 利用dataframe便捷操作数据
        # 对除了常数列的列进行标准化
        for i in range(1, 14):
            t = zx[zx.columns[i]]
            t = (t - t.mean()) / t.std()
            zx[zx.columns[i]] = t
        zx = np.array(zx)
        return zx

    def leastsquares_show(self):
        theta = self.least_squares()  # 得到计算出来的参数
        theta = np.matrix(theta)  # 转化出来的矩阵
        x = self.x
        x = pd.DataFrame(x)
        y_test = []
        # 往列表添加计算出来的预测值
        for i in range(x.shape[0]):
            x_test = np.matrix(x.iloc[i])
            x_test = np.matrix(x_test)
            test = (x_test) * theta
            y_test.append(test)
        # 根据预测值标签值为轴画图
        plt.scatter(y, y_test, color='red')
        # 绘画 y = x 衡量你和情况
        plt.axline(slope=1, xy1=(0, 0))
        # 添加轴名字
        plt.xlabel('标签值', fontproperties="STSong", fontsize=18, color='brown',
                   weight=15)
        plt.ylabel('预测值', fontproperties="STSong", fontsize=18, rotation=45,
                   color='brown')
        plt.title('leasts squares', fontsize=18, color='brown')
        plt.show()
        pass

    def bgd_show(self):
        theta = self.bgradient_descent()  # 得到计算出来的参数
        x = self.zscore() # 将参数标准化便于绘图
        y_test2 = []
        theta = np.matrix(theta)
        x = pd.DataFrame(x)
        # 往列表添加预测值
        for i in range(x.shape[0]):
            x_test = np.matrix(x.iloc[i])
            x_test = np.matrix(x_test)
            test = (x_test) * theta
            y_test2.append(test)
            pass
        y = np.array(self.y)
        # 根据预测值与标签值画图
        plt.scatter(y, y_test2, color='red')
        # y = x 绘画拟合情况
        plt.axline(slope=1, xy1=(0, 0))
        plt.xlabel('标签值', fontproperties="STSong", fontsize=18, color='brown',
                   weight=15)
        plt.ylabel('预测值', fontproperties="STSong", fontsize=18, rotation=45,
                   color='brown')
        plt.title('gradient descent', fontsize=18, color='brown')
        plt.show()
        pass
    pass

Boston = Linear(x, y)



import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## 处理数据
def load_data():
    # 从文件导入数据
    datafile = './housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data
  
# 获取数据
training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]

# 查看数据
# print(x[0])
# print(y[0])

## 模型设计
# 权重
# w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
# w = np.array(w).reshape([13, 1])

# # 取出第1条样本数据，观察样本的特征向量与参数向量相乘的结果。
# x1 = x[0]
# t = np.dot(x1, w)
# print(t)

# # 取b值
# b = -0.2
# z = t + b
# print(z)

class Network(object):
  def __init__(self, num_of_weights):
     # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
  
  def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
     
  ## 损失函数 
  def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost

  def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        
        return gradient_w, gradient_b
   
  def update(self, gradient_w5, gradient_w9, eta=0.01):
        net.w[5] = net.w[5] - eta * gradient_w5
        net.w[9] = net.w[9] - eta * gradient_w9
        
  def train(self, x, y, iterations=100, eta=0.01):
      points = []
      losses = []
      for i in range(iterations):
          points.append([net.w[5][0], net.w[9][0]])
          z = self.forward(x)
          L = self.loss(z, y)
          gradient_w, gradient_b = self.gradient(x, y)
          gradient_w5 = gradient_w[5][0]
          gradient_w9 = gradient_w[9][0]
          self.update(gradient_w5, gradient_w9, eta)
          losses.append(L)
          if i % 50 == 0:
              print('iter {}, point {}, loss {}'.format(i, [net.w[5][0], net.w[9][0]], L))
      return points, losses   
    
# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]
# 创建网络
net = Network(13)
num_iterations=2000
# 启动训练
points, losses = net.train(x, y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
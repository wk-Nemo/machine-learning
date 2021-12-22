# 导入图像读取第三方库
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# import paddle
# from paddle.nn import Linear
# import paddle.nn.functional as F
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F

# class MNIST(paddle.nn.Layer):
#     def __init__(self):
#         super(MNIST, self).__init__()
        
#         # 定义一层全连接层，输出维度是1
#         self.fc = paddle.nn.Linear(in_features=784, out_features=1)
        
#     # 定义网络结构的前向计算过程
#     def forward(self, inputs):
#         outputs = self.fc(inputs)
#         return outputs

class MNIST(paddle.nn.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是1
         self.fc = Linear(in_features=980, out_features=1)
         
    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层不使用激活函数
     def forward(self, inputs):
         x = self.conv1(inputs)
         x = F.relu(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool2(x)
         x = paddle.reshape(x, [x.shape[0], -1])
         x = self.fc(x)
         return x
       #网络结构部分之后的代码，保持不变

# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    # print(np.array(im))
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    # 图像归一化，保持和数据集的数据范围一致
    im = 1 - im / 255
    return im

# 定义预测过程
model = MNIST()
params_file_path = './mnist.pdparams'
img_path = './test/example_6.jpg'

# 加载模型参数
param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)

# 灌入数据
model.eval()
tensor_img = load_image(img_path)
tensor_img = np.reshape(tensor_img, [1, 28, 28]).astype('float32')
result = model(paddle.to_tensor([tensor_img]))
print('result', result)

#  预测输出取整，即为预测的数字，打印结果
print("predict number", result.numpy().astype('int32'))
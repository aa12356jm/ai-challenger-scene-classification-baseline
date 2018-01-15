'''
本模块的主要作用是：计算整个训练集的均值和方差，然后使用均值和方差对图像进行预处理，也就是将图像减去均值除以标准差

所以我们首先要在训练集上进行均值和方差的计算，方法非常简单，遍历每一张图片，然后计算每个channel上的均值和方差即可
'''

import mxnet as mx
from mxnet import nd
import numpy as np
import os

path = './data/ai_challenger_scene_train_20170904/scene_train_images_20170904/'

img_list = os.listdir(path) #找到这个路径下的所有文件名

#存储所有图像的RGB值
r = 0  # r mean
g = 0  # g mean
b = 0  # b mean

#存储所有图像的RGB值的平方，为了计算方差
r_2 = 0  # r^2
g_2 = 0  # g^2
b_2 = 0  # b^2

total = 0
#遍历训练集的每一张图片
for img_name in img_list:
    img = mx.image.imread(path + img_name)  # ndarray, width x height x 3
    img = img.astype('float32') / 255. #将所有像素值归一化到[0,1]
    total += img.shape[0] * img.shape[1] #图像的宽*高，总像素数

    r += img[:, :, 0].sum().asscalar() #将所有像素的r值相加
    g += img[:, :, 1].sum().asscalar()
    b += img[:, :, 2].sum().asscalar()

    r_2 += (img[:, :, 0] ** 2).sum().asscalar()#将所有像素的r值的平方相加
    g_2 += (img[:, :, 1] ** 2).sum().asscalar()
    b_2 += (img[:, :, 2] ** 2).sum().asscalar()

#得到整个训练集的均值
r_mean = r / total
g_mean = g / total
b_mean = b / total

#得到整个训练集的方差
r_var = r_2 / total - r_mean ** 2
g_var = g_2 / total - g_mean ** 2
b_var = b_2 / total - b_mean ** 2

#显示均值和方差
print('mean r: {} g: {}, b: {}'.format(r_mean, g_mean, b_mean))
print('var r: {}, g: {}, b: {}'.format(np.sqrt(r_var), np.sqrt(g_var), np.sqrt(b_var)))



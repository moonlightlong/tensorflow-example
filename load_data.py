# -*- coding:utf-8 -*-
'''加载数据 CIFAR10'''
import numpy as np
import pickle
import sys
import random


def load_CIFAR10_batch(filename):
    """加载数据CIFAR10"""
    with open(filename, 'rb') as f:
        # 判断Python版本
        if sys.version_info[0] < 3:
            datas = pickle.load(f)
        else:
            datas = pickle.load(f, encoding='latin1')
        x = datas['data']
        y = datas['data']
        x = x.astype(float)
        y = np.array(y)
    return x, y


def load_data():
    '''加载数据'''
    xs = list()
    ys = list()
    # 加载训练集
    # 每一个batch文件包括10000个训练图片
    # 注意这里的batch是外部文件命名中的字符，不同与后面数据里面的batch。
    # batch：一批
    for i in range(1, 6):
        filename = 'cifar-10-batches-py/data_batch_' + str(i)
        X, Y = load_CIFAR10_batch(filename)
        xs.append(X)
        ys.append(Y)
    # 将数据集拼接起来
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    # 删除变量xs,ys.优化内存
    del xs, ys

    # 加载测试集
    # 测试集只有一个batch文件
    x_test, y_test = load_CIFAR10_batch('cifar-10-batches-py/test_batch')

    # 数据集的类别
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']

    # 数据规范化
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image

    data_dict = {
        'images_train': x_train,
        'labels_train': y_train,
        'images_test': x_test,
        'labels_test': y_test,
        'classes': classes
    }
    return data_dict


def reshape_data(data_dict):
    '''将数据转变成图片点阵'''
    # 转变训练集
    im_tr = np.array(data_dict['images_train'])
    im_tr = np.reshape(im_tr, (-1, 3, 32, 32))
    # 进行轴变换，因为图片点阵是（行，列，通道）
    # 数据集（sample，行， 列， 通道）
    im_tr = np.transpose(im_tr, (0, 2, 3, 1))
    data_dict['images_train'] = im_tr
    # 转变测试集
    im_te = np.array(data_dict['images_test'])
    im_te = np.reshape(im_te, (-1, 3, 32, 32))
    im_te = np.transpose(im_te, (0, 2, 3, 1))
    data_dict['images_test'] = im_te
    return data_dict


def gen_batch(data, batch_size, num_iter):
    '''随机生成num_iter个大小为batch_size的batch。'''
    data = np.array(data)
    data_size = len(data)
    for i in random.sample(range(data_size // batch_size), num_iter):
        index = i * batch_size
        if (index + batch_size > data_size):
            print(u'生成batch出现错误！')
            index = 0
        shuffled_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffled_indices]
        yield data[index: index + batch_size]

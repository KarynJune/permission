# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    '''输入数据到模型'''
    model = Sequential([
        Dense(32, input_shape=(100,)),  # 输入尺寸为(*,100)的数组，输出尺寸为(*,32)
        Activation('relu'),  # 应用激活函数‘relu’到输出
        Dense(1),  # 输出尺寸为(*,1)，不是第一层不需要输入数据
        Activation('softmax'),  # 应用激活函数‘softmax’到输出
    ])
    '''配置学习过程，编译'''
    model.compile(
        optimizer='rmsprop',  # 优化器
        loss='binary_crossentropy',  # 损失函数，这里以二分类为例
        metrics=['accuracy']  # 评估标准，分类问题都可以用‘accuracy’
    )
    '''模拟数据'''
    data = np.random.random((1000, 100))  # 1000行100列的二维数组，这里的列应与前面输入的一致
    labels = np.random.randint(2, size=(1000, 1))  # 这里的列应与前面输出的一致，2类分类，目标数据
    '''训练'''
    model.fit(data, labels, epochs=10, batch_size=32)  # 以32个样本迭代10次
    '''测试'''
    test_data = np.random.random((100, 100))
    test_labels = np.random.randint(2, size=(100, 1))
    score = model.evaluate(test_data, test_labels, batch_size=32)
    print(score)



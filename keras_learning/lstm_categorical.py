# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import re
import nltk
from sklearn.model_selection import train_test_split
import collections
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential, load_model, model_from_yaml
from keras.layers import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略没有avx2的警告

from datas import databases
from keras_learning import service

SENTENCE_LENGTH = 40  # 每条数据最大长度（根据实际数据调整,这里取大于平均值）
BATCH_SIZE = 256  # 每次迭代大小（根据训练数据时准确率调整,不宜过小也不宜过大）
EPOCH = 30  # 迭代次数（根据训练数据时准确率调整）


def get_data_info(datas):
    """
    获取数据信息
    :param datas:
    :return:
        freq_list：词频列表
    """
    words = []
    words_len_info = []
    labels = []
    for data in datas:
        # print "/".join(data[0])
        words += data[0]
        labels.append(data[1])
        words_len_info.append(len(data[0]))

    print collections.Counter(labels)
    data_len = len(datas)  # 数据长度
    word_avg_len = np.mean(words_len_info)  # 数据平均词数
    word_max_len = max(words_len_info)  # 数据最大词数
    freq_list = collections.Counter(words)
    word_len = len(freq_list)  # 不同单词数

    print("data length:%d" % (data_len))
    print("word length:%d,word avg length:%.0f,word max length:%d" % (word_len, word_avg_len, word_max_len))
    print("-----------------------------------------------------------------above data info")
    return freq_list


def convert_sequence(datas, word2index):
    """
    转换序列到实际数据
    :param datas:
    :param word2index:
    :return:X：序列数据, y：标签
    """
    data_len = len(datas)
    X = np.empty(data_len, dtype=list)  # 样本数据：[None None None ... None None None]
    y = np.zeros(data_len)  # 情感标签：[0. 0. 0. ... 0. 0. 0.]
    for index, data in enumerate(datas):
        words, label = data[0], data[1]
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[index] = seqs
        y[index] = int(label)
    X = sequence.pad_sequences(X, maxlen=SENTENCE_LENGTH)
    print(X.shape)  # (1023, 20)
    print(X)
    print(y)
    print("-----------------------------------------------------------------above data sequence")
    return X, y


def create_lstm(freq_list):
    """构建lstm网络"""
    EMBEDDING_SIZE = 200
    HIDDEN_LAYER_SIZE = 4

    model = Sequential()
    model.add(Embedding(len(freq_list), EMBEDDING_SIZE, input_length=SENTENCE_LENGTH))
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(5))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


if __name__ == '__main__':
    '''准备数据'''
    datas = databases.get_data_from_sort()
    freq_list = get_data_info(datas)
    word2index, index2word = service.lookup_table(freq_list, 1000)
    X, y = convert_sequence(datas, word2index)
    y_dict = {2: 0, 3: 1, 4: 2, 9: 3, 10: 4}
    y = [y_dict[yy] for yy in y]  # label需要下标0开始标注
    print collections.Counter(y)
    y = to_categorical(y, num_classes=5)  # 要使用categorical_crossentropy，需要把labels转换成二进制向量
    print y.shape
    """划分数据"""
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=1)

    model = create_lstm(freq_list)
    """训练"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
    model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(Xtest, ytest),
              callbacks=[early_stopping])

    """验证"""
    score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    print("score:%.2f,acc:%.2f" % (score, acc))

    ypred = model.predict(Xtest)
    for index, pred in enumerate(ypred):
        pred = list(pred)
        words = [index2word[x] for x in Xtest[index] if x != 0]
        print np.argwhere(pred == np.max(pred)), " :", "/".join(words)

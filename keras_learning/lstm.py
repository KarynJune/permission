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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略没有avx2的警告

from datas import databases
from keras_learning import service


SENTENCE_LENGTH = 40  # 每条数据最大长度（根据实际数据调整,这里取大于平均值）
BATCH_SIZE = 32  # 每次迭代大小（根据训练数据时准确率调整,不宜过小也不宜过大）
EPOCH = 5  # 迭代次数（根据训练数据时准确率调整）


def get_data_info(datas):
    """
    获取数据信息
    :param datas:
    :return:
        freq_list：词频列表
    """
    words = []
    words_len_info = []
    neg_len = 0  # 消极数量
    pos_len = 0  # 积极数量
    for data in datas:
        words += data[0]
        words_len_info.append(len(data[0]))
        if data[1] == 0:
            neg_len += 1
        else:
            pos_len += 1
    data_len = len(datas)  # 数据长度
    word_avg_len = np.mean(words_len_info)  # 数据平均词数
    word_max_len = max(words_len_info)  # 数据最大词数
    freq_list = collections.Counter(words)
    word_len = len(freq_list)  # 不同单词数

    # print(np.array(datas))
    print neg_len
    print("data length:%d,Positive length:%d,Negative length:%d" % (data_len, pos_len, neg_len))
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


if __name__ == '__main__':
    '''准备数据'''
    datas = databases.get_data_from_sql()
    freq_list = get_data_info(datas)
    word2index, index2word = service.lookup_table(freq_list)
    X, y = convert_sequence(datas, word2index)
    """划分数据"""
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=1)
    # Xtrain, ytrain = X, y
    # 更换测试数据
    # test_datas = databases.get_data_self()
    # Xtest, ytest = convert_sequence(test_datas, word2index)

    """构建网络"""
    EMBEDDING_SIZE = 128
    HIDDEN_LAYER_SIZE = 64

    model = Sequential()
    model.add(Embedding(len(freq_list), EMBEDDING_SIZE, input_length=SENTENCE_LENGTH))
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    """训练"""
    model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(Xtest, ytest))
    # model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=EPOCH)

    """验证"""
    score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    print("score:%.2f,acc:%.2f" % (score, acc))
    # ypred = model.predict(Xtest1)
    # for index, pred in enumerate(ypred):
    #     print(pred[0], " :", [index2word[x] for x in Xtest1[index] if x != 0])
    model.save(os.path.dirname(os.path.realpath(__file__))+'/lstm_model.h5'.encode("utf-8"))

    # model.save_weights('lstm_model.h5'.encode("utf-8"))
    # model.save('lstm_model.hdf5'.encode("utf-8"))
    #
    # json_string = model.to_json()
    # open('lstm_model.json'.encode("utf-8"), 'w').write(json_string)

    del model  # 删除已经存在的模型
    # model = load_model('lstm_model.h5'.encode("utf-8"))
    # ypred = model.predict(Xtest)
    # for index, pred in enumerate(ypred):
    #     print(pred[0], " :", [index2word[x] for x in Xtest[index] if x != 0])



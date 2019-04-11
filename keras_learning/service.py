# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from sklearn.model_selection import train_test_split
import collections
import numpy as np
import os
import commands
import nltk
from nltk.util import bigrams, trigrams, skipgrams
from keras.preprocessing import sequence
from keras.models import Sequential, load_model, model_from_yaml
from keras.backend import clear_session

from datas import databases, cut_word

SENTENCE_LENGTH = 40  # 每条数据最大长度（根据实际数据调整，这里取大于平均值）


def get_data_info(datas):
    """
    获取数据信息
    :param datas:
    :return:
        freq_list：词频列表
    """
    words = []
    for data in datas:
        words += data[0]
    freq_list = collections.Counter(words)

    return freq_list


def lookup_table(freq_list, cut_size=720):
    """
    建立单词序列配置表
    :param freq_list:
    :return:
        word2index:{'的': 2, '了': 3, '我': 4,....., '帐关': 3700, '还卡顿': 3701, 'PAD': 0, 'UNK': 1}
        index2word:{2: '的', 3: '了', 4: '我'....., 3700: '帐关', 3701: '还卡顿', 0: 'PAD', 1: 'UNK'}
    """
    MAX_FEATURES = len(freq_list) - cut_size  # 去掉词频低的部分数据
    word2index = {x[0]: i + 2 for i, x in enumerate(freq_list.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v: k for k, v in word2index.items()}
    return word2index, index2word


def convert_sequence(datas, word2index):
    """
    转换序列到实际数据
    :param datas:
    :param word2index:
    :return:
    """
    data_len = len(datas)
    sequence_data = np.empty(data_len, dtype=list)  # 样本数据：[None None None ... None None None]
    for index, words in enumerate(datas):
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        sequence_data[index] = seqs
    sequence_data = sequence.pad_sequences(sequence_data, maxlen=SENTENCE_LENGTH)
    return sequence_data


def predict_data(datas):
    '''获取序列'''
    db_datas = databases.get_data_from_filter()
    word2index, index2word = lookup_table(get_data_info(db_datas))
    '''预测数据'''
    datas = [cut_word.CutWord(data, 'zh').cut() for data in datas]  # 分词：[[word1,word2],[word1,word2]...]
    sequence_data = convert_sequence(datas, word2index)  # 转换序列

    clear_session()  # 使用tensorflow作为后端时防止并发需要清除，cntk不用

    model = load_model(os.path.dirname(os.path.realpath(__file__)) + '/lstm_model.h5'.encode("utf-8"))

    y = model.predict(sequence_data)
    # return y
    pred_list = []
    word_list, pos_list, neg_list = [], [], []
    for index, pred in enumerate(y):
        words = [index2word[x] for x in sequence_data[index] if x != 0 and x != 1]
        if words:words.append("/")
        word_list += words
        if pred >= 0.7:
            pos_list += words
        if pred < 0.3:
            neg_list += words
        pred_list.append({"pred": pred[0], "words": "/".join(words)})
        # print(pred[0], " :", [index2word[x] for x in sequence_data[index] if x != 0])
    return pred_list, freq_handle(pos_list), freq_handle(neg_list)


def retrain():
    str = "python keras_learning/lstm.py".encode("utf-8")
    status, output = commands.getstatusoutput(str)
    return status, output


def freq_handle(word_list):
    # freq_list = collections.Counter(word_list)
    # freq_list = [{"word": word, "count": count} for word,count in freq_list.items() if len(word)>1]
    # freq_list = sorted(freq_list, key=lambda x: x['count'], reverse=True)[:10]

    # text = nltk.Text(word_list)
    # double_words = bigrams(word_list)
    double_words = skipgrams(word_list, 2, 1)
    double_word_dict = {}
    for dw in double_words:
        key = dw[0] + " " + dw[1]
        if "/" not in key and dw[0]!=dw[1]:
            try:
                double_word_dict[key] += 1
            except KeyError:
                double_word_dict[key] = 1
    # double_word_list = [{"word": word, "count": count} for word, count in double_word_dict.items() if count > 1]+freq_list
    double_word_list = [{"word": word, "count": count} for word, count in double_word_dict.items() if count > 1]
    double_word_list = sorted(double_word_list, key=lambda x: x['count'], reverse=True)

    return double_word_list

# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.test import TestCase

# Include libraries.
import paddle
import paddle.fluid as fluid
from functools import partial

import os
import re
import MySQLdb

import collections
import numpy as np
import tarfile
import string
import six


import cut_word

CLASS_DIM = 2
EMB_DIM = 128
HID_DIM = 512
STACKED_NUM = 3
BATCH_SIZE = 128
USE_GPU = False


def get_data_from_sql():
    """获取数据"""
    mysql_cn = MySQLdb.connect(host='10.250.40.99', port=3306, user='root', passwd='88888888', db='zjy_test')
    sql = "SELECT content,sort FROM language_filterdata where sort!=0 and sort!=3"
    cursor = mysql_cn.cursor()
    cursor.execute(sql)
    alldata = cursor.fetchall()
    cursor.close()
    mysql_cn.close()
    r = '{机器型号:[\s\S]*?}|回复：[\s\S]*'
    alldata = [(cut_word.CutWord(re.sub(r, '', data[0].decode("utf-8")), 'zh').cut(), data[1]) for data in alldata]
    return alldata


def build_dict(datas):
    """
    Build a word dictionary from the corpus. Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    word_freq = collections.defaultdict(int)
    for data in datas:
        for word in data[0]:
            word_freq[word] += 1

    words, _ = list(zip(*word_freq.items()))
    word_idx = dict(list(zip(words, range(len(words)))))
    word_idx['<unk>'] = len(words)
    return word_idx


def train_reader(word_dict, datas):
    UNK = word_dict['<unk>']

    def reader():
        for data in datas:
            words = [word_dict.get(w, UNK) for w in data[0]]
            label = 0
            if data[1] > 1:
                label = 1
            yield words, label

    return reader


def convolution_net(data, input_dim, class_dim, emb_dim, hid_dim):
    emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)
    conv_3 = fluid.nets.sequence_conv_pool(input=emb, num_filters=hid_dim, filter_size=5, act="tanh", pool_type="sqrt")
    conv_4 = fluid.nets.sequence_conv_pool(input=emb, num_filters=hid_dim, filter_size=5, act="tanh", pool_type="sqrt")
    prediction = fluid.layers.fc(input=[conv_3, conv_4], size=class_dim, act="softmax")
    return prediction


def stacked_lstm_net(data, input_dim, class_dim, emb_dim, hid_dim, stacked_num):
    emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)

    fc1 = fluid.layers.fc(input=emb, size=hid_dim)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    prediction = fluid.layers.fc(input=[fc_last, lstm_last], size=class_dim, act='softmax')
    return prediction


def inference_program(word_dict):
    data = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)

    dict_dim = len(word_dict)
    net = convolution_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM)
    # net = stacked_lstm_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM, STACKED_NUM)
    return net


def train_program(word_dict):
    prediction = inference_program(word_dict)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost, accuracy]


def optimizer_func():
    return fluid.optimizer.Adagrad(learning_rate=0.002)


params_dirname = "understand_sentiment_conv.inference.model"


def event_handler(event):
    if isinstance(event, fluid.EndStepEvent):
        print "Step {0}, Epoch {1} Metrics {2}".format(event.step, event.epoch, map(np.array, event.metrics))

        if event.step == 10:
            trainer.save_params(params_dirname)
            trainer.stop()


if __name__ == '__main__':
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    print "Loading IMDB word dict...."
    # word_dict = paddle.dataset.imdb.word_dict()

    datas = get_data_from_sql()
    word_dict = build_dict(datas)
    print "Reading training data...."
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            train_reader(word_dict, datas), buf_size=25000), batch_size=BATCH_SIZE)

    print "create trainer...."
    trainer = fluid.Trainer(train_func=partial(train_program, word_dict), place=place, optimizer_func=optimizer_func)

    feed_order = ['words', 'label']

    print "train start...."
    trainer.train(num_epochs=1, event_handler=event_handler, reader=train_reader, feed_order=feed_order)

    inferencer = fluid.Inferencer(
        infer_func=partial(inference_program, word_dict), param_path=params_dirname, place=place)

    reviews_str = ['因为明日之后，所以我要把所有网易游戏都评一星', '垃圾游戏，不要问我为什么', '真的很棒的手游?下了卸卸了下，终于固定下来了，加油加油']
    reviews = [cut_word.CutWord(c,'zh').cut() for c in reviews_str]

    UNK = word_dict['<unk>']
    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, UNK) for words in c])

    base_shape = [[len(c) for c in lod]]

    tensor_words = fluid.create_lod_tensor(lod, base_shape, place)

    results = inferencer.infer({'words': tensor_words})

    for i, r in enumerate(results[0]):
        print reviews_str[i], ",positive:", r[0], " , negative:", r[1]



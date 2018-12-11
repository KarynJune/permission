# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.test import TestCase

import re
import MySQLdb
import collections
import paddle
import paddle.fluid as fluid
from functools import partial
import numpy as np

from paddle.fluid.trainer import *
from paddle.fluid.inferencer import *

import cut_word


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


def train_reader_self(word_dict, datas):
    UNK = word_dict['<unk>']

    def reader():
        for data in datas:
            words = [word_dict.get(w, UNK) for w in data[0]]
            label = 0
            if data[1] > 1:
                label = 1
            yield words, label

    return reader


CLASS_DIM = 2
EMB_DIM = 128
HID_DIM = 512
BATCH_SIZE = 128


def convolution_net(data, input_dim, class_dim, emb_dim, hid_dim):
    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        act="tanh",
        pool_type="sqrt")
    conv_4 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=4,
        act="tanh",
        pool_type="sqrt")
    prediction = fluid.layers.fc(
        input=[conv_3, conv_4], size=class_dim, act="softmax")
    return prediction


def inference_program(word_dict):
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    dict_dim = len(word_dict)
    net = convolution_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM)
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


def train(use_cuda, train_program, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    print("Loading IMDB word dict....")
    word_dict = paddle.dataset.imdb.word_dict()

    print("Reading training data....")

    print paddle.dataset.imdb.train(word_dict)
    """
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=25000),
        batch_size=BATCH_SIZE)
    
    print("Reading testing data....")
    test_reader = paddle.batch(
        paddle.dataset.imdb.test(word_dict), batch_size=BATCH_SIZE)
    
    trainer = Trainer(
        train_func=partial(train_program, word_dict),
        place=place,
        optimizer_func=optimizer_func)

    feed_order = ['words', 'label']

    def event_handler(event):
        if isinstance(event, EndStepEvent):
            if event.step % 10 == 0:
                avg_cost, acc = trainer.test(
                    reader=test_reader, feed_order=feed_order)

                print('Step {0}, Test Loss {1:0.2}, Acc {2:0.2}'.format(
                    event.step, avg_cost, acc))

                print("Step {0}, Epoch {1} Metrics {2}".format(
                    event.step, event.epoch, list(map(np.array,
                                                      event.metrics))))

        elif isinstance(event, EndEpochEvent):
            trainer.save_params(params_dirname)

    trainer.train(
        num_epochs=1,
        event_handler=event_handler,
        reader=train_reader,
        feed_order=feed_order)
"""

def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    word_dict = paddle.dataset.imdb.word_dict()

    inferencer = Inferencer(
        infer_func=partial(inference_program, word_dict),
        param_path=params_dirname,
        place=place)

    reviews_str = [
        'read the book forget the movie', 'this is a great movie',
        'this is very bad'
    ]
    reviews = [c.split() for c in reviews_str]

    UNK = word_dict['<unk>']
    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, UNK) for words in c])

    base_shape = [[len(c) for c in lod]]

    tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
    results = inferencer.infer({'words': tensor_words})

    for i, r in enumerate(results[0]):
        print("Predict probability of ", r[0], " to be positive and ", r[1],
              " to be negative for review \'", reviews_str[i], "\'")


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "understand_sentiment_conv.inference.model"
    train(use_cuda, train_program, params_dirname)
    # infer(use_cuda, inference_program, params_dirname)


if __name__ == '__main__':
    use_cuda = False  # set to True if training with GPU
    main(use_cuda)



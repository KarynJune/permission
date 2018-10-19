# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.test import TestCase

import paddle
import paddle.fluid as fluid
import paddle.fluid.contrib.trainer as trainer
import paddle.fluid.contrib.inferencer as inferencer
from functools import partial
import numpy as np
import codecs
import pandas as pd
from snownlp import SnowNLP
import os
# import jieba
# jieba.load_userdict(os.path.dirname(os.path.realpath(__file__)) + "/dicts/game_dict.txt")
# jieba.load_userdict(os.path.dirname(os.path.realpath(__file__)) + "/dicts/yys_dict.txt")


# Create your tests here.

CLASS_DIM = 2
EMB_DIM = 128
HID_DIM = 512
STACKED_NUM = 3
BATCH_SIZE = 128
USE_GPU = False

good_texts = [
    u"这个东西真心很赞",
    u"很好玩，但是我手残，还是不太习惯，网络有问题，经常999延迟",
    u"画风什么的确实都是偏黑暗向，不过玩多了就知道其实只是音乐有些诡异",
    u"个人觉得还不错，但是我真的好好奇侦探的剧情啊",
    u"这也是一款很不错的游戏啦游戏体验还是蛮不错的，也希望更多人能喜欢",
    u"游戏虽好，充钱太假",
    u"不能玩了;iPad更新之后就不能玩了，一进去就闪退，满级号白白没了，希望官方能看一下，不过游戏挺好玩的。",
    u"很棒;一点也不恐怖，很好只是有的服装好贵，琼楼遗恨，兰 闺惊梦好贵，红蝶可以弱点吗？其它都excellent",
]

bad_texts = [
    u"总体，并不推荐，抖m才玩排位赛监管者，优化差，手感差，画质……流畅度差，平衡性差",
    u"我是真的找不到游戏乐趣没错我很菜，因为这游戏没有给我一丁点练技术的欲望，这是我的真实感受",
    u"下载之后打不开;更新失败，然后就彻底打不开了，，，本来想试一下的。。还看到有专门的修复客户端，然而什么都点不动",
    u"不怎么样;补丁慢的你能等到猴年马月",
    u"有些bug一直不修真是烦;比如翻一次窗就不能翻第二次 小丑隔窗拉锯 杰克隔墙打人 这都合理么",
    u"匹配一小时、游戏五分钟;、玩这游戏大部分时间都在等待。匹配机制太差劲！改成聊天工具好了、简直可笑！",
    u"抄袭黎明杀机，实锤;如题",
]


def snowNLP(text):
    """SnowNLP情感分析"""
    return SnowNLP(text).sentiments


def get_stopwords():
    filepath = os.path.dirname(os.path.realpath(__file__)) + "/dicts/stop_words.txt"
    stopwords = [line.strip() for line in codecs.open(filepath, 'r', 'utf-8').readlines()]
    return stopwords


# def cut_word(text):
#     """分词"""
#     results = []
#     words = jieba.cut(text)
#     stopwords = get_stopwords()
#     for word in words:
#         if word.strip() in stopwords:
#             continue
#         else:
#             results.append(word)
#     return " ".join(results)


def convolution_net(data, input_dim, class_dim, emb_dim, hid_dim):
    """文本卷积神经网络"""
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
    """预测程序"""
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

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
    """优化函数"""
    return fluid.optimizer.Adagrad(learning_rate=0.002)


params_dirname = "understand_sentiment_conv.inference.model"


def event_handler(event):
    """事件处理器:在每步训练结束后查看误差"""
    if isinstance(event, fluid.contrib.trainer.EndStepEvent):
        print("Step {0}, Epoch {1} Metrics {2}".format(
                event.step, event.epoch, list(map(np.array, event.metrics))))

        if event.step == 10:
            trainer.save_params(params_dirname)
            trainer.stop()


if __name__ == '__main__':
    print("start....")
    '''定义在cpu上训练'''
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    print("Loading IMDB word dict....")
    word_dict = paddle.dataset.imdb.word_dict()
    #
    # print ("Reading training data....")
    # train_reader = paddle.batch(
    #     paddle.reader.shuffle(
    #         paddle.dataset.imdb.train(word_dict), buf_size=25000),
    #     batch_size=BATCH_SIZE)
    # '''构造训练器'''
    # trainer = trainer.Trainer(
    #     train_func=partial(train_program, word_dict),
    #     place=place,
    #     optimizer_func=optimizer_func)
    #
    # feed_order = ['words', 'label']
    # print "daozheli "
    # trainer.train(
    #     num_epochs=1,
    #     event_handler=event_handler,
    #     reader=train_reader,
    #     feed_order=feed_order)
    print "训练完毕"
    inferencer = inferencer.Inferencer(
        infer_func=partial(inference_program, word_dict), param_path=params_dirname, place=place)
    reviews_str = [
        'read the book forget the movie', 'this is a great movie', 'this is very bad'
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
        print(
        "Predict probability of ", r[0], " to be positive and ", r[1], " to be negative for review \'", reviews_str[i],
        "\'")

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import re
import nltk
import gensim
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import collections
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential, load_model, model_from_yaml
from keras.layers import Dense, Activation, Dropout, Conv1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.layers.wrappers import Bidirectional
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略没有avx2的警告

from datas import databases
from keras_learning import service

NUM_CLASSES = 2  # 类别数量
EMBEDDING_DIM = 200  # 词语维度
SENTENCE_LENGTH = 100  # 每条数据的最大词语长度
BATCH_SIZE = 300  # 迭代大小
EPOCH = 50  # 迭代次数
HIDDEN_LAYER_SIZE = 64  # 隐藏层数

word_model = gensim.models.Word2Vec.load(BASE_DIR + "/nlp/taptap_200d.model".encode("utf-8"))


def gen_vec_matrix():
    """生成向量矩阵"""
    print "generate embedding vector..."
    word2index = {w: i + 1 for i, w in enumerate(word_model.wv.vocab.keys())}

    embedding_matrix = np.zeros((len(word2index) + 1, EMBEDDING_DIM))
    for w, i in word2index.items():
        embedding_matrix[i] = word_model[w]

    print "generate end."
    return word2index, embedding_matrix


def prepare_data(datas, word2index):
    """预处理数据"""
    print "preparing data..."
    X, y = [], []
    for i, d in enumerate(datas):
        if d[1] in [3,4]:
            X.append([word2index.get(word, 0) for word in d[0]])
            y.append(0 if d[1]==3 else 1)
        # X.append([word2index.get(word, 0) for word in d[0]])
        # y.append(d[1])
    X = sequence.pad_sequences(np.array(X), maxlen=SENTENCE_LENGTH)
    y = np.array(y)
    return X, y


def lstm_model(input_dim, embedding_matrix):
    """LSTM"""
    print "build LSTM model..."
    model = Sequential()
    model.add(Embedding(input_dim, EMBEDDING_DIM, weights=[embedding_matrix], input_length=SENTENCE_LENGTH,
                        trainable=False))
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


def cnn_model(input_dim, embedding_matrix):
    """CNN"""
    print "build CNN model..."
    model = Sequential()
    model.add(Embedding(input_dim, EMBEDDING_DIM, weights=[embedding_matrix], input_length=SENTENCE_LENGTH,
                        trainable=False))
    model.add(Conv1D(SENTENCE_LENGTH, NUM_CLASSES, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())  # 池化
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.add(Dropout(0.2))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def tain():
    """训练数据"""
    print "train start..."
    word2index, embedding_matrix = gen_vec_matrix()
    X, y = prepare_data(databases.get_data_from_sort(), word2index)
    y = to_categorical(y, num_classes=NUM_CLASSES)
    X, y = shuffle(X, y)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)

    model = lstm_model(len(word2index)+1, embedding_matrix)
    # model = cnn_model(len(word2index)+1, embedding_matrix)

    """训练"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
    model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(Xtest, ytest), callbacks=[early_stopping])
    score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    print("score:%.2f,acc:%.2f" % (score, acc))

    model.save(os.path.dirname(os.path.realpath(__file__)) + '/lstm_w2d_model.h5'.encode("utf-8"))


def test():
    """测试"""
    print "test start..."
    test_datas = [
        (u"不怎么好玩，一款末日生存游戏，每天做着重复的事，感觉受不了",10),
        (u"知道为什么评分低吗？玩了半个月我卸载了。",10),
        (u"6个月前 我做了一件特大的错事 我上了网易的黑车 尽管好心的僵尸怎么追喊让我下来不要上当 我都没有听他们的 还拿枪打死了他们 我在秋日冒着大雨砍柴维持生活的时候 一只可爱的丧尸发现是我 他双手抓着我的肩膀对我大喊快下车！快下车！ 但是我并没有领情 还推倒了他…",10),
        (u"僵尸：活下去，一起！ 帝国士兵：活下去，一起！ 特殊变异体：活下去一起！ 众反派：活下去，一起！！！ （狗头保命）",3),
        (u"有点懵逼",10),
        (u"画面模型非常喜欢给两星，完美抄袭了王者荣耀的460，甚至青出于蓝胜于蓝在Wifi的时候也会出现，高画质下有点卡，给个忠告，先把优化做好再想着赚钱",10),
        (u"游戏不错，立意很新颖，平衡性还行，符文系统没有王者荣耀那么变态，角色技能也不是全抄，值得鼓励",10),
        (u"250……整局都在250",10),
        (u"掉帧严重",10),
        (u"不太恐怖的恐怖游戏 emmmmm还有点卡",10),
        (u"屠夫玩家的评价（就不能打零颗星吗？难受。。。。。。）",10),
        (u"我觉得一星就好了，不要骄傲哟",10),
    ]
    word2index, embedding_matrix = gen_vec_matrix()
    model = load_model('lstm_w2d_model.h5'.encode("utf-8"))
    X, y = prepare_data(databases.handle_datas2(test_datas), word2index)
    ypred = model.predict(X)
    for index, pred in enumerate(ypred):
        print pred[0], " :", test_datas[index][0]


if __name__ == '__main__':
    # tain()
    test()


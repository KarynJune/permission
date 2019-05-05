# -*- coding: utf-8 -*-

import numpy
import nltk
import gensim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
plt.figure(figsize=(8*3.13,6*3.13))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from datas.databases import get_raw_data, write_file

"""
词向量

"""


def word2vec_save():
    datas = get_raw_data('g66')+get_raw_data('g37')+get_raw_data('h55')
    train_list = [d[0] for d in datas]  # [["氪金","真的","不好","坚持","下来"],["很","好玩"],...]
    model = gensim.models.Word2Vec(train_list, size=200, iter=100)
    model.save("taptap_200d.model")


def word2vec_load():
    model = gensim.models.Word2Vec.load("taptap_200d.model")
    for i in model.most_similar(u"不错", topn=20):
        print i[0], i[1]

    # X = model[model.wv.vocab]
    # pca = PCA(n_components=2)
    # result = pca.fit_transform(X)
    #
    # plt.scatter(result[:,0],result[:,1])
    # words = list(model.wv.vocab)
    # for i, word in enumerate(words):
    #     # print "{name:'"+word+"',x:",result[i, 0],",y:", result[i, 1],"},"
    #     plt.annotate(word, xy=[result[i, 0], result[i, 1]])
    # plt.show()


if __name__ == '__main__':
    word2vec_load()


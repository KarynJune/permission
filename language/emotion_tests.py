# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.test import TestCase

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import SklearnClassifier


import re
from time import time
import codecs
import pandas as pd
import MySQLdb
import numpy as np
from snownlp import SnowNLP
import os
import jieba
jieba.load_userdict(os.path.dirname(os.path.realpath(__file__)) + "/dicts/game_dict.txt")
jieba.load_userdict(os.path.dirname(os.path.realpath(__file__)) + "/dicts/yys_dict.txt")


# Create your tests here.

def get_data_from_sql():
    """获取数据"""
    mysql_cn = MySQLdb.connect(host='192.168.42.112', port=3306, user='unimonitor', passwd='unimonitor', db='g37')
    sql = "SELECT subject,comment_count FROM weibo where source like 'taptap%' and comment_count!=0 and length(subject)<200 "
    cursor = mysql_cn.cursor()
    cursor.execute(sql)
    alldata = cursor.fetchall()
    cursor.close()
    mysql_cn.close()
    return alldata


def get_data_from_zjy():
    """获取数据"""
    mysql_cn = MySQLdb.connect(host='10.250.40.99', port=3306, user='root', passwd='88888888', db='zjy_test')
    sql = "SELECT content,sort FROM language_filterdata where sort!=0"
    cursor = mysql_cn.cursor()
    cursor.execute(sql)
    alldata = cursor.fetchall()
    cursor.close()
    mysql_cn.close()
    return alldata


def get_data_self():
    datas = (
        ('吃相难看   甚至简直赤裸裸的骗钱', 1),
        ('我之前打了五星，现在回来改成一星，⑧好意思，道歉了', 1),
        ('不觉得最近的二次元骗钱渣作垃圾卡游越来越多了吗，还全是预约期间分数虚高', 1),
        ('辣鸡游戏，吃相难看，告辞', 1),
        ('感觉这游戏体验很差', 1),
        ('这游戏的舞姬抄的《黑暗之魂》的舞娘，还有《血源诅咒》的奶妈，咋了，当我们大剑莽夫死光了？？？抄袭可耻！抄袭可耻！抄袭可耻！', 1),
        ('除了美工没有看出半点有诚意的地方，要不是闪耀暖暖老是没出我也不会跑来玩这个，说休闲并不具有休闲的乐趣，说肝也没有肝的兴趣，定位及其模糊。', 1),
        ('尼玛，抄袭黑暗之魂的垃圾游戏，已有大神发对比视频给官方了，等着被告吧', 1),
        ('比较坑，就是一个单机养成，动作卡片。。', 1),
        ('游戏玩的很尴尬，台词也很尬', 1),
        ('抽卡几率简直令人发疯；体力，体力，体力', 1),
        ('无聊', 1),
        ('氪金游戏鉴定完毕，脸黑不玩了。。这游戏终于快给自己作死了。', 1),
        ('公测准时下载，满怀希望到失望，只玩了一天就卸载了，不好玩', 1),
        ('真的一无是处的游戏', 1),
        ('很喜欢游戏，但机制是真的恶心', 5),
        ('玩着还行啊，心疼客服哈哈哈哈', 5),
        ('挺不错的', 5),
        ('球球不要再给我拓印了啊啊啊啊啊！我要卡！！我要ssr！！！', 5),
        ('游戏感觉不错，就是ssr妖灵阶级有点氪不动，人物和妖灵的故事很好玩，适合长期养成。', 5),
        ('还不错啦！！！！看脸，总能慢慢都有的', 5),
        ('被吸引的点主要是剧情了，剧情真的做的非常的走心和出彩了。妖怪的画风和形象是也特别合理的。', 5),
        ('战斗系统好评，可以好友助战来对抗妖灵，身为一个菜鸡只能厚脸皮去加好友助战了，爽哈哈哈哈', 5),
        ('熙儿在玩的游戏肯定好评', 5),
        ('这游戏很好玩，画面精致，可以捏脸，风景美，动作酷炫，剧情引人入胜，还有好多小姐姐小哥哥互动，说话又好听，超喜欢玩这个游戏', 5),
        ('好玩﻿|｡･㉨･)っ♡，除了我非，，，没有太大的缺点，非可能是我自己的缺点T_T，提个小意见，鉴宝会在蓝色或者别的高级颜色的未知宝藏里鉴别出最低级的东西。。。', 5),
        ('除了没有楚留香那种打的舒服一点。其他都非常满意。', 5),
        ('虽然有很多不足，但是画质，文案都很棒。', 5),
        ('这个捏脸系统真的是怎么捏都好看啊啊啊！我女儿简直盛世美颜', 5),
        ('感觉游戏挺好的，里面的人物做的很好，画质也不错，就是玩法单一，而且还不能跨服加好友，这就不好。', 5),
    )
    return datas


def handle_datas(datas):
    """处理数据"""
    r = '{机器型号:[\s\S]*?}|回复：[\s\S]*'
    results = []
    for data in datas:
        star = 1
        if int(data[1]) > 3:
            star = 2
        elif int(data[1]) < 3:
            star = 0
        content = data[0]
        if isinstance(content, str):
            content = content.decode('utf-8')
        content = re.sub(r, '', content)
        if len(content) < 500:
            results.append((content, star))
    return results


def get_filewords(file_name):
    """获取停用词"""
    filepath = os.path.dirname(os.path.realpath(__file__)) + "/dicts/"+file_name
    words = [line.strip() for line in codecs.open(filepath, 'r', 'utf-8').readlines()]
    return words


def cut_word(text):
    """分词"""
    results = []
    words = jieba.cut(text)
    denywords = get_filewords("deny_dict.txt")
    stopwords = get_filewords("stop_words.txt")
    for word in words:
        word = re.sub(r'^[0-9]*$', '', word)
        word = word.strip()
        if word in denywords or word not in stopwords:
            results.append(word)
    # print "------------------------------------------"
    # print " ".join(results)
    return " ".join(results)


def get_features(datas):
    """获取特征集"""
    features = []
    word_list = []
    for data in datas:
        cut_words = cut_word(data[0]).split()
        # cut_words = tuple(bigram_words(cut_words))
        word_list += cut_words

        features.append({"content": data[0], "cut_word":" ".join(cut_words), "star": data[1]})

    # freqs = cal_freq(word_list)
    # words = freqs.items()
    # words = sorted(freqs.items(), key=lambda d: d[1], reverse=True)
    # idf_list = []
    # for w in set(word_list):
    #     tf_idf = cal_tfidf(w, word_list)
    #     if tf_idf < 0.1 and tf_idf > 0.01:
    #         idf_list.append((w, tf_idf))
    return features


def split_datas(datas):
    """拆分数据"""
    df = pd.DataFrame(datas)
    X = df[['cut_word', 'content']]
    y = df['star']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print X_train.head()
    return X_train, X_test, y_train, y_test


def matrix_datas(X):
    """转换为向量矩阵"""
    vect = CountVectorizer(max_df=0.1, min_df=0.01, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b')
    # vect = TfidfVectorizer()
    vect_array = vect.fit_transform(X.cut_word).toarray()
    term_matrix = pd.DataFrame(vect_array, columns=vect.get_feature_names())
    print term_matrix.items
    return term_matrix, vect


def cal_tfidf(word, word_list):
    """计算tf-idf"""
    text_collection = nltk.text.TextCollection(word_list)
    return text_collection.tf_idf(word, word_list)


def cal_freq(word_list):
    """统计词频"""
    freq_dist = nltk.probability.FreqDist(word_list)
    return freq_dist


def xlsx_to_csv_pd():
    filepath = os.path.dirname(os.path.realpath(__file__)) + "/dicts/"
    data_xls = pd.read_excel(filepath+'datas.xlsx', index_col=0)
    data_xls.to_csv(filepath+'datas.csv', encoding='utf-8')


def snowNLP(text):
    """SnowNLP情感分析"""
    return SnowNLP(text).sentiments


def split_snownlp(x):
    if x>0.7:
        return 2
    elif x<0.3:
        return 0
    else:
        return 1


def _transfer(x):
    result = 1
    x = int(x)
    if x>=4:
        result = 2
    elif x<=2:
        result = 0
    return result


def make_label(df):
    """归类:2积极,1中性，0消极"""
    r = '{机器型号:[\s\S]*?}|回复：'
    df["content"] = df["content"].apply(lambda x: re.sub(r, '', x))
    df["sentiment"] = df["star"].apply(_transfer)


def test():
    '''读取数据'''
    filepath = os.path.dirname(os.path.realpath(__file__)) + "/dicts/datas.csv"
    df = pd.read_csv(filepath, encoding='utf-8', error_bad_lines=False, names=['content', 'star'])
    '''分类'''
    make_label(df)
    print df.head()
    print df.shape
    '''分词'''
    X = df[['content']].copy()  # 特征
    y = df.sentiment
    X['cut_content'] = X.content.apply(cut_word)
    print X.head()
    '''拆分数据'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print X_train.shape
    print y_train.shape
    '''向量化语句 '''
    max_df = 0.8  # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
    min_df = 3  # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
    vect = CountVectorizer(max_df=max_df, min_df=min_df, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', )
    print X_train.cut_content
    term_matrix = pd.DataFrame(vect.fit_transform(X_train.cut_content).toarray(), columns=vect.get_feature_names())
    print term_matrix.head()
    '''朴素贝叶斯分类'''
    nb = MultinomialNB()
    '''管道连接'''
    pipe = make_pipeline(vect, nb)
    print pipe.steps
    '''交叉验证'''
    print cross_val_score(pipe, X_train.cut_content, y_train, cv=5, scoring='accuracy').mean()
    '''拟合模型'''
    pipe.fit(X_train.cut_content, y_train)
    '''测试'''
    y_pred = pipe.predict(X_test.cut_content)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)
    for doc, category in zip(X_test.content, y_pred):
        print doc, ":", category


def bag_of_words(words):
    return dict([(word, True) for word in words])


def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq):
    """卡方统计-当前词与双词搭配"""
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = []
    try:
        bigrams = bigram_finder.nbest(score_fn, 10)  # 返回分数高的前10个词组
    except ZeroDivisionError:
        pass
    return [word for word in words+bigrams]


def test1(X_train, X_test, y_train, y_test):
    """
    result:
    bayes:
    0.6024734982332155
    [[ 20  24  49]
     [ 14  57 111]
     [  7  20 264]]

    snownlp:
    0.4169611307420495
    [[ 45  24  24]
     [ 70  36  76]
     [ 69  67 155]]

    """
    term_matrix, vect = matrix_datas(X_train)

    '''连接特征标准化、分类器（朴素贝叶斯）'''
    pipe = make_pipeline(vect, MultinomialNB())
    print pipe.steps

    '''交叉验证：计算模型准确率均值'''
    print cross_val_score(pipe, X_train.cut_word, y_train).mean()
    '''拟合模型'''
    pipe.fit(X_train.cut_word, y_train)

    '''测试'''
    y_pred = pipe.predict(X_test.cut_word)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)
    # for doc, category in zip(X_test.content, y_pred):
    #     print doc, ":", category

    y_pred_snownlp = X_test.content.apply(snowNLP)
    y_pred_snownlp_normalized = y_pred_snownlp.apply(split_snownlp)
    print metrics.accuracy_score(y_test, y_pred_snownlp_normalized)
    print metrics.confusion_matrix(y_test, y_pred_snownlp_normalized)
    # for doc, category in zip(X_test.content, y_pred_snownlp_normalized):
    #     print doc, ":", category


def get_best_param(pipeline):
    """找到分类器最佳参数"""
    parameters = {
        'vect__max_df': (0.1, 0.8),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001),
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1)
    from time import time

    t0 = time()
    grid_search.fit(X_train.cut_word, y_train)
    print "done in %0.3fs" % (time() - t0)
    print "Best score: %0.3f" % grid_search.best_score_

    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])


def test2(X_train, X_test, y_train, y_test):
    """
    result:
    0.6413427561837456
    [[ 25  30  38]
     [  8  77  97]
     [  3  27 261]]
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    # get_best_param(pipeline)
    pipeline.set_params(clf__alpha=0.1, tfidf__use_idf=False, vect__max_df=0.1, vect__max_features=None)
    pipeline.fit(X_train.cut_word, y_train)
    y_pred = pipeline.predict(X_train.cut_word)
    print metrics.accuracy_score(y_train, y_pred)
    print metrics.confusion_matrix(y_train, y_pred)

    y_pred = pipeline.predict_proba(X_test.cut_word)  # 概率
    for doc, category in zip(X_test.content, y_pred):
        print doc, ":", category


if __name__ == '__main__':
    # test()
    # datas = get_data_from_zjy()
    datas = get_data_from_sql()
    # datas = get_data_self()
    datas = handle_datas(datas)
    features = get_features(datas)
    X_train, X_test, y_train, y_test = split_datas(features)
    # test1(X_train, X_test, y_train, y_test)
    test2(X_train, X_test, y_train, y_test)





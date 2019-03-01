# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.test import TestCase

import MySQLdb
import re
import os
import codecs
import numpy as np

import nltk
from nltk.util import bigrams, trigrams, ngrams, skipgrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import SklearnClassifier

import cut_word


# Create your tests here.
def get_data_from_sql():
    """获取数据"""
    mysql_cn = MySQLdb.connect(host='192.168.42.112', port=3306, user='unimonitor', passwd='unimonitor', db='g37')
    sql = "SELECT data_content FROM facebook where source='taptap_review' and " \
          "post_datetime>='2019-01-01' and post_datetime<'2019-02-01' order by post_datetime"
    cursor = mysql_cn.cursor()
    cursor.execute(sql)
    alldata = cursor.fetchall()
    cursor.close()
    mysql_cn.close()
    r = '{机器型号:[\s\S]*?}|回复：[\s\S]*|\n'
    alldata = [re.sub(r, '', data[0].decode("utf-8")) for data in alldata]
    return alldata


def get_stop_list(file_name):
    """获取停用词列表"""
    stop_list = []
    with codecs.open(os.path.dirname(os.path.realpath(__file__)) + "/dicts/"+file_name+".txt", 'r', encoding='utf8') as f:
        while True:
            line = f.readline().strip()  # 逐行读取
            if not line:
                break
            stop_list.append(line),
    return stop_list


def get_cut_words(comments):
    stop_words = get_stop_list("stop_words")
    positive_words = get_stop_list("positive_dict")
    negative_words = get_stop_list("negative_dict")
    words = []
    for comment in comments:
        print comment
        cut_words = cut_word.CutWord(comment, 'zh').cut()
        new_cut_words = [word for word in cut_words if word in positive_words or word in negative_words or word not in stop_words]
        words += new_cut_words
        print "-" * 50
        print "cut words:", "/".join(new_cut_words)
        print "=" * 100
    return words


def get_cut_words2(comment):
    stop_words = get_stop_list("stop_words")
    positive_words = get_stop_list("positive_dict")
    negative_words = get_stop_list("negative_dict")
    words = cut_word.CutWord(comment).cut()
    return [word for word in words if word in positive_words or word in negative_words or word not in stop_words]


def test():
    data = '你这么漂亮的，还是这么优秀的，并且这么自恋的，还有很多这么崇拜你的，真的贼优秀了，贼漂亮了，贼可爱了'
    words = get_cut_words([data])
    print "/".join(words)
    text = nltk.Text(words)
    # print text.concordance(word=u'五星'.encode("utf-8"), width=10, lines=5)
    # print text.concordance('漂亮')
    text.similar('漂亮')
    text.common_contexts(['可爱', '优秀'])
    text.concordance(word="这么", width=10, lines=3)
    print text.count("漂亮")
    text_collection = nltk.text.TextCollection(text)
    print text_collection.idf("漂亮")
    print text_collection.tf("漂亮", data)
    print text_collection.tf_idf("漂亮", data)
    content_index = nltk.text.ContextIndex(text)
    similarity_scores = content_index.word_similarity_dict(word='漂亮')
    for key, value in similarity_scores.items():
        if value > 0:
            print key, ":", value
    # bigrams_words = bigrams(words)
    # for word1,word2 in bigrams_words:
    #     print "[",word1,"/",word2,"]",
    # print ""
    # trigrams_words = trigrams(words)
    # for word1, word2,word3 in trigrams_words:
    #     print "[", word1, "/", word2, "/", word3, "]",
    # print ""
    # ngrams_words = ngrams(words, 4)
    # for word1, word2, word3, word4 in ngrams_words:
    #     print "[", word1, "/", word2, "/", word3, "/", word4, "]",
    # print ""
    # skipgrams_words = skipgrams(words,2,1)
    # for word1,word2 in skipgrams_words:
    #     print "[",word1,"/",word2,"]",
    # print ""
    # text.dispersion_plot(['\u6f02\u4eae', '\u53ef\u7231'])
    freq_dist = nltk.probability.FreqDist(words)
    for word, count in freq_dist.items():
        print word, count
    print freq_dist.max()
    print freq_dist["漂亮"]


def bayes_test():
    '''文本分类-贝叶斯'''
    '''训练数据'''
    boy_names = ["梓豪", "俊宇", "宇轩", "梓睿", "梓洋", "浩轩", "俊杰", "子轩", "浩然", "俊杰", "伟强", "思浩"]
    girl_names = ["芷晴", "雨桐", "梓琳", "思颖", "梓淇", "梓瑜", "梓妍", "晓彤", "梓琪", "嘉欣", "丽娟", "丽珍", "少英"]
    '''组成特征集，这里以第一个字和第二个字分别作为一个特征，并给特征集相对应的标签'''
    labeled_featuresets = [({'fname': bname[0], 'sname': bname[1]}, 'boy') for bname in boy_names] + \
                          [({'fname': gname[0], 'sname': gname[1]}, 'girl') for gname in girl_names]
    '''开始训练'''
    classifier = nltk.NaiveBayesClassifier.train(labeled_featuresets)
    '''测试数据'''
    test_names = ["丽晴", "俊杰", "子欣", "俊豪", "嘉琳", "空白"]
    '''组成特征集'''
    featuresets = [{'fname': tname[0], 'sname': tname[1]} for tname in test_names]
    print classifier.labels()
    print classifier.classify_many(featuresets)
    prob_dist_list = classifier.prob_classify_many(featuresets)
    for index, prob_dist in enumerate(prob_dist_list):
        print test_names[index]
        for label in prob_dist.samples():
            print("%s: %f" % (label, prob_dist.prob(label)))

    print classifier.show_most_informative_features()
    print classifier.most_informative_features()
    test_labeled_featuresets = [({'fname': '丽', 'sname': '晴'}, 'girl'), ({'fname': '俊', 'sname': '杰'}, 'boy'),
                                ({'fname': '子', 'sname': '欣'}, 'girl'), ({'fname': '俊', 'sname': '豪'}, 'boy'),
                                ({'fname': '嘉', 'sname': '琳'}, 'girl'), ({'fname': '空', 'sname': '白'}, 'boy')]
    print nltk.classify.accuracy(classifier, test_labeled_featuresets)


def decision_tree_test():
    '''文本分类-决策树'''
    '''训练数据'''
    boy_names = ["梓豪", "俊宇", "宇轩", "梓睿", "梓洋", "浩轩", "俊杰", "子轩", "浩然", "俊杰", "伟强", "思浩"]
    girl_names = ["芷晴", "雨桐", "梓琳", "思颖", "梓淇", "梓瑜", "梓妍", "晓彤", "梓琪", "嘉欣", "丽娟", "丽珍", "少英"]
    '''组成特征集，这里以第一个字和第二个字分别作为一个特征，并给特征集相对应的标签'''
    labeled_featuresets = [({'fname': bname[0], 'sname': bname[1]}, 'boy') for bname in boy_names] + \
                          [({'fname': gname[0], 'sname': gname[1]}, 'girl') for gname in girl_names]
    '''开始训练'''
    classifier = nltk.DecisionTreeClassifier.train(labeled_featuresets)
    '''测试数据'''
    test_names = ["丽晴", "俊杰", "子欣", "俊豪", "嘉琳", "空白"]
    test_labeled_featuresets = [({'fname': '丽', 'sname': '晴'}, 'girl'), ({'fname': '俊', 'sname': '杰'}, 'boy'),
                                ({'fname': '子', 'sname': '欣'}, 'girl'), ({'fname': '俊', 'sname': '豪'}, 'boy'),
                                ({'fname': '嘉', 'sname': '琳'}, 'girl'), ({'fname': '空', 'sname': '白'}, 'boy')]
    '''组成特征集'''
    featuresets = [{'fname': tname[0], 'sname': tname[1]} for tname in test_names]
    print classifier.labels()
    print classifier.classify_many(featuresets)
    print classifier.error(test_labeled_featuresets)
    print classifier.pretty_format()
    print classifier.leaf(test_labeled_featuresets)
    print classifier.pseudocode()
    print classifier.stump("fname", test_labeled_featuresets)



if __name__ == '__main__':
    # test()
    # bayes_test()
    decision_tree_test()
    """
    datas = get_data_from_sql()
    print "datas size:", len(datas)
    words = get_cut_words(datas)
    text = nltk.Text(words)
    print "words size:", len(words)
    # text.collocations(num=100)
    # double_words = bigrams(words)
    # trible_words = trigrams(words)
    # for tw in trible_words:
    #     print tw[0], ",", tw[1], ",", tw[2]
    bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
    bigrams = sorted(bigram_finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
    for bigram in bigrams:
        if bigram[1]>5:
            print bigram[0][0],",",bigram[0][1],":",bigram[1]
    # score_fn = BigramAssocMeasures.chi_sq
    # bigrams = bigram_finder.nbest(score_fn, 100)
    # for bigram in bigrams:
    #     print bigram[0], ",", bigram[1]
    """
    """
    datas = get_data_from_sql()
    print "datas size:", len(datas)
    words = get_cut_words(datas)
    """


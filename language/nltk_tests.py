# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.test import TestCase

import MySQLdb
import re
import os
import codecs
import numpy as np

import nltk
from nltk.util import bigrams, trigrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import SklearnClassifier

from language.cut_word import CutWord


# Create your tests here.
def get_data_from_sql():
    """获取数据"""
    mysql_cn = MySQLdb.connect(host='192.168.42.112', port=3306, user='unimonitor', passwd='unimonitor', db='g37')
    sql = "SELECT subject FROM weibo where source like 'taptap%' and comment_count!=0 and " \
          "post_date>='2018-10-1' and post_date<'2018-11-1' order by post_date"
    cursor = mysql_cn.cursor()
    cursor.execute(sql)
    alldata = cursor.fetchall()
    cursor.close()
    mysql_cn.close()
    r = '{机器型号:[\s\S]*?}|回复：[\s\S]*'
    alldata = [re.sub(r, '', data[0].decode("utf-8")) for data in alldata]
    return alldata


def get_stop_list():
    """获取停用词列表"""
    stop_list = []
    with codecs.open(os.path.dirname(os.path.realpath(__file__)) + "/dicts/stop_words.txt", 'r', encoding='utf8') as f:
        while True:
            line = f.readline()  # 逐行读取
            if not line:
                break
            stop_list.append(line.strip()),
    return stop_list


def get_cut_words(comments):
    stop_words = get_stop_list()
    words = []
    for comment in comments:
        cut_words = CutWord(comment, 'zh').cut()
        words += [word for word in cut_words if word not in stop_words]
    return words


def test():
    data = '你这么漂亮的，还是这么优秀的，并且这么自恋的，还有很多这么崇拜你的，真的贼优秀了，贼漂亮了，贼可爱了'
    words = get_cut_words([data])
    print "/".join(words)
    text = nltk.Text(words)
    # print text.concordance(word=u'五星'.encode("utf-8"), width=10, lines=5)
    # print text.concordance('漂亮')
    print text.similar('漂亮')
    print text.common_contexts(['漂亮', '优秀'])
    contentindex = nltk.text.ContextIndex(words)
    similarity_scores = contentindex.word_similarity_dict(word='漂亮')
    for key, value in similarity_scores.items():
        if value > 0:
            print key, ":", value
    print text.dispersion_plot(['\u6f02\u4eae', '\u53ef\u7231'])


if __name__ == '__main__':
    # test()
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
    bigrams = sorted(bigram_finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:100]
    for bigram in bigrams:
        print bigram[0][0],",",bigram[0][1],":",bigram[1]
    # score_fn = BigramAssocMeasures.chi_sq
    # bigrams = bigram_finder.nbest(score_fn, 100)
    # for bigram in bigrams:
    #     print bigram[0], ",", bigram[1]




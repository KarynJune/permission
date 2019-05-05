# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import MySQLdb
import codecs
import os
import collections
import numpy as np

from datas import cut_word


def get_data_from_filter():
    """
    获取分类好的数据
    [(['word1','word2'],label),(['word1','word2','word3'],label)]
    """
    mysql_cn = MySQLdb.connect(host='10.250.30.158', port=3306, user='root', passwd='88888888', db='zjy_test')
    sql = "SELECT content,sort FROM language_filterdata where sort!=0 and sort!=3"
    cursor = mysql_cn.cursor()
    cursor.execute(sql)
    alldata = cursor.fetchall()
    cursor.close()
    mysql_cn.close()
    return handle_datas(alldata)


def get_data_from_sort():
    """
    获取分类好的数据
    [(['word1','word2'],label),(['word1','word2','word3'],label)]
    """
    mysql_cn = MySQLdb.connect(host='10.250.30.158', port=3306, user='root', passwd='88888888', db='zjy_test')
    sql = """
    SELECT
        sd.content,
        sd.sort_id
    FROM
        language_sortdata sd
    LEFT JOIN language_sort s ON sd.sort_id = s.id
    where sort_id in (2,3,4,9,10)
"""
    cursor = mysql_cn.cursor()
    cursor.execute(sql)
    alldata = cursor.fetchall()
    cursor.close()
    mysql_cn.close()
    return handle_datas2(alldata)


def get_raw_data(db_name='g66'):
    """
    获取源数据
    [(['word1','word2'],label),(['word1','word2','word3'],label)]
    """
    mysql_cn = MySQLdb.connect(host='192.168.42.112', port=3306, user='unimonitor', passwd='unimonitor', db=db_name)
    sql = "SELECT data_content, content_note FROM facebook where `source`='taptap_review' and content_note is not null and content_note!='' order by post_datetime desc"
    cursor = mysql_cn.cursor()
    cursor.execute(sql)
    alldata = cursor.fetchall()
    cursor.close()
    mysql_cn.close()
    return handle_datas3(alldata)


def get_filewords(file_name):
    """获取停用词"""
    filepath = os.path.dirname(os.path.realpath(__file__)) + "/dicts/"+file_name
    words = [line.strip() for line in codecs.open(filepath, 'r', 'utf-8').readlines()]
    return words


def handle_datas(datas):
    """处理好数据并分词"""
    r = '{机器型号:[\s\S]*?}|回复：[\s\S]*'
    data_list = []
    stopwords = get_filewords("stop_words_min.txt")
    for data in datas:
        if int(data[1]) > 3:
            label = 1
        else:
            label = 0
        words_list = []
        words = cut_word.CutWord(re.sub(r, '', data[0]), 'zh').cut()
        for word in words:
            if word not in stopwords:
                words_list.append(word)
        data_list.append((words_list, label))
    return data_list


def handle_datas2(datas):
    """处理好数据并分词"""
    data_list = []
    # stopwords = get_filewords("stop_words_min.txt")
    label_dict = {2L: 0, 3L: 1, 4L: 2, 9L: 3, 10L: 4}
    data_dict = {}
    for data in datas:
        label = label_dict[data[1]]
        _list = data_dict.get(label, [])
        _list.append(cut_word.CutWord(data[0], 'zh').cut())
        data_dict[label] = _list
    max = np.max([len(data) for data in data_dict.values()])
    print "max data length:", max
    # 复制少量的样本的类别，过采样
    for label, data in data_dict.items():
        length = len(data)
        if length < max:
            data = data*(max/length)
        data_list += [(d,  label)for d in data]
    print collections.Counter([data[1] for data in data_list])
    return data_list


def handle_datas3(datas):
    """处理好数据并分词"""
    r = '{机器型号:[\s\S]*?}|回复：[\s\S]*'
    data_list = []
    for data in datas:
        if int(data[1]) > 3:
            label = 1
        else:
            label = 0
        words = cut_word.CutWord(re.sub(r, '', data[0]), 'zh').cut()
        data_list.append((words, label))
    return data_list


def write_file():
    datas = get_raw_data('g66')+get_raw_data('g37')+get_raw_data('h55')
    with codecs.open('taptap_g66_g37_h55.txt', 'w', 'utf-8') as f:
        for d in datas:
            f.write(" ".join(d[0]))
            f.write('\n')


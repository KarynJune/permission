# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.test import TestCase

import re, os, sys
import django

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

os.environ['DJANGO_SETTINGS_MODULE'] = 'permission.settings'
django.setup()

import codecs
import numpy as np
import string
from zhon import hanzi

import nltk
from nltk.util import bigrams, trigrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import SklearnClassifier

from datas import cut_word, databases
from language import service, util


# Create your tests here.

def build_date_dict(start_str, end_str):
    date_dict = {}
    gen = util.gen_range_by_day(util.toD(start_str), util.toD(end_str))
    for start, end in gen:
        start_str = util.toS(start)
        date_dict[start_str] = []

    return date_dict


def handle_data(start_str, end_str):
    datas = service.get_data_by_month('g37', start_str, end_str)

    full_words = []
    for data in datas:
        con = data['myContent']
        words = cut_word.CutWord(con, 'zh').cut()
        full_words += words + [","]
        print con
        print "【", "/".join(words), "】"
        print "="*100

    double_words = bigrams(full_words)
    double_word_dict = {}
    for dw in double_words:
        key = dw[0] + " " + dw[1]
        if "/" not in key and dw[0] != dw[1] and \
                not re.search(ur"[%s]+" % hanzi.punctuation, key) and \
                not re.search(ur"[%s]+" % string.punctuation.decode('unicode-escape'), key):
            _count = double_word_dict.get(key, 0)
            double_word_dict[key] = _count + 1
    for key,value in double_word_dict.items():
        if value>2:
            print key,value,"//",


if __name__ == '__main__':
    start_str = "2018-12-01"
    end_str = "2018-12-07"
    handle_data(start_str, end_str)

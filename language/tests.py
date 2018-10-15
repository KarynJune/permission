# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from snownlp import SnowNLP
import os
import jieba
jieba.load_userdict(os.path.dirname(os.path.realpath(__file__)) + "/dicts/game_dict.txt")

from django.test import TestCase

# Create your tests here.


texts = [
    u"这个东西真心很赞",
    u"很好玩，但是我手残，还是不太习惯，网络有问题，经常999延迟",
    u"画风什么的确实都是偏黑暗向，不过玩多了就知道其实只是音乐有些诡异",
    u"个人觉得还不错，但是我真的好好奇侦探的剧情啊",
    u"这也是一款很不错的游戏啦游戏体验还是蛮不错的，也希望更多人能喜欢",
    u"总体，并不推荐，抖m才玩排位赛监管者，优化差，手感差，画质……流畅度差，平衡性差",
    u"我是真的找不到游戏乐趣没错我很菜，因为这游戏没有给我一丁点练技术的欲望，这是我的真实感受",
    u"下载之后打不开;更新失败，然后就彻底打不开了，，，本来想试一下的。。还看到有专门的修复客户端，然而什么都点不动",
    u"不怎么样;补丁慢的你能等到猴年马月",
    u"游戏虽好，充钱太假",
    u"有些bug一直不修真是烦;比如翻一次窗就不能翻第二次 小丑隔窗拉锯 杰克隔墙打人 这都合理么",
    u"匹配一小时、游戏五分钟;、玩这游戏大部分时间都在等待。匹配机制太差劲！改成聊天工具好了、简直可笑！",
    u"不能玩了;iPad更新之后就不能玩了，一进去就闪退，满级号白白没了，希望官方能看一下，不过游戏挺好玩的。",
]


def snowNLP():
    """SnowNLP情感分析"""
    for text in texts:
        s = SnowNLP(text)
        print s.sentiments


def cut_word():
    for text in texts:
        words = jieba.cut(text)
        for word in words:
            print word,"/",
        print ""

snowNLP()

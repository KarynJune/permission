# -*- coding: utf-8 -*-
import re
import jieba
import os
import nltk
# from nltk.corpus import stopwords
import jieba.analyse
from zhon import hanzi
import string


jieba.load_userdict(os.path.dirname(os.path.realpath(__file__)) + "/dicts/game_dict.txt")
jieba.load_userdict(os.path.dirname(os.path.realpath(__file__)) + "/dicts/yys_dict.txt")


class CutWord:

    def __init__(self, comment, language):
        self.comment = comment.lower()
        self.language = language
        self.temp_list = []

    def cut(self):
        self._clean_data()
        self._language_classify()

        return self.temp_list

    def revert_tag(self):
        """还原词性"""
        lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
        return lmtzr.lemmatize(self.comment)

    def _clean_data(self):
        r = '[a-zA-z]+://[^\s]*'
        temp_comment = re.sub(r, '', self.comment)
        # temp_comment = re.sub(ur"[%s]+" % hanzi.punctuation, " ", temp_comment)  # 处理中文标点
        # temp_comment = re.sub(ur"[%s]+" % string.punctuation.decode('unicode-escape'), " ", temp_comment)  # 处理英文标点
        self.comment = temp_comment

    def _language_classify(self):
        if self.language == 'zh':
            self._cut_word_zh()
        elif self.language == 'en':
            self._cut_word_en()

    def _cut_word_zh(self):
        words = jieba.cut(self.comment)
        self.temp_list = [word for word in words if len(word.strip()) > 0]
        # self.temp_list = jieba.cut(self.comment)

    def _cut_word_en(self):
        self._sentences_tokenizer()
        self._words_tokenizer()
        self._words_pos_tag()
        self._works_revert()

    def _sentences_tokenizer(self):
        """
        self.temp_list = [[['sentence 11'], ['sentence 12']], [['sentence 21'], ['sentence 22']], ...]
        """
        sentences = nltk.sent_tokenize(self.comment)
        self.temp_list = sentences

    def _words_tokenizer(self):
        """
        self.temp_list = [[sentence1 word1, sentence1 word2, ...], [sentence2 words], ...]
        """
        words = []
        for sentence in self.temp_list:
            words.append(nltk.word_tokenize(sentence))
        self.temp_list = words

    def _words_pos_tag(self):
        """
        self.temp_list = [[(sentence1 word1, tag1), (sentence1 word2, tag2), ...] ...]
        """
        words_tags = []
        for sentence_tokens in self.temp_list:
            words_tags.append(nltk.pos_tag(sentence_tokens))
        self.temp_list = words_tags

    def _works_revert(self, is_all=True):
        words = []
        lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
        for tag_words in self.temp_list:
            for word, tag in tag_words:
                lemmatized_tag = self._get_wordnet_pos(tag)
                if lemmatized_tag:
                    if tag.startswith('V') or tag.startswith('R'):
                        word = self._match_possessive(word)
                    lemmatized_word = lmtzr.lemmatize(word, lemmatized_tag)
                    if not is_all:  # 过滤后
                        words.append(lemmatized_word)
                else:
                    lemmatized_word = lmtzr.lemmatize(word)
                if is_all:  # 没有过滤
                    words.append(lemmatized_word)
        self.temp_list = words

    def _match_possessive(self, word):
        _dict = {
            "'s": "is",
            "s": "is",
            "'ve": "have",
            "ve": "have",
            "n't": "not",
            "nt": "not",
            "'m": "am",
            "m": "am",
        }
        return _dict.get(word, word)

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return ''

    def extract_word(self):
        self.temp_list = jieba.analyse.tfidf(self.comment, topK=50)

    def _filter(self):
        self.comment = re.sub(ur"[%s]+" % hanzi.punctuation, " ", self.comment)
        self.comment = re.sub(ur"[%s]+" % string.punctuation.decode('unicode-escape'), " ", self.comment)


if __name__ == '__main__':
    comment = u"今天的天气真好……二周年回坑的游戏，蓝票150抽出了2个白藏主,还好3张紫票出了茨木童子和大天狗，加上换的妖刀姬，这游戏又能玩了"
    # comment = "储值了没勾玉问题处理好没? 钱我就给了，勾玉呢? 有没在处理"
    # comment = "游戏还可以，除了bug很多，数学太难，但其他体验都很好，一生挚爱我的艾米丽。"
    # comment = "網路一直盪，中華電信說我的網路通暢，那到底是誰的問題？"
    # comment = 'Please tell me what is "slide to look around" in settings? What does it do?'
    wc = CutWord(comment, 'zh')
    # words = wc.extract_word()
    words = wc.cut()
    for w in words:
        print w,"/",

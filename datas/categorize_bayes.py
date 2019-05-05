# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn import metrics
import collections
import datetime

from databases import get_data_from_sort


def get_best_param(pipeline):
    """
    找到分类器最佳参数，2000数据大约需要2min
    :param pipeline:
    :return:
    """
    parameters = {
        'vect__max_df': (0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        'vect__max_features': (None, 50, 100, 500, 1000, 2000, 5000, 10000),
        # 'tfidf__use_idf': (True, False),
        'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001),
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1)
    from time import time

    t0 = time()
    grid_search.fit(X_train, y_train)
    print "done in %0.3fs" % (time() - t0)
    print "Best score: %0.3f" % grid_search.best_score_

    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])


if __name__ == '__main__':
    """
    starttime = datetime.datetime.now()
    datas = get_data_from_sort()
    print "sum datas count:", len(datas)
    X = []  # 特征集
    y = []  # 类别
    # 统计词频
    labels = [0, 1, 2, 3, 4]  # 总类别
    features_dict = {label: [] for label in labels}
    for data in datas:
        features_dict[data[1]] += data[0]
    # 剪掉一些不要的词
    for label, feature in features_dict.items():
        counts = collections.Counter(feature).most_common()
        for word, count in counts:
            if count <= 1:
                feature.remove(word)
    # 组装
    for data in datas:
        words = features_dict[data[1]]
        new_words = []
        for w in data[0]:
            if w in words:
                new_words.append(w)
        X.append(" ".join(new_words))
        y.append(data[1])

    X_train, y_train = shuffle(X, y)  # 打乱数据顺序
    # for index, x in enumerate(X_train):
    #     print y_train[index]
    #     print x
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        # ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    # get_best_param(pipeline)
    #
    # pipeline.set_params(clf__alpha=0.01, tfidf__use_idf=True, vect__max_df=0.2, vect__max_features=None)
    pipeline.set_params(clf__alpha=0.0001, vect__max_df=0.4, vect__max_features=None)
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, 'categorize_model.joblib')  # 保存模型

    y_pred = pipeline.predict(X_train)

    print metrics.accuracy_score(y_train, y_pred)
    print metrics.confusion_matrix(y_train, y_pred)

    print "spend ",(datetime.datetime.now()-starttime)

    y_pred = pipeline.predict(X_test)

    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)
"""
    pipeline = joblib.load("categorize_model.joblib")
    #
    X_test = [
        "联合 到底 是 有 什么 问题",
        "更新 后 开始 逃生 时 蓝屏 怎么办",
        "非常 好玩",
        "为什么 进不去",
        "我 不知道 怎么 同居 谁 能 告诉 我",
        "氪金",
        "bug 投诉 十六 遍 网易 做 什么 四次 更新 bug 还 不修 动荡 之城 匹配 机制 什么 高战 营地 匹配 中 低 等级 营地 休闲 营地 匹配 高战 营地 什么 意思 娱乐 玩家 战争 玩家 划等号 战争 营地 匹配 休闲 营地 动荡 什么 意思 纯粹 吊打 搞 私服 吗 天天 开新服 生怕 别人 不 知道 私服 吗 人 去 新服 再 氪金 停服 民心 不 朋友 都 退游 退游 评论 1 谢谢 退游 人数 7",
        "一个字 肝",
        "还行",
        "菜 玩 几个 小时",
        "还 可以",
        "优化 比较 差",
        "还 可以 就是 服务器 真的 垃圾",
        "问 一个 问题 什么 时候 联动",
        "肝 快乐",
        "问 一下 怎么 加 好友",
        "希望 室友 可以 交易 啊",
        "还 可以 氪金 才能 变强",
        "很 棒棒",
    ]
    y_category = pipeline.predict(X_test)
    # # print metrics.accuracy_score(y_test, y_pred)
    # # print metrics.confusion_matrix(y_test, y_pred)

    y_proba = pipeline.predict_proba(X_test)
    for doc, category, proba in zip(X_test, y_category, y_proba):
        print doc, ":", category, ":", proba
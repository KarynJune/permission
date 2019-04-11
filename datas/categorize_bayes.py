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
    datas = get_data_from_sort()
    print "sum datas count:",len(datas)
    X = []  # 特征集
    y = []  # 类别
    # 统计词频
    labels = [2, 3, 4, 9, 10]  # 总类别
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
        # print data[1]
        # print " ".join(new_words)

    X_train, y_train = shuffle(X, y)
    # for index, x in enumerate(X_train):
    #     print y_train[index]
    #     print x
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        # ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    # get_best_param(pipeline)
    #
    # pipeline.set_params(clf__alpha=0.01, tfidf__use_idf=True, vect__max_df=0.2, vect__max_features=None)
    pipeline.set_params(clf__alpha=0.01, vect__max_df=0.2, vect__max_features=None)
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, 'categorize_model.joblib')  # 保存模型

    y_pred = pipeline.predict(X_train)

    print metrics.accuracy_score(y_train, y_pred)
    print metrics.confusion_matrix(y_train, y_pred)

    #
    pipeline = joblib.load("categorize_model.joblib")
    #
    X_test = ["联合 到底 是 有 什么 问题","更新 后 开始 逃生 时 蓝屏 怎么办","非常 好玩","为什么 进不去","真的 生气"]
    y_pred = pipeline.predict(X_test)
    # # print metrics.accuracy_score(y_test, y_pred)
    # # print metrics.confusion_matrix(y_test, y_pred)
    #
    for doc, category in zip(X_test, y_pred):
        print doc, ":", category

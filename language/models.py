# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from rest_framework import serializers


class SomeData(models.Model):
    name = models.CharField(max_length=255)


class Post(models.Model):
    # content = models.TextField(null=True, db_column='post_content')
    content = models.TextField(null=True, db_column='subject')
    post_date = models.DateTimeField(null=True)
    source = models.CharField(max_length=255)
    score = models.IntegerField(default=0, db_column="comment_count")  # 评分

    class Meta:
        db_table = "weibo"


class Facebook(models.Model):
    unique_id = models.CharField(max_length=255, db_column="unique_identifier")
    content = models.TextField(null=True, db_column="data_content")
    post_date = models.DateTimeField(null=True, db_column="post_datetime")
    modified_date = models.DateTimeField(null=True, db_column="modified_datetime")
    source = models.CharField(max_length=255)
    text_source = models.CharField(max_length=255, db_column="relation_kind")
    url = models.CharField(max_length=255, null=True, blank=True, db_column="data_url")
    author_name = models.CharField(max_length=255, db_column="urs_nickname")
    author_head = models.CharField(max_length=255, db_column="urs_note")
    author_id = models.CharField(max_length=255, db_column="urs_account")
    like_count = models.IntegerField(default=0)
    share_count = models.IntegerField(default=0)
    star = models.TextField(db_column="content_note")
    phone_type = models.CharField(max_length=255, db_column="urs_device_model")
    isDelete = models.CharField(max_length=255, db_column="recall_note")
    floor = models.CharField(max_length=100)

    parent_node = models.ForeignKey('self', db_column="master_data_id", related_name="children")

    class Meta:
        # db_table = "facebook"
        db_table = "g66_taptap"


class FilterData(models.Model):
    content = models.TextField(null=True)
    post_date = models.DateTimeField(null=True)
    source = models.CharField(max_length=255)
    sort = models.CharField(max_length=255)
    product = models.CharField(max_length=255, default='g37')


class Sort(models.Model):
    name = models.TextField(null=True)


class SortData(models.Model):
    content = models.TextField(null=True)
    comment_id = models.IntegerField(unique=True)
    post_date = models.DateTimeField(null=True)
    source = models.CharField(max_length=255)
    sort = models.ForeignKey(Sort, related_name='sorts')
    product = models.CharField(max_length=255, default='g37')


class CommentInfo(models.Model):
    # 对应评论在爬虫数据库里的id
    comment_id = models.IntegerField()
    product_id = models.IntegerField()

    COMMENT_TYPE_CHOICES = (u'好评', u'差评', 'BUG', u'建议', u'咨询', u'充值', u'外挂', u'其他',)

    comment_type = models.CharField(max_length=255, null=True, blank=True)
    sub_type = models.CharField(max_length=255, null=True, blank=True)

    CHOICES = (
        (0, u'未处理'),
        (1, u'已回复'),
        (2, u'不处理'),
    )
    status = models.IntegerField(choices=CHOICES, default=0)
    update_time = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "business_commentinfo"




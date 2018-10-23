# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from rest_framework import serializers

import re

# Create your models here.

r = r'\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}|' \
            '\[.*?\]|<style>[\s\S]*?<\/style>|<script>[\s\S]*?<\/script>|<[^>]+>?|' \
            '本帖最后由[\s\S]*?编辑|\d{1,3}楼.|Reply[\s\S]*?[)]|' \
            '{机器型号:[\s\S]*?}|回复：[\s\S]*|' \
            '[a-zA-z]+://[^\s]*|\n*'
pattern = re.compile(r)


class SomeData(models.Model):
    name = models.CharField(max_length=255)


class SomeDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SomeData
        fields = ('id', 'name')


class Post(models.Model):
    # content = models.TextField(null=True, db_column='post_content')
    content = models.TextField(null=True, db_column='subject')
    post_date = models.DateTimeField(null=True)
    source = models.CharField(max_length=255)
    score = models.IntegerField(default=0, db_column="comment_count")  # 评分

    class Meta:
        db_table = "weibo"


class PostSerializers(serializers.ModelSerializer):
    myContent = serializers.SerializerMethodField('get_content_text')
    myDate = serializers.SerializerMethodField('get_post_date')

    def get_content_text(self, obj):
        _content = pattern.sub("", obj.content)
        content_arr = _content.split("发表于")
        if len(content_arr) > 1: _content = content_arr[1]
        return _content

    def get_post_date(self, obj):
        _date = obj.post_date.strftime("%Y-%m-%d %H:%M")
        return _date

    class Meta:
        model = Post
        fields = ('id', 'content', 'post_date', 'source', 'score', 'myContent', 'myDate')


class FilterData(models.Model):
    content = models.TextField(null=True)
    post_date = models.DateTimeField(null=True)
    source = models.CharField(max_length=255)
    sort = models.CharField(max_length=255)


class FilterDataSerializers(serializers.ModelSerializer):
    class Meta:
        model = FilterData
        fields = ('id', 'content', 'post_date', 'source', 'sort')

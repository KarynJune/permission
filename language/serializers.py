# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from rest_framework import serializers
from language.models import SomeData,Post,Facebook,FilterData,SortData

import re

# Create your models here.

r = r'\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}|' \
            '\[.*?\]|<style>[\s\S]*?<\/style>|<script>[\s\S]*?<\/script>|<[^>]+>?|' \
            '本帖最后由[\s\S]*?编辑|\d{1,3}楼.|Reply[\s\S]*?[)]|' \
            '{机器型号:[\s\S]*?}|回复：[\s\S]*|' \
            '[a-zA-z]+://[^\s]*|\n*'
pattern = re.compile(r)


class SomeDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SomeData
        fields = ('id', 'name')


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


class FacebookSerializers(serializers.ModelSerializer):
    post_date = serializers.SerializerMethodField()

    def get_post_date(self, obj):
        _date = obj.post_date.strftime("%Y-%m-%d %H:%M")
        return _date
    """
    无子舆情序列化器，含完整字段
    """
    class Meta:
        model = Facebook
        fields = '__all__'


class FilterDataSerializers(serializers.ModelSerializer):
    class Meta:
        model = FilterData
        fields = ('id', 'content', 'post_date', 'source', 'sort', 'product')


class SortDataSerializers(serializers.ModelSerializer):
    class Meta:
        model = SortData
        fields = '__all__'


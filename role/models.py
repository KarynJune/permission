# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from django.contrib.auth.models import User, Group, Permission
from rest_framework import serializers
import re


# Create your models here.


class SomeData(models.Model):
    name = models.CharField(max_length=255)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'password')


class GroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = Group
        fields = ('id', 'name', 'permissions')


class PermissionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Permission
        fields = ('id', 'name', 'content_type', 'codename')


class SomeDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SomeData
        fields = ('id', 'name')


class Post(models.Model):
    content = models.TextField(null=True, db_column='post_content')
    post_date = models.DateTimeField(null=True)
    source = models.CharField(max_length=255)
    subjectUrl = models.CharField(max_length=255, null=True, blank=True)
    author_name = models.CharField(max_length=255)
    floor = models.IntegerField(default=0)

    class Meta:
        db_table = "post"


class PostSerializers(serializers.ModelSerializer):
    myContent = serializers.SerializerMethodField('get_content_text')

    def get_content_text(self, obj):
        r = '\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}|' \
            '\[.*?\]|<style>[\s\S]*?<\/style>|<script>[\s\S]*?<\/script>|<[^>]+>?|' \
            '[\s\S]*?发表于|本帖最后由[\s\S]*?编辑|' \
            '[a-zA-z]+://[^\s]*|\n*'
        obj.content = re.sub(r, '', obj.content) if obj.content else ''
        obj.content = obj.content.replace(',', ';')

        return obj.content

    class Meta:
        model = Post
        fields = ('id', 'content', 'post_date', 'source', 'subjectUrl', 'author_name', 'floor', 'myContent')

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from django.contrib.auth.models import User
from rest_framework import serializers

# Create your models here.


class SomeData(models.Model):
    name = models.CharField(max_length=255)


class UserSerializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields = ('id', 'username', 'password')


class SomeDataSerializer(serializers.ModelSerializer):

    class Meta:
        model = SomeData
        fields = ('id', 'name')

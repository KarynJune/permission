# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from rest_framework import viewsets
from language.models import SomeData, SomeDataSerializer, Post, PostSerializers, FilterData, FilterDataSerializers
from role.permission import IsAdminOrReadOnly

import MySQLdb

# Create your views here.


class SomeDataViewSet(viewsets.ModelViewSet):
    queryset = SomeData.objects.all()
    serializer_class = SomeDataSerializer
    permission_classes = (IsAdminOrReadOnly,)


class FilterDataViewSet(viewsets.ModelViewSet):
    queryset = FilterData.objects.all()
    serializer_class = FilterDataSerializers
    # permission_classes = (IsAdminOrReadOnly,)





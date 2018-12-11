# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view
from language.models import SomeData, SomeDataSerializer, Post, PostSerializers, FilterData, FilterDataSerializers
from role.permission import IsAdminOrReadOnly
import MySQLdb

from keras_learning import service

# Create your views here.


class SomeDataViewSet(viewsets.ModelViewSet):
    queryset = SomeData.objects.all()
    serializer_class = SomeDataSerializer
    permission_classes = (IsAdminOrReadOnly,)


class FilterDataViewSet(viewsets.ModelViewSet):
    queryset = FilterData.objects.all()
    serializer_class = FilterDataSerializers
    # permission_classes = (IsAdminOrReadOnly,)


@api_view(['POST'])
def get_emotion_test(request):
    content_arr = request.POST.getlist('content_arr[]', [])

    preds = service.predict_data(content_arr)
    return Response(preds)


@api_view(['GET'])
def retrain(request):

    result = service.retrain()
    return Response(result)


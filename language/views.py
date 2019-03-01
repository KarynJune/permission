# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view
from language.models import SomeData, Post, FilterData, Sort,SortData
from language.serializers import SomeDataSerializer, PostSerializers, FilterDataSerializers,SortDataSerializers
from language.service import get_data2
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


class SortDataViewSet(viewsets.ModelViewSet):
    queryset = SortData.objects.all()
    serializer_class = SortDataSerializers


@api_view(['POST'])
def get_emotion_test(request):
    content_arr = request.POST.getlist('content_arr[]', [])

    preds, pos_list, neg_list = service.predict_data(content_arr)
    return Response({"preds": preds, "pos_list": pos_list, "neg_list": neg_list})


@api_view(['GET'])
def retrain(request):
    result = service.retrain()
    return Response(result)


"""===============================================   分类"""


def to_sort_index(request, product_code):
    page = int(request.GET.get("page", 1))
    key = request.GET.get("key", "")
    datas = get_data2(page, product_code, key)
    return render(request, "sort.html", {
        "datas": datas,
        "sorts": Sort.objects.all(),
        "page": page,
        "product_code": product_code})




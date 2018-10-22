# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, redirect
from django.contrib.auth.models import User, Group, Permission
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from rest_framework import viewsets
from rest_framework.response import Response

from models import UserSerializer, GroupSerializer, PermissionSerializer, SomeData, SomeDataSerializer, Post, \
    PostSerializers
from permission import IsAdminOrReadOnly
import MySQLdb
import json


# Create your views here.


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

    def create(self, request, *args, **kwargs):
        User.objects.create_user(username=request.POST['username'], password=request.POST['password'])
        to_login(request)


class GroupViewSet(viewsets.ModelViewSet):
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = (IsAdminOrReadOnly,)


class PermissionViewSet(viewsets.ModelViewSet):
    queryset = Permission.objects.all()
    serializer_class = PermissionSerializer
    permission_classes = (IsAdminOrReadOnly,)


class SomeDataViewSet(viewsets.ModelViewSet):
    queryset = SomeData.objects.all()
    serializer_class = SomeDataSerializer
    permission_classes = (IsAdminOrReadOnly,)


def get_data(page):
    posts = Post.objects.using("g37").all()[(page - 1) * 100:100 * page]
    posts_serializers = PostSerializers(posts, many=True)
    return posts_serializers.data


def to_login(request):
    if request.method == 'POST':
        user = authenticate(username=request.POST['username'], password=request.POST['password'])
        if user is not None:
            login(request, user)
        return redirect("/index/")
    else:
        return render(request, "login.html")


def to_logout(request):
    logout(request)
    return redirect("/login/")


@login_required
def to_index(request):
    page = int(request.GET.get("page", 1))
    datas = get_data(page)
    return render(request, "index.html", {"groups": Group.objects.all(), "datas": datas})

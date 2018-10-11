# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, redirect
from django.contrib.auth.models import User, Group, Permission
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from rest_framework import viewsets
from rest_framework.response import Response

from models import UserSerializer,SomeData, SomeDataSerializer
from permission import IsAdminOrReadOnly

# Create your views here.


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

    def create(self, request, *args, **kwargs):
        super(UserViewSet, self).create(request, *args, **kwargs)
        role_name = '管理员'
        try:
            Group.objects.get(name=role_name)
        except Group.DoesNotExist:
            groups = Group.objects.create(name=role_name)
            groups.permissions.add(Permission.objects.get(codename='add_somedata'))
            user = User.objects.get(username=request.POST['username'], password=request.POST['password'])
            if user is not None:
                user.groups.add(groups)
        to_login(request)
        return redirect("/index/")


class SomeDataViewSet(viewsets.ModelViewSet):
    queryset = SomeData.objects.all()
    serializer_class = SomeDataSerializer
    permission_classes = (IsAdminOrReadOnly,)


def to_login(request):
    if request.method == 'POST':
        user = User.objects.get(username=request.POST['username'], password=request.POST['password'])
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
    if request.method == 'POST':
        Response()
    else:
        return render(request, "index.html")

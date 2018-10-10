# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from rest_framework import viewsets
from rest_framework.response import Response

from models import UserSerializer

# Create your views here.


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

    def create(self, request, *args, **kwargs):
        super(UserViewSet, self).create(request, *args, **kwargs)
        to_login(request)
        return to_index(request)


def to_login(request):
    if request.method == 'POST':
        user = User.objects.get(username=request.POST['username'], password=request.POST['password'])
        if user is not None:
            login(request, user)
        return render(request, "index.html")
    else:
        return render(request, "login.html")


def to_index(request):
    if request.method == 'POST':
        Response()
    else:
        return render(request, "index.html")

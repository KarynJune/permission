"""permission URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url,include
from django.contrib import admin
from rest_framework import routers

import views

router = routers.DefaultRouter()
router.register(r'data', views.SomeDataViewSet)
router.register(r'filter/data', views.FilterDataViewSet)
router.register(r'sort/data', views.SortDataViewSet)

urlpatterns = [
    url(r'^', include(router.urls)),
    url(r'^emotion/', views.get_emotion_test),
    url(r'^retrain/', views.retrain),

    url(r'^sort/(?P<product_code>\w+)', views.to_sort_index),
]
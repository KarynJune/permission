# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.db.models import Q

from language.models import Post, PostSerializers
from language.db import connectDB


def get_data(page, product_code):
    connectDB(product_code)
    query = Q(source__istartswith='taptap', post_date__lt='2018-12-01') & ~Q(score=0)
    posts = Post.objects.using(product_code).extra(where=['LENGTH(subject)<=500']).filter(query).order_by("-post_date")[(page - 1) * 200:200 * page]
    # posts = Post.objects.using("g37").all().order_by("-post_date")[(page - 1) * 100:100 * page]
    posts_serializers = PostSerializers(posts, many=True)
    return posts_serializers.data

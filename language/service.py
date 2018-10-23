# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.db.models import Q

from language.models import Post, PostSerializers


def get_data(page):
    query = Q(source__istartswith='taptap',) & ~Q(score=0)
    posts = Post.objects.using("g37").filter(query).order_by("-post_date")[(page - 1) * 100:100 * page]
    # posts = Post.objects.using("g37").all().order_by("-post_date")[(page - 1) * 100:100 * page]
    posts_serializers = PostSerializers(posts, many=True)
    return posts_serializers.data

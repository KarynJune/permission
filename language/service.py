# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db.models import Q

from language.db import connectDB
from language.models import Post, Facebook, SortData, CommentInfo
from language.serializers import PostSerializers, FacebookSerializers


def get_data(page, product_code):
    connectDB(product_code)
    query = Q(source='taptap_review') & ~Q(score=0)
    posts = Post.objects.using(product_code).extra(where=['char_length(subject)<=500']).filter(query).order_by(
        "-post_date")[(page - 1) * 200:200 * page]
    # posts = Post.objects.using("g37").all().order_by("-post_date")[(page - 1) * 100:100 * page]
    posts_serializers = PostSerializers(posts, many=True)
    return posts_serializers.data


def get_taptap_data(page, product_code):
    query = Q(source='taptap_review', text_source=None)
    datas = Facebook.objects.filter(query).order_by("-post_date")[(page - 1) * 200:200 * page]
    print datas
    datas_serializers = FacebookSerializers(datas, many=True).data
    infos = CommentInfo.objects.values_list("comment_id", "comment_type").filter(product_id=222)
    info_dict = {info[0]:info[1] for info in infos}
    for data in datas_serializers:
        data["b_sort"] = info_dict.get(data['id'], "")

    return datas_serializers


def get_data2(page, product_code, key):
    connectDB(product_code)

    sorts = SortData.objects.values_list("comment_id","sort__name").filter(product=product_code)
    sort_dict = {sort[0]:sort[1] for sort in sorts}
    query = Q(source='taptap_review',text_source=None, content__icontains=key)
    datas = Facebook.objects.using(product_code).filter(query).order_by("-post_date")[(page - 1) * 200:200 * page]
    data_list =[]
    for data in datas:
        data_list.append({
            "post_date":data.post_date.strftime("%Y-%m-%d %H:%M:%S"),
            "content":data.content,
            "source":data.source,
            "unique_id":data.unique_id,
            "sort": sort_dict.get(long(data.unique_id),"")
        })
    return data_list


def get_data_by_month(product_code, start_str, end_str):
    connectDB(product_code)
    query = Q(source='taptap_review', post_date__gte=start_str, post_date__lt=end_str) & ~Q(score=0)
    posts = Post.objects.using(product_code).filter(query).order_by("-post_date")
    posts_serializers = PostSerializers(posts, many=True)
    return posts_serializers.data

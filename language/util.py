# -*- coding: utf-8 -*-
import datetime
import re


def toS(date):
    """时间转字符串"""
    return date.strftime("%Y-%m-%d")


def toD(string):
    """字符串转时间"""
    return datetime.datetime.strptime(string, "%Y-%m-%d")


def to_time_str(date):
    """时间转字符串 到分钟"""
    return date.strftime("%Y-%m-%d %H:%M")


def to_time_date(string):
    """字符串转时间 到分钟"""
    return datetime.datetime.strptime(string, "%Y-%m-%d %H:%M")


def gen_range(start, end, step):
    cursor = start
    while cursor < end:
        next_point = cursor + datetime.timedelta(**step)
        yield cursor, next_point
        cursor = next_point


def gen_range_by_minute(start, end):
    """获取每分钟"""
    return gen_range(start, end, {"minutes": 1})


def gen_range_by_hour(start, end):
    """获取每小时"""
    return gen_range(start, end, {"hours": 1})


def gen_range_by_day(start, end):
    """获取每日时间"""
    return gen_range(start, end, {"days": 1})


def gen_range_by_week(start, end):
    """获取每周时间"""
    start = start - datetime.timedelta(days=(start.isoweekday() - 1))
    end = end - datetime.timedelta(days=end.isoweekday())
    if datetime.datetime.now() < end:  # 判断这周是否过完
        end = end - datetime.timedelta(weeks=1)

    return gen_range(start, end, {"days": 7})


def get_week(start_date, end_date):
    """获取周"""
    start = toD(start_date)
    end = toD(end_date)

    start = start - datetime.timedelta(days=(start.isoweekday() - 1))
    end = end - datetime.timedelta(days=end.isoweekday())
    if datetime.datetime.now() < end:  # 判断这周是否过完
        end = end - datetime.timedelta(weeks=1)

    return toS(start), toS(end + datetime.timedelta(days=1))


def diff_minute(date_str, minute_scope):
    """获取某时间的前几分钟"""
    return to_time_date(date_str) - datetime.timedelta(minutes=minute_scope)


def check_contain_chinese(check_str):
    """是否含有中文"""
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    if zh_pattern.search(check_str):
        return True
    return False


def differ_list(diff_list, _list):
    """差集"""
    return list(set(_list).difference(set(diff_list)))


def and_list(diff_list, _list):
    """交集"""
    return list(set(_list) & set(diff_list))

# -*- coding: utf-8 -*-

from rest_framework import permissions


class IsAdminOrReadOnly(permissions.BasePermission):
    def has_permission(self, request, view):
        print "has_permission...."
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user.is_superuser or 'role.add_somedata' in request.user.get_group_permissions()

    # def has_object_permission(self, request, view, blog):
    #     # Read permissions are allowed to any request,
    #     # so we'll always allow GET, HEAD or OPTIONS requests.
    #     print "has_object_permission...."
    #     if request.method in permissions.SAFE_METHODS:
    #         return True
    #     print request.user.is_superuser
    #     return request.user.is_superuser

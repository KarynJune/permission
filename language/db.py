from django.db import connections


def connectDB(projectName):
    # connections.close_all()
    connections.databases[projectName]={
        'ENGINE': 'django.db.backends.mysql',
        'NAME': projectName,
        'USER': 'unimonitor',
        'PASSWORD': 'unimonitor',
        'HOST': '192.168.42.112',
        'PORT': '3306',
        'OPTIONS': {'charset': 'utf8'},
    }
    # connections.databases[projectName] = {
    #     'ENGINE': 'django.db.backends.mysql',
    #     'NAME': projectName,
    #     'USER': 'root',
    #     'PASSWORD': '88888888',
    #     'HOST': '10.250.40.99',
    #     'PORT': '3306',
    #     'OPTIONS': {'charset': 'utf8'},
    # }
    return connections


def get_connections():
    return connections

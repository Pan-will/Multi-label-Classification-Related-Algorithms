"""MyBlog URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls import include, url
from django.urls import path
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    #增加命名空间函数，根URL中app_name是多余参数，要删掉
    url(r'^blog/', include('blog.urls', namespace='blog')),
    url(r'^account/', include('account.urls', namespace='account')),
    url(r'^article/', include('article.urls', namespace='article')),
    url(r'^home/', TemplateView.as_view(template_name="home.html"), name="home"),
]

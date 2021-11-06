from . import views
from django.conf.urls import url

app_name = 'blog'   #在子URL中加上这一行
urlpatterns = [
    url(r'^$', views.blog_title, name="blog_title"),
    url(r'(?P<article_id>\d)/$', views.blog_article, name="blog_detail"),
]
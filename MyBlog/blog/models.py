from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


# 自定义模型类
class Articles(models.Model):
    title = models.CharField(max_length=300)
    author = models.ForeignKey(User, related_name="blog_posts", on_delete=models.CASCADE)
    body = models.TextField()
    publish = models.DateTimeField(default=timezone.now)

class Meta:
    # 文章按发布日期倒序排列
    ordering = ['-publish']

def __str__(self):
    return self.title
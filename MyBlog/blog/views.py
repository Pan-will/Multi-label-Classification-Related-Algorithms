from django.shortcuts import render
from .models import Articles
from django.shortcuts import get_object_or_404, render

def blog_title(request):
    blogs = Articles.objects.all()
    return render(request, "blog/titles.html", {"blogs": blogs})

# 通过ID查看博客文章内容
def blog_article(request, article_id):
    article = Articles.objects.get(id=article_id)
    # article = get_object_or_404(Articles, id=article_id)
    return render(request, "blog/content.html", {"article":article})
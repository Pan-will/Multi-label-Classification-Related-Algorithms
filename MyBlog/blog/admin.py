from django.contrib import admin
from .models import Articles

# 为了让显示信息的列表页更丰富
class ArticleAdmin(admin.ModelAdmin):
    list_display = ("title", "author", "publish")
    list_filter = ("author", "publish", )
    ordering = ['publish']
    search_fields = ('title', 'body')
    raw_id_fields = ("author",)
    date_hierarchy = "publish"

admin.site.register(Articles, ArticleAdmin)

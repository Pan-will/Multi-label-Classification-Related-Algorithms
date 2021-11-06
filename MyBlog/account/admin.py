from django.contrib import admin
from .models import UserInfo, UserData

class UserInfoAdmin(admin.ModelAdmin):
    list_display = ('user', 'birth', 'phone')
    list_filter = ('phone',)

class UserDataAdmin(admin.ModelAdmin):
    list_display = ('user', 'company', 'profession', 'address', 'aboutme', 'photo')
    list_filter = ('company',)

admin.site.register(UserData, UserDataAdmin)

from django.db import models
from django.contrib.auth.models import User

class UserInfo(models.Model):
    user = models.OneToOneField(User, unique=True, on_delete=models.CASCADE)
    birth = models.DateField(blank=True, null=True)
    phone = models.CharField(blank=True, null=True, max_length=11)

    def __str__(self):
        return 'user {}'.format(self.user.username)

class UserData(models.Model):
    user = models.OneToOneField(User, unique=True, on_delete=models.CASCADE)
    company = models.CharField(blank=True, null=True, max_length=100)
    profession = models.CharField(blank=True, null=True, max_length=100)
    address = models.CharField(blank=True, null=True, max_length=100)
    aboutme = models.CharField(blank=True, null=True, max_length=500)
    photo = models.ImageField(blank=True)

    def __str__(self):
        return 'user {}'.format(self.user.username)

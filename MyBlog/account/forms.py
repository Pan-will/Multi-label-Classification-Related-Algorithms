from django import forms
from django.contrib.auth.models import User
from .models import UserInfo, UserData

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

class RegisterForm(forms.ModelForm):
    password = forms.CharField(label="Password", widget=forms.PasswordInput)
    password2 = forms.CharField(label="Confirm Password", widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ("username", "email")

    def clean_password2(self):
        cd = self.cleaned_data
        if cd['password'] != cd['password2']:
            raise forms.ValidationError("两次输入的密码不一致！")
        return cd['password2']

class UserInfoForm(forms.ModelForm):
    class Meta:
        model = UserInfo
        fields = ("birth", "phone")

class UserDataForm(forms.ModelForm):
    class Meta:
        model = UserData
        fields = ("company", "profession", "address", "aboutme", "photo")

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ("email",)
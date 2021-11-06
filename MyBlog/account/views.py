from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from .forms import LoginForm, RegisterForm, UserInfoForm, UserDataForm, UserForm
from .models import UserInfo, UserData
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

# 用户登录
def user_login(request):
    if request.method == "POST":
        # 接收前台传来的表单
        login_form = LoginForm(request.POST)
        if login_form.is_valid():
            userdata = login_form.cleaned_data
            user = authenticate(username=userdata['username'], password=userdata['password'])
            if user:
                login(request, user)
                return HttpResponse("<h2>Welcome!</h2>")
            else:
                return HttpResponse("请输入正确的用户名或密码。")
        else:
            return HttpResponse("无效登录。")

    if request.method == "GET":
        login_form = LoginForm()
        return render(request, "account/login.html", {"form": login_form})

# 用户注册
def register(request):
    if request.method == "POST":
        # 接收前台传来的表单
        user_form = RegisterForm(request.POST)
        userinfo_form = UserInfoForm(request.POST)
        if user_form.is_valid() * userinfo_form.is_valid():
            new_user = user_form.save(commit=False)
            new_user.set_password(user_form.cleaned_data['password'])
            new_user.save()

            new_userinfo = userinfo_form.save(commit=False)
            new_userinfo.user = new_user
            new_userinfo.save()

            # 保存用户注册信息后同事在个人信息表中写入该用户的ID
            UserData.objects.create(user=new_user)

            return render(request, "account/reg_success.html")
        else:
            return render(request, "account/reg_error.html")
    else:
        register_form = RegisterForm()
        userinfo_form = UserInfoForm()
        return render(request, "account/register.html", {"form": register_form, "userinfo": userinfo_form})

# 用户资料查看
@login_required(login_url='/account/login/')
def myself(request):
    user = User.objects.get(username=request.user.username)
    userinfo = UserInfo.objects.get(user=user)
    userdata = UserData.objects.get(user=user)
    return render(request, "account/myself.html", {"user": user, "userdata": userdata, "userinfo": userinfo})

# 用户资料修改
@login_required(login_url='/account/login/')
def myself_edit(request):
    user = User.objects.get(username=request.user.username)
    userinfo = UserInfo.objects.get(user=user)
    userdata = UserData.objects.get(user=user)
    if request.method == "POST":
        user_form = UserForm(request.POST)
        userinfo_form = UserInfoForm(request.POST)
        userdata_form = UserDataForm(request.POST)

        if user_form.is_valid() * userinfo_form.is_valid() * userdata_form.is_valid():
            user_cd = user_form.cleaned_data
            userinfo_cd = userinfo_form.cleaned_data
            userdata_cd = userdata_form.cleaned_data
            print(user_cd["email"])
            user.email = user_cd['email']
            userinfo.birth = userinfo_cd['birth']
            userinfo.phone = userinfo_cd['phone']
            userdata.company = userdata_cd['company']
            userdata.profession = userdata_cd['profession']
            userdata.address = userdata_cd['address']
            userdata.aboutme = userdata_cd['aboutme']
            user.save()
            userinfo.save()
            userdata.save()
        return HttpResponseRedirect('/account/my-information')
    else:
        user_form = UserForm(instance=request.user)
        userinfo_form = UserInfoForm(initial={"birth": userinfo.birth, "phone": userinfo.phone})
        userdata_form = UserDataForm(initial={"company": userdata.company, "profession": userdata.profession,
                                              "address": userdata.address, "aboutme": userdata.aboutme})
        return render(request, "account/myself_edit.html",
                      {"user_form": user_form, "userinfo_form": userinfo_form, "userdata_form": userdata_form})

# 裁剪用户头像
@login_required(login_url='/account/login/')
def my_image(request):
    if request.method == 'POST':
        img = request.POST['img']
        userdata = UserData.objects.get(user=request.user.id)
        userdata.photo = img
        userdata.save()
        return HttpResponse("1")
    else:
        return render(request, 'account/imagecrop.html', )
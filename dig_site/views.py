from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import auth

# Create your views here.


def ex_info(request):
    return render(request, 'dig_site/ex_info.html')

def index(request):
    return render(request, 'dig_site/main.html')

def login(request):
    if(request.method =="POST"):
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('http://127.0.0.1:8000/')
        else:
            return render(request, 'dig_site/Login.html', {'error':'아이디 또는 비밀번호가 일치하지 않습니다.'})
    else:
        return render(request, 'dig_site/login.html')

def signup(request):
    if request.method == "POST":
        if request.POST['password1'] == request.POST['password2']:
            user=User.objects.create_user(request.POST['username'], password=request.POST['password1'])
            auth.login(request, user)
            return redirect('main')
    return render(request, 'dig_site/signup.html')

def logout(request):
    if request.method == "POST":
        auth.logouti(request)
        return redirect('main')
    return render(request, 'dig_site/signup.html')

def weather(request):
    return render(request, 'dig_site/weather.html')

def report(request):
    return render(request, 'dig_site/report.html')

def work_daily(request):
    return render(request, 'dig_site/work_daily.html')
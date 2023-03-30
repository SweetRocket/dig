from django.shortcuts import render

# Create your views here.


def ex_info(request):
    return render(request, 'dig_site/ex_info.html')

def index(request):
    return render(request, 'dig_site/main.html')


def login(request):
    return render(request, 'dig_site/login.html')

def signup(request):
    return render(request, 'dig_site/signup.html')

def weather(request):
    return render(request, 'dig_site/weather.html')

def report(request):
    return render(request, 'dig_site/report.html')

def work_daily(request):
    return render(request, 'dig_site/work_daily.html')
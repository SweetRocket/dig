from django.shortcuts import render

from .models import SiteInfo
from django.contrib.auth.models import User
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
    site = SiteInfo.objects.all()
    
    data = {
        'site': [ { 'id': s.pk, 'name': s.name } for s in site ],
        'workers': [ { 'id': w.pk, 'name': w.last_name + w.first_name } for w in User.objects.all() ]
    }
    
    return render(request, 'dig_site/work_daily.html', data)
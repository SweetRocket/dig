from django.shortcuts import render, redirect
from django.contrib import auth

from .models import SiteInfo, Report
from django.contrib.auth.models import User
from django.db.models import Q
# Create your views here.


def ex_info(request):
    return render(request, 'dig_site/ex_info.html')

def index(request):
    return render(request, 'dig_site/main.html')

def weather(request):
    return render(request, 'dig_site/weather.html')

def report(request):
    q = Q()
    
    if (a := request.GET.get('age', None)) is not None:
        q &= Q(age=a)
    
    if (r := request.GET.get('region', None)) is not None:
        q &= Q(region=r)

    if (s := request.GET.get('query', None)) is not None:
        q &= Q(name__icontains=s)

    reports = Report.objects.filter(q).all()
    
    return render(request, 'dig_site/report.html', {
        'reports': reports
    })

def work_daily(request):
    site = SiteInfo.objects.all()
    
    data = {
        'sites': [ { 'id': s.pk, 'name': s.name } for s in site ]
    }

    return render(request, 'dig_site/work_daily.html', data)
from django.shortcuts import render, redirect
from django.contrib import auth

from .models import SiteInfo, Report
from django.contrib.auth.models import User
from django.db.models import Q

from django.contrib.auth.decorators import login_required
# Create your views here.


def ex_info(request):
    return render(request, 'dig_site/ex_info.html')

def index(request):
    return render(request, 'dig_site/main.html')

def weather(request):
    return render(request, 'dig_site/weather.html')

def report(request):
    # django 쿼리를 사용하여 데이터를 가져옴
    q = Q()
    
    # 파라미터에 age가 있으면 age에 해당하는 데이터만 가져옴
    if (a := request.GET.get('age', None)) is not None:
        q &= Q(age=a)
    
    # 파라미터에 region이 있으면 region에 해당하는 데이터만 가져옴
    if (r := request.GET.get('region', None)) is not None:
        q &= Q(region=r)

    # 파라미터에 query가 있으면 query에 해당하는 데이터만 가져옴
    if (s := request.GET.get('query', None)) is not None:
        q &= Q(name__icontains=s)

    # 가져온 데이터를 쿼리에 맞게 가져옴
    reports = Report.objects.filter(q).all()
    
    # 가져온 데이터를 사용하여 렌더링
    return render(request, 'dig_site/report.html', {
        'reports': reports
    })

@login_required
def work_daily(request):
    # 사이트 전체 정보를 가져옴
    site = SiteInfo.objects.all()
    
    # 페이지에 맞게 데이터를 전달
    data = {
        'sites': [ { 'id': s.pk, 'name': s.name } for s in site ]
    }

    return render(request, 'dig_site/work_daily.html', data)

def dashboard(request):
    return render(request, 'dig_site/dashboard.html')
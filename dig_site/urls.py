from . import views
from django.urls import path

app_name = 'dig_site'

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login, name='login'),
    path('signup/', views.signup, name='signup'),
    path('weather/', views.weather, name='weather'),
    path('ex_info/', views.ex_info, name='ex_info'),
    path('report', views.report, name='report'),
    path('work_daily', views.work_daily, name='work_daily')
    
]
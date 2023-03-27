from . import views
from django.urls import path

app_name = 'dig_site'

urlpatterns = [
    path('', views.index),
    path('ex_info/', views.ex_info, name='ex_info'),
]
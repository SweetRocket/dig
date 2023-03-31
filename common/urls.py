from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from django.contrib import admin

app_name = 'common'

urlpatterns = [
    path('login/', auth_views.LoginView.as_view(template_name='common/login.html', ), name='login'),
    path('signup/', views.SignupView.as_view(), name='signup'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout')
]

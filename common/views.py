from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import auth

# Create your views here.
def login(request):
    if(request.method =="POST"):
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('http://127.0.0.1:8000/')
        else:
            return render(request, 'dig_site/Login.html', {'error':'username or password is incorrect'})
    else:
        return render(request, 'dig_site/Login.html')

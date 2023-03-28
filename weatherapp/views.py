from django.shortcuts import render
import requests
import datetime

# Create your views here.
def weather(request):
    return render(request, 'weatherapp/weather.html',)
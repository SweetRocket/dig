from django.shortcuts import render
import requests
import datetime

# Create your views here.
def weather(request):

    if 'city' in request.POST:
        city = request.POST['city']
    else:
        city = 'Seoul'
    
    appid = '558f5e94b3fdbc3f4a005005fda193bf'
    URL = 'https://api.openweathermap.org/data/2.5/weather'
    PARAMS = {'q':city, 'appid':appid, 'units':'metric', 'lang':'kr'}
    r = requests.get(url=URL, params=PARAMS)
    res = r.json()
    description = res['weather'][0]['description']
    icon = res['weather'][0]['icon']
    temp = res['main']['temp']

    day = datetime.date.today()

    return render(request, 'weatherapp/weather.html', {'description':description,
                                                    'icon':icon, 
                                                    'temp':temp, 
                                                    'day':day,
                                                    'city':city})

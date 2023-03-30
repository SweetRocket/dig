from . import views
from django.urls import path, include

app_name = 'api'

weather_urlpatterns = [
    path('hourly', views.weather.hourly, name='load'),
];

work_urlpatterns = [
    path('<date>', views.work.load, name='load'),
];

urlpatterns = [
    path('work/', include((work_urlpatterns, 'work'))),
    path('weather/', include((weather_urlpatterns, 'weather'))),
]

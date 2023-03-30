from . import views
from django.urls import path, include

app_name = 'api'

weather_urlpatterns = [
    path('hourly', views.weather.hourly, name='load'),
];

work_urlpatterns = [
    path('load', views.work.load, name='load'),
    path('new/<date>/<int:site>', views.work.new, name='new'),
    path('update/<int:id>', views.work.update, name='update'),
    
];

urlpatterns = [
    path('work/', include((work_urlpatterns, 'work'))),
    path('weather/', include((weather_urlpatterns, 'weather'))),
]

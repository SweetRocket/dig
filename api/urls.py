from . import views
from django.urls import path, include

app_name = 'api'

weather_urlpatterns = [
    path('hourly', views.weather.hourly, name='load'),
]

work_urlpatterns = [
    path('load', views.work.load, name='load'),
    path('workers', views.work.workers, name='workers'),
    path('new/<date>/<int:site>', views.work.new, name='new'),
    path('update/<int:id>', views.work.update, name='update'),
    path('recent', views.work.recent, name='recent'),
]

dashboard_urlpatterns = [
    path('video_equipment', views.dashboard.video_equipment, name='video_equipment'),
]

urlpatterns = [
    path('work/', include((work_urlpatterns, 'work'))),
    path('weather/', include((weather_urlpatterns, 'weather'))),
    path('dashboard/', include((dashboard_urlpatterns, 'dashboard'))),
]

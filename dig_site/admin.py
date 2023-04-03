from django.contrib import admin

from .models import Report, SiteInfo, SiteJoin, WorkHistory

# Register your models here.
@admin.register(Report)
class EventAdmin(admin.ModelAdmin):
    fields = ('name', 'age', 'region', 'image', 'url')
    
@admin.register(SiteInfo)
class SiteInfoAdmin(admin.ModelAdmin):
    fields = ('name', 'address', 'relic_era', 'area')
    
@admin.register(SiteJoin)
class SiteJoinAdmin(admin.ModelAdmin):
    fields = ('user', 'site')

@admin.register(WorkHistory)
class WorkHistoryAdmin(admin.ModelAdmin):
    fields = ('zone', 'note', 'workers', 'images', 'site', 'date')

from django.contrib import admin

from .models import Report

# Register your models here.
@admin.register(Report)
class EventAdmin(admin.ModelAdmin):
    fields = ('name', 'age', 'region', 'image', 'url')
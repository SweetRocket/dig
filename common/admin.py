from django.contrib import admin

from .models import Image, Employee, Profile, VideoEquipment

# Register your models here.
@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    fields = ('image',)
    
@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    fields = ('emp_id', 'joinable')
    
@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    fields = ('user', 'emp', 'sex', 'age', 'phone_number', 'address', 'emg_contact')
    
@admin.register(VideoEquipment)
class VideoEquipmentAdmin(admin.ModelAdmin):
    fields = ('name', 'device_address')
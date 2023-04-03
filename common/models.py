from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

from phonenumber_field.modelfields import PhoneNumberField


class PositionChoices(models.IntegerChoices):
    ADMIN = 0, 'Admin'
    MANAGER = 1, 'Manager'
    EMPLOYEE = 2, 'Employee'

# 프로필
class Profile(models.Model):
    # Django 기본 User 모델과 1:1, Cascade 관계 설정
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    
    # 직원 모델과의 1:1, Cascade 관계 설정
    emp = models.OneToOneField("Employee", on_delete=models.CASCADE, null=True, blank=True)

    # 성별
    sex = models.CharField(max_length=32)
    
    # 직급
    PositionChoices = models.PositiveIntegerField(choices=PositionChoices.choices, default=PositionChoices.EMPLOYEE)
    
    # 나이
    age = models.PositiveIntegerField(default=0)
    
    # 연락처
    phone_number = PhoneNumberField(region='KR') # type: ignore
    
    # 주소
    address = models.TextField(blank=True)
    
    # 비상 연락처
    emg_contact = PhoneNumberField(region='KR') # type: ignore

    def __str__(self):
        return self.user.username

    
# 전체 직원
class Employee(models.Model):
    # 직원 번호를 CharField, Primary Key로 설정
    emp_id = models.CharField(max_length=32, primary_key=True)
    
    # 가입 가능 여부 필드
    joinable = models.BooleanField(default=True)


# 로그
class Log(models.Model):
    # 발생 유저
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    
    # 내용
    content = models.TextField()
    
    # 생성(발생) 시간
    created_at = models.DateTimeField(auto_now_add=True)


# d이미지
class Image(models.Model):
    # 이미지 파일
    image = models.ImageField(upload_to='images/')
    
    def __str__(self):
        return self.image.name


# 영상 기기
class VideoEquipment(models.Model):
    # 영상 기기 주소
    device_address = models.TextField(null=False, blank=False)

    
# 유저 생성시 유저의 Profile도 생성 
@receiver(post_save, sender=User)
def update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    instance.profile.save()
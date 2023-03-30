from django.db import models
from django.contrib.auth.models import User

from common.models import Image

# 시대
class AgeChoices(models.IntegerChoices):
    PALEOLITHIC = 0, '구석기'
    NEOLITHIC = 1, '신석기'
    BRONZE = 2, '청동기'
    IRON = 3, '철기'


# 지역
class RegionChoices(models.IntegerChoices):
    SEOUL = 0, '서울'
    GYEONGGI = 1, '경기도'
    GANGWON = 2, '강원도'
    CHUNGCHONG_BUKDO = 3, '충청북도'
    CHUNGCHONG_NAMDO = 4, '충청남도'
    GYEONGSANG_BUKDO = 5, '경상북도'
    GYEONGSANG_NAMDO = 6, '경상남도'
    JEOLLA_BUKDO = 7, '전라북도'
    JEOLLA_NAMDO = 8, '전라남도'
    JEJU = 9, '제주도'


# 현장 정보
class SiteInfo(models.Model):
    # 현장 이름
    name = models.CharField(max_length=128)
    
    # 현장 주소
    address = models.TextField()
    
    # 유물 발굴 시대
    relic_era = models.TextField()

    # 면적
    area = models.FloatField()


# 현장 참여
class SiteJoin(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    site = models.ForeignKey(SiteInfo, on_delete=models.CASCADE)

    class Meta:
        unique_together = [("user", "site")]
        index_together = [("user", "site")]
        verbose_name_plural = "SiteJoin"
    
    def __str__(self):
        return f"{self.user} at {self.site}"


# 작업 일지
class WorkHistory(models.Model):
    # 유구 (구역)
    zone = models.CharField(max_length=128, blank=True)
    
    # 특이사항 
    note = models.TextField(blank=True)
    
    # 참여자
    workers = models.ManyToManyField(User)
    
    # 이미지
    images = models.ManyToManyField(Image, blank=True)
    
    # 현장
    site = models.ForeignKey(SiteInfo, on_delete=models.SET_NULL, null=True)
    
    # 작업 날짜
    date = models.DateField()
    
    # 갱신 일자
    updated_at = models.DateTimeField(auto_now=True)


# 발굴 보고서
class Report(models.Model):
    # 시대
    age = models.PositiveIntegerField(choices=AgeChoices.choices)
    
    # 지역
    region = models.PositiveIntegerField(choices=RegionChoices.choices)
    
    # 이름
    name = models.CharField(max_length=128)
    
    # 표지 이미지
    image = models.ForeignKey(Image, on_delete=models.SET_NULL, null=True)
    
    # 보고서 링크
    url = models.URLField()

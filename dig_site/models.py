from django.db import models
from django.contrib.auth.models import User

# 현장 정보
class SiteInfo(models.Model):
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
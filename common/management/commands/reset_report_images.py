from django.core.management.base import BaseCommand
from dig_site.models import WorkHistory

import pathlib

import logging

logger = logging.getLogger("cli")


class Command(BaseCommand):
    help = "Bulk reset report images"

    # handle 함수를 사용하여 명령어가 실행될 때 실행될 코드를 작성
    def handle(self, *args, **options):
        # 작업 일지
        wh = WorkHistory.objects.all()
        
        for h in wh:
            # 이미지 설정 해제
            h.images.set([])
            
            # 저장
            h.save()
            
            logger.info(f'Cleared images for {h} ({h.pk})')

        logger.info('Done')

import argparse
from django.core.management.base import BaseCommand, CommandError
from common.models import User
from dig_site.models import SiteInfo, SiteJoin

import pathlib
import pandas as pd
import numpy as np
import random

import logging

logger = logging.getLogger("cli")

class Command(BaseCommand):
    help = "Bulk create sites"
    
    # add_argument 를 사용하여 명령어에 사용할 파라미터를 추가
    def add_arguments(self, parser) -> None:
        parser.add_argument('max_site', type=int)
    
    # handle 함수를 사용하여 명령어가 실행될 때 실행될 코드를 작성
    def handle(self, *args, **options):
        # 랜덤으로 현장 생성
        for i in range(options['site']):
            s, _ = SiteInfo.objects.get_or_create(
                pk = i,
                defaults={
                    'name': f'현장 {i}',
                    'address': f'주소 {i}',
                    'relic_era': random.choice(['']),
                    'area': float(999999.99)
                }
            )
            logger.info(f'Created site {s.name}')
            
            
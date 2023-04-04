from django.core.management.base import BaseCommand, CommandError
from common.models import User
from dig_site.models import SiteInfo, SiteJoin, WorkHistory

import pathlib
import pandas as pd
import random

import logging

logger = logging.getLogger("cli")


class Command(BaseCommand):
    help = "Bulk create work histories"

    # add_argument 를 사용하여 명령어에 사용할 파라미터를 추가
    def add_arguments(self, parser) -> None:
        parser.add_argument('file', type=pathlib.Path)
        parser.add_argument('seed', type=int, default=42, nargs='?')

    # handle 함수를 사용하여 명령어가 실행될 때 실행될 코드를 작성
    def handle(self, *args, **options):
        # 랜덤 시드 설정
        random.seed(options['seed'])

        # 파일 경로를 가져옴
        source = options['file']

        # 엑셀 파일을 읽어옴
        if source.suffix in ['.xls', '.xlsx']:
            df = pd.read_excel(source)
        # csv 파일을 읽어옴
        elif source.suffix in ['.csv']:
            df = pd.read_csv(source)
        # 지원하지 않는 파일 형식일 경우 에러를 발생시킴
        else:
            raise CommandError('Unsupported file type')

        # 자체 인덱스 설정
        df = df.set_index('번호')

        # 시간 변환
        df['날짜'] = pd.to_datetime(df['날짜'], format='%Y%m%d')

        # 랜덤 현장용 리스트
        sites = list(SiteInfo.objects.all())

        for k, d in df.to_dict('index').items():
            # 날짜
            date = d['날짜'].date().strftime('%Y-%m-%d')

            # 이름 분리
            first_name = d['작업 근로자'][1:]
            last_name = d['작업 근로자'][0]

            # 현장
            site = random.choice(sites)

            # 유저
            try:
                user = User.objects.get(first_name=first_name, last_name=last_name)
            except:
                logger.error(f'User {last_name}{first_name} not found')
                continue

            # 현장 참여
            sj, _ = SiteJoin.objects.get_or_create(site=site, user=user)
            sj.save()

            # 작업 일지
            wh, _ = WorkHistory.objects.get_or_create(
                pk=k,
                defaults={
                    'zone': d['유구'],
                    'note': d['출토 유물'],
                    'site': site,
                    'date': date,
                }
            )

            # 작업 일지 재설정
            wh.zone = d['유구']
            wh.note = d['출토 유물']
            wh.workers.set([user])
            wh.images.set([])
            wh.site = site
            wh.date = date

            # 저장
            wh.save()

            logger.info(f'Created {wh} ({user}, {site})')

        logger.info('Done')

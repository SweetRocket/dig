from django.core.management.base import BaseCommand, CommandError
from dig_site.models import Report, Image, AgeChoices, RegionChoices
from django.core.files.base import ContentFile

import requests
from bs4 import BeautifulSoup

import pathlib
import pandas as pd
import random

import logging

logger = logging.getLogger("cli")

class Command(BaseCommand):
    help = "Bulk create reports"
    
    # add_argument 를 사용하여 명령어에 사용할 파라미터를 추가
    def add_arguments(self, parser) -> None:
        parser.add_argument('file', type=pathlib.Path)
    
    # handle 함수를 사용하여 명령어가 실행될 때 실행될 코드를 작성
    def handle(self, *args, **options):
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
        df.set_index('번호', inplace=True)
        
        # 등록일을 datetime 형식으로 변환
        df['등록일'] = pd.to_datetime(df['등록일'])
        
        # 루프를 돌면서 데이터를 생성
        for k, d in df.to_dict('index').items():
            name = d['제목']
            url = d['링크']
            
            # 이미 존재하는 데이터는 스킵
            if Report.objects.filter(name=name).exists():
                logger.info(f'{name} already exists, skipping {k}...')
                continue
            
            # 데이터를 가져옴
            resp = requests.get(url)
            if resp.status_code != 200:
                logger.error(f'Failed to get {url}, skipping {k}...')
                continue
            
            # 데이터를 파싱
            html = resp.text
            soup = BeautifulSoup(html, 'html.parser')
            
            # 이미지를 가져옴
            img = soup.select_one('.board_view .b_content img')

            # 이미지가 없는 경우 스킵            
            if img is None:
                logger.error(f'Failed to get image tag on {k}, skipping...')
                continue
            
            # 이미지의 경로를 가져옴
            img_src = img.attrs['src']
            
            # 상대 경로인 경우 절대 경로로 변환
            if img_src.startswith('/'):
                baseurl = url[:url.find('/', 8)]
                img_src = baseurl + img_src
                
            # 이미지를 가져옴
            resp = requests.get(img_src)
            if resp.status_code != 200:
                logger.info(f'Failed to get image on {k}, skipping...')
                continue
            
            # 이미지의 확장자를 가져옴
            ext = resp.headers['Content-Type'].split('/')[-1]
            ext = ext.split(';')[0]
            
            # 파일명 설정
            img_name = f"report_{name.replace(' ', '_')}.{ext}"
            
            # 이미지 데이터를 가져옴
            img_data = resp.content
            
            # 장고 파일을 사용하여 이미지를 생성
            dj_file = ContentFile(img_data, name=img_name)
            
            # 이미지를 저장
            img = Image.objects.create(image=dj_file)
            
            # 카테고리 설정
            # FIXME: 카테고리가 랜덤임
            age = random.choice(AgeChoices.values)
            region = random.choice(RegionChoices.values)
            
            # 보고서를 생성
            report = Report.objects.create(
                name = name,
                age = age,
                region = region,
                image = img,
                url = url,
            )
            
            # 저장
            report.save()
            
            logger.info(f'{k} {name} created')
            
        logger.info('Done')
        
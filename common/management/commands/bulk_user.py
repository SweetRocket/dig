from django.core.management.base import BaseCommand, CommandError
from common.models import Profile, User, PositionChoices, Employee
from django.contrib.auth.hashers import make_password

import pathlib
import pandas as pd
import numpy as np
import random

import logging

logger = logging.getLogger("cli")

class Command(BaseCommand):
    help = "Bulk create users"
    
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
        
        # 직급 변환
        conv_position = {
            name: num for num, name in PositionChoices.choices
        }
        
        for k, d in df.to_dict('index').items():
            # 이름 분리
            last_name = d['직원이름'][0]
            first_name = d['직원이름'][1:]

            # 직급
            position = d['직급']
            position = conv_position[position]
            
            # FIXME: 데이터 고쳐야함
            if position == PositionChoices.SUPPORTER or position == PositionChoices.SEMI_RESEARCHER:
                position = random.randint(4, 6)
            
            # 나이
            age = d['나이']
            
            # FIXME: 데이터 고쳐야함
            if np.isnan(age):
                age = random.randint(20, 70)
            
            # 성별
            sex = d['성별']
            
            # 연락처
            phone_number = d['연락처']
            
            # 이메일
            address = d['주소']
            
            # 비상연락처
            emg_contact = d['비상연락처']
            
            # 아이디
            username = d['아이디']
            
            # 비밀번호
            password = d['비밀번호']
            
            # 직원번호
            emp = d['직원번호']
            
            # 직원번호로 직원 생성
            e, _ = Employee.objects.get_or_create(
                emp_id = emp,
                defaults = {
                    'joinable': True,
                }
            )
            
            # 직원 재설정
            e.joinable = True
            
            e.save()
            
            # 유저 생성
            u, _ = User.objects.get_or_create(
                username = username,
                defaults= {
                    'first_name': first_name,
                    'last_name': last_name,
                    'password': make_password(password),
                }
            )
            
            # 유저 재설정
            u.first_name = first_name
            u.last_name = last_name
            u.password = make_password(password)
            
            u.save()
            
            # 프로필 생성          
            p, _ = Profile.objects.get_or_create(
                user = u,
                defaults = {
                    'user': u,
                    'emp': e,
                    'PositionChoices': position,
                    'age': age,
                    'phone_number': phone_number,
                    'address': address,
                    'emg_contact': emg_contact,
                    'sex': sex
                }
            )
            
            # 프로필 재설정
            p.user = u
            p.emp = e
            p.PositionChoices = position
            p.age = age
            p.phone_number = phone_number
            p.address = address
            p.emg_contact = emg_contact
            p.sex = sex
            
            p.save()
            
            logger.info(f'Created user {u}')
        logger.info('Done')
        
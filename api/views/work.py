from django.shortcuts import render
from django.views.decorators.csrf import requires_csrf_token
from django.http import JsonResponse
from django.db.models import Q
from django.core.files.base import ContentFile


from common.models import Image
from dig_site.models import SiteInfo, SiteJoin, WorkHistory

from ..utils import work_to_dict, login_required_json
from django.contrib.auth.models import User

import base64
import datetime
import json
import random


@requires_csrf_token
@login_required_json
def load(request):
    """
    작업 일지 데이터를 가져오는 API
    """

    # query 에서 데이터를 가져옴
    date = request.GET.get(
        'date', datetime.datetime.now().strftime('%Y-%m-%d'))
    site = request.GET.get('site', None)

    # site 가 있으면 site 에 해당하는 데이터만 가져옴
    q = Q(date=datetime.datetime.strptime(date, '%Y-%m-%d'))
    if site is not None:
        q &= Q(site=site)

    # 작업 일지 데이터를 쿼리에 맞게 가져옴
    works = WorkHistory.objects.filter(q).prefetch_related(
        'images', 'workers').order_by('pk').all()

    ret = []

    # 가져온 데이터를 dict 형태로 변환
    for work in works:
        ret.append(work_to_dict(work))

    # dict 형태로 변환된 데이터를 json 형태로 반환
    return JsonResponse({
        'status': 'ok',
        'result': ret
    })


@requires_csrf_token
@login_required_json
def new(request, date, site):
    """
    새로운 작업 일지 데이터를 생성하는 API
    """

    # 작업 일지 데이터를 생성
    work = WorkHistory.objects.create(
        date=datetime.datetime.strptime(date, '%Y-%m-%d'),
        site=SiteInfo.objects.get(pk=site)
    )

    # 작업일지 저장
    work.save()

    # dict 형태로 변환된 데이터를 json 형태로 반환
    return JsonResponse({
        'status': 'ok',
        'result': work_to_dict(work),
    })


@requires_csrf_token
@login_required_json
def update(request, id):
    """
    작업 일지 데이터를 수정하는 API
    """

    # POST 요청이 아니면 에러
    if request.method != 'POST':
        return JsonResponse({
            'status': 'error',
            'error': 'invalid method'
        }, status=400)

    # id 에 해당하는 작업 일지 데이터를 가져옴
    try:
        work = WorkHistory.objects.get(pk=id)
    # 작업 일지 데이터가 없으면 에러
    except:
        return JsonResponse({
            'status': 'error',
            'error': 'invalid work id'
        }, status=400)

    # POST 요청의 body 를 json 형태로 변환
    data = json.loads(request.body)

    # zone이 있으면 zone을 수정
    if (z := data.get('zone', None)) is not None:
        work.zone = z

    # date가 있으면 date를 수정
    if (w := data.get('workers', None)) is not None:
        # workers에서 -1 필터링
        w = list(set(map(int, w)))
        if -1 in w:
            w.remove(-1)
        
        # workers 가 없으면 모든 작업자 삭제
        if len(w) == 0:
            work.workers.clear()
        # workers 가 있으면 해당하는 작업자만 추가
        else:
            workers = [User.objects.get(pk=int(i)) for i in w]
            work.workers.set(workers)

    # date가 있으면 date를 수정
    if (n := data.get('note', None)) is not None:
        work.note = n

    # image가 있으면 image를 수정
    if (i := data.get('image', None)) is not None:
        # 필요없는 부분을 제거
        fmt, img = i.split(';base64,')
        ext = fmt.split('/')[-1]

        # 중복 방지를 위해 랜덤한 id를 생성
        rand_id = random.randint(0, 1000000000)
        rand_id = f'{rand_id:010d}'
        
        # 파일명
        name = f'{id}_{rand_id}.{ext}'

        # 디코딩하여 image를 생성
        data = ContentFile(base64.b64decode(img), name=name)

        # 해당 image를 필드에 추가
        image = Image.objects.create(image=data)
        work.images.add(image)

    # 작업일지 저장
    work.save()

    # dict 형태로 변환된 최신 데이터를 json 형태로 반환
    return JsonResponse({
        'status': 'ok',
        'result': work_to_dict(work)
    })


@requires_csrf_token
@login_required_json
def workers(request):
    """
    작업자 목록을 가져오는 API
    """

    # site 가 없으면 작업에 참여한 모든 작업자를 가져와 반환
    if (w := request.GET.get('site', None)) is None:
        all_workers = User.objects.filter(
            sitejoin__isnull=False).distinct().all()
        return JsonResponse({
            'status': 'ok',
            'result': [{
                'id': w.pk,
                'name': f'{w.last_name}{w.first_name}'
            } for w in all_workers]})

    # site 가 있으면 해당 site 에 참여한 작업자를 가져오는걸 시도
    try:
        site = SiteInfo.objects.get(pk=w)
    except:
        # site 가 없으면 에러
        return JsonResponse({
            'status': 'error',
            'error': 'invalid site id'
        })

    # site 에 참여한 작업자를 가져옴
    site_join = SiteJoin.objects.filter(site=site).select_related('user').all()

    # 작업자 목록을 dict 형태로 변환
    workers = [{
        'id': sj.user.pk,
        'name': f'{sj.user.last_name}{sj.user.first_name}'
    } for sj in site_join]

    # dict 형태로 변환된 데이터를 json 형태로 반환
    return JsonResponse({
        'status': 'ok',
        'result': workers
    })


@requires_csrf_token
@login_required_json
def recent(request):
    """
    최근 작업 일지 데이터를 가져오는 API
    """

    # 최근 10개의 작업 일지 데이터를 가져옴
    works = WorkHistory.objects.all().order_by('-updated_at')[:10]

    # dict 형태로 변환된 데이터를 json 형태로 반환
    return JsonResponse({
        'status': 'ok',
        'result': [work_to_dict(w) for w in works]
    })

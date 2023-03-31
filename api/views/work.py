from django.shortcuts import render
from django.views.decorators.csrf import requires_csrf_token
from django.http import JsonResponse
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile


from common.models import Image
from dig_site.models import SiteInfo, SiteJoin, WorkHistory

from ..utils import work_to_dict
from django.contrib.auth.models import User

import base64
import datetime
import json

@requires_csrf_token
#@login_required
def load(request):
    date = request.GET.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))
    site = request.GET.get('site', None)
    
    q = Q(date=datetime.datetime.strptime(date, '%Y-%m-%d'))
    if site is not None:
        q &= Q(site=site)
    
    works = WorkHistory.objects.filter(q).prefetch_related('images', 'workers').order_by('pk').all()
     
    ret = []
    
    for work in works:
        ret.append(work_to_dict(work))
            
    return JsonResponse({
        'status': 'ok',
        'result': ret
    })

@requires_csrf_token
#@login_required
def new(request, date, site):
    work = WorkHistory.objects.create(
        date=datetime.datetime.strptime(date, '%Y-%m-%d'),
        site=SiteInfo.objects.get(pk=site)
    )

    work.save()
    
    return JsonResponse({
        'status': 'ok',
        'result': work_to_dict(work),
    })
    
@requires_csrf_token
#@login_required
def update(request, id):
    if request.method != 'POST':
        return JsonResponse({
            'status': 'error',
            'error': 'invalid method'
        }, status=400)
    
    work = WorkHistory.objects.get(pk=id)
    if work is None:
        return JsonResponse({
            'status': 'error',
            'error': 'invalid work id'
        }, status=400)
        
    data = json.loads(request.body)
    
    if (z := data.get('zone', None)) is not None:
        work.zone = z
    
    if (w := data.get('workers', None)) is not None:
        if len(w) == 0 or (len(w) == 1 and (w[0] == -1 or w[0] == '-1')):
            work.workers.clear()
        else:
            workers = [User.objects.get(pk=int(i)) for i in w]
            work.workers.set(workers)
    
    if (n := data.get('note', None)) is not None:
        work.note = n
    
    if (i := data.get('image', None)) is not None:
        fmt, img = i.split(';base64,')
        ext = fmt.split('/')[-1]
        data = ContentFile(base64.b64decode(img), name=f'{id}.{ext}')
        
        image = Image.objects.create(image = data)
        work.images.add(image)
    
    work.save()
    
    return JsonResponse({
        'status': 'ok',
        'result': work_to_dict(work)
    })
    

@requires_csrf_token
#@login_required
def workers(request):
    if (w := request.GET.get('site', None)) is None:
        all_workers = User.objects.filter(sitejoin__isnull=False).distinct().all()
        return JsonResponse({
            'status': 'ok',
            'result': [{
                'id': w.pk,
                'name': w.get_full_name()
            } for w in all_workers]})

    try:
        site = SiteInfo.objects.get(pk=w)
    except:
        return JsonResponse({
            'status': 'error',
            'error': 'invalid site id'
        })
    
    site_join = SiteJoin.objects.filter(site=site).select_related('user').all()
        
    workers = [{
        'id': sj.user.pk,
        'name': sj.user.get_full_name()
    } for sj in site_join ]
    
    return JsonResponse({
        'status': 'ok',
        'result': workers
    })

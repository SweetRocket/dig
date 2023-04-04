from django.views.decorators.csrf import requires_csrf_token
from django.http import JsonResponse
from django.db.models import Q

from common.models import VideoEquipment

@requires_csrf_token
def video_equipment(request):
    """
    비디오 장비 데이터를 가져오는 API
    """
    
    # 비디오 장비 데이터를 가져옴
    video_equipment = VideoEquipment.objects.all().order_by('pk')

    ret = []

    # 가져온 데이터를 dict 형태로 변환
    for video in video_equipment:
        ret.append({
            'id': video.pk,
            'name': video.name,
            'device_address': video.device_address,
        })

    # dict 형태로 변환된 데이터를 json 형태로 반환
    return JsonResponse({
        'status': 'ok',
        'result': ret
    })



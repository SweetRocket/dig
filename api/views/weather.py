from ..weather_api import request_hourly
from django.http import JsonResponse
from django.views.decorators.cache import cache_page

# 30분 캐시
@cache_page(60 * 30)
def hourly(request):
    """
    기상청 API를 이용한 시간별 초단기 예보를 가져오는 API
    """
    
    # 기상청 자체 좌표계 nx, ny 를 사용
    nx = request.GET.get('nx', 60)
    ny = request.GET.get('ny', 125)

    # 기상청 API 를 이용해 데이터를 가져옴
    data = request_hourly(nx, ny)
    
    # 데이터가 없으면 서버 에러
    if data is None:
        return JsonResponse({ 'status': 'error', 'result': 'no data' }, status=500)
    
    # 데이터가 있으면 데이터를 가공
    items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
    ret = {}
    
    # 데이터를 가공
    for i in items:
        # 날짜와 시간을 합쳐서 키로 사용
        date = i['fcstDate'] + 'T' + i['fcstTime']
        
        # 날짜 키가 없으면 생성
        if ret.get(date, None) is None:
            ret[date] = {}

        try:
            # 값이 숫자면 float 형으로 변환
            value = float(i['fcstValue'])
        except:
            # 값이 숫자가 아니면 그대로 사용
            value = i['fcstValue']
        
        # 카테고리를 키로 사용
        ret[date][i['category']] = value
    
    # 가공된 데이터를 json 형태로 반환
    resp = JsonResponse({ 'status': 'ok', 'result': ret }, json_dumps_params={'ensure_ascii': False, 'indent': 2})
    # resp["Access-Control-Allow-Origin"] = "*"
    # resp["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    # resp["Access-Control-Max-Age"] = "1000"
    # resp["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return resp

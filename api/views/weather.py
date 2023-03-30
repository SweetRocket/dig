from ..weather_api import request_hourly
from django.http import HttpResponse, HttpResponseServerError, JsonResponse
from django.views.decorators.cache import cache_page

@cache_page(60 * 30)
def hourly(request):
    nx = request.GET.get('nx', 60)
    ny = request.GET.get('ny', 125)

    data = request_hourly(nx, ny)
    
    if data is None:
        return HttpResponseServerError('서버 에러')
    
    items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
    ret = {}
    
    for i in items:
        print(i)
        date = i['fcstDate'] + 'T' + i['fcstTime']
        if ret.get(date, None) is None:
            ret[date] = {}

        try:
            value = float(i['fcstValue'])
        except:
            value = i['fcstValue']
        
        ret[date][i['category']] = value
    
    resp = JsonResponse({ 'status': 'ok', 'result': ret }, json_dumps_params={'ensure_ascii': False, 'indent': 2})
    # resp["Access-Control-Allow-Origin"] = "*"
    # resp["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    # resp["Access-Control-Max-Age"] = "1000"
    # resp["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return resp

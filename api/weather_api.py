from django.core.cache import cache
import requests
import datetime


# 기상청 API 호출
def request_hourly(nx, ny):
    api_url = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"

    # 정각 즈음에는 발표가 없을수 있어 1시간 전 데이터를 가져옴
    now = datetime.datetime.now() - datetime.timedelta(hours=1)
    base_date = now.strftime('%Y%m%d')
    base_time = now.strftime('%H%M')

    data_param = {
        'base_date': base_date,
        'base_time': base_time,
        'nx': nx,
        'ny': ny
    }
    
    # api의 무차별 소모를 방지하기 위해 캐시를 사용
    cache_key = f'weather_data_{base_date}_{base_time}_{nx}_{ny}'    
    if (json := cache.get(cache_key, None)) is not None:
        return json

    # api 호출
    query = {
        'serviceKey': '9M2t/mX0iuKLvY/YfROagn0qPkwmDYvlu3cH1GWAIKhpEGl4vs+s3kz+xMkiwsr+fYQs3Q0lZQMbLkZ6nr5wQA==',
        'pageNo': 1,
        'numOfRows': 1000,
        'dataType': 'JSON',
        **data_param
    }

    r = requests.get(api_url, query)
    
    # 서버가 200 OK를 반환하지 않으면 None을 반환
    if r.status_code != 200:
        return None
    
    json = r.json()
    
    # 서버에서 에러가 발생하면 None을 반환
    if json.get('response', {}).get('header', {}).get('resultCode') != '00':
        return None

    cache.set(cache_key, json)
    return 
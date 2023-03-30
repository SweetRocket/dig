from django.shortcuts import render
from django.views.decorators.csrf import requires_csrf_token
from django.http import JsonResponse


@requires_csrf_token
def load(request, date):
    return JsonResponse({
        'status': 'ok',
        'result': [
            {
                'id': 1,
                'name': 'test1',
                'date': date,
                'images': [],
                'note': 'test1',
                'workers': [-1, -1]
            }
        ]
    })
    
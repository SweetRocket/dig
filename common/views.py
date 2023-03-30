from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
from django.http.response import HttpResponseForbidden
from django.contrib import auth
from django.views import generic
from django.urls import reverse

from common.forms import SignupForm
from common.models import Employee

# Create your views here.

class SignupView(generic.CreateView):
    form_class = SignupForm
    template_name = 'common/signup.html'

    def form_valid(self, form):
        # 사원 id 검증
        emp_id = form.cleaned_data['emp_id']
        try:
            emp = Employee.objects.get(emp_id=emp_id)
        except Employee.DoesNotExist:
            emp = None

        # 사원이 없거나, 가입이 불가능한 경우
        if emp is None or emp.joinable is not True:
            return self.render_to_response({'error': 'Invalid Employee', 'form': form })

        # form 저장
        form.save()

        # 자동 로그인
        username = form.cleaned_data['username']
        raw_password = form.cleaned_data['password1']
        user = authenticate(username=username, password=raw_password)
        login(self.request, user)

        return super().form_valid(form)

    def get_success_url(self) -> str:
        return reverse('dig_site:index')

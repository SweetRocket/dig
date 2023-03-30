from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class SignupForm(UserCreationForm):
    emp_id = forms.CharField(label="사원번호")
    name = forms.CharField(label='이름')

    class Meta:
        model = User
        fields = ("username", "password1", "password2", "emp_id", "name")
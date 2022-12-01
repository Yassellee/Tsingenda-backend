from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib import auth
from django.contrib.auth.hashers import check_password
from .models import *
import json

def gen_response(code: int, data):
    return JsonResponse({"code": code, "data": data}, status=code)



# Create your views here.
def login(request):
    if request.method == 'GET':
        return gen_response(200, {
            'status': 'request_error',
            'detail': 'invalid method get'
        })
    data = {'action':request.POST.get("action"),'username':request.POST.get("username"),'password':request.POST.get("password"),'old_password':request.POST.get("old_password"),'new_password':request.POST.get("new_password")}
    try:   
        action = data['action']
        if action == 'login':
            username = data['username']
            password = data['password']
            try:
                user = User.objects.get(name=username)
            except:
                return gen_response(
                    200, 
                    {
                        'status': 'login_error',
                        'detail': 'no user found'
                    }
                )
            django_user = user.django_user
            if check_password(password, django_user.password):
                auth.login(request, django_user) 
                return gen_response(
                    200,
                    {
                        'name': username,
                    }
                )
            else:
                return gen_response(
                    200,
                    {
                        'status': 'login_fail',
                        'detail': 'wrong password'
                    }
                )
        elif action == 'register':
            username = data['username']
            password = data['password']
            try:
                User_django.objects.get(username=username)
            except:
                django_user = User_django.objects.create_user(
                    username=username,
                    password=password,
                    is_superuser=False,
                    is_active=True,
                )
                user = User.objects.create(
                    name=username,
                    django_user=django_user,
                )
                return gen_response(
                    200,
                    {}
                )
            else:
                return gen_response(
                    200,
                    {
                        'status': 'register_error',
                        'detail': 'user already exists'
                    }
                )
        elif action == 'change_password':
            username = data['username']
            # old_password = data['old_password']
            new_password = data['new_password']
            try:
                user = User.objects.get(name=username)
            except:
                return gen_response(
                    200, 
                    {
                        'status': 'login_error',
                        'detail': 'no user found'
                    }
                )
            django_user = user.django_user
            # if check_password(old_password, django_user.password):
            user.django_user.set_password(new_password)
            return gen_response(
                    200,
                    {}
            )

        elif action == 'logout':
            username = data['username']
            try:
                user = User.objects.get(username)
            except:
                return gen_response(
                    200, 
                    {
                        'status': 'logout_error',
                        'detail': 'no user found'
                    }
                )
            django_user = user.django_user
            auth.logout(request, django_user)
        else:
            return gen_response(
                200, 
                {
                    'status': 'request_error',
                    'detail': 'unexpected request format'
                }
            )
    except KeyError:
        return gen_response(
            200, 
            {
                'status': 'request_error',
                'detail': 'unexpected request format'
            }
        )


def raw_text(request):
    if request.method == 'get':
        return gen_response(200, {
            'status': 'request_error',
            'detail': 'invalid method get'
        })
    body = json.loads(request.body)
    try:
        data = body['data']
        assert isinstance(data, list)
        response = [None] * len(data)
        for i, text in enumerate(data):
            # TODO
            is_agenda = True
            confidence = 1.0
            response[i] = {
                'is_agenda': is_agenda,
                'confidence': confidence
            }
        return gen_response(
            200,
            response
        )
            
    except (KeyError, AssertionError):
        return gen_response(
            200, 
            {
                'status': 'request_error',
                'detail': 'unexpected request format'
            }
        )
       

def image(request):
    if request.method == 'GET':
        return gen_response(
            200,
            {
                'status': 'request_error',
                'detail': 'invalid method get'
            }
        )

    body = json.loads(request.body)
    try:
        data = body['data']
        assert isinstance(data, list)
        response = [None] * len(data)
        for i, image in enumerate(data):
            # TODO
            raw_text = 'raw_text'
            is_agenda = True
            confidence = 1.0
            response[i] = {
                'raw_text': raw_text,
                'is_agenda': is_agenda,
                'confidence': confidence
            }
        return gen_response(
            200,
            response
        )
            
    except (KeyError, AssertionError):
        return gen_response(
            200, 
            {
                'status': 'request_error',
                'detail': 'unexpected request format'
            }
        )    

def feedback(request):
    if request.method == 'GET':
        return gen_response(
            200,
            {
                'status': 'request_error',
                'detail': 'invalid method get'
            }
        )
    body = json.loads(request.body)
    try:
        data = body['data']
        assert isinstance(data, list)
        for item in data:
            text = item['text']
            is_agenda = item['is_agenda']
            is_agenda = bool(is_agenda)
            # TODO
            pass

        return gen_response(
            200, {}
        )

    except:
        return gen_response(
            200, 
            {
                'status': 'request_error',
                'detail': 'unexpected request format'
            }
        )    
    
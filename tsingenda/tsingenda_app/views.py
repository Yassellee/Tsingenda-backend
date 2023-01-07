import base64
import os
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib import auth
from django.contrib.auth.hashers import check_password
from .models import *
import json
import logging
import torch
import time

from src.Classifier import train as src_train
from src.Classifier import predict as src_predict
from src.Classifier import confidence as src_confidence
from src.utils.ner_extractor import NERExtractor
from src.utils.ocr_extractor import OCRExtractor
from src.utils.text_summarizer import summarizer
from sys import argv
from src.Classifier.main import conf_model_base, conf_data_base


logger = logging.getLogger('django')
min_ocr_len = 10
t = time.time()
if 'runserver' in argv:
    args = src_predict.get_args()
    # model = src_predict.get_model(args)
    tokenizer = src_predict.get_tokenizer(args)
    ner = NERExtractor()
    ocr = OCRExtractor(gpu_state=True)
    titler = summarizer()

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
    # print('login:', data)
    try:   
        action = data['action']
        if action == 'login':
            username = data['username']
            password = data['password']
            try:
                django_user = User_django.objects.get(username=username)
            except:
                return gen_response(
                    200, 
                    {
                        'status': 'login_error',
                        'detail': 'no user found'
                    }
                )
            # django_user = user.django_user
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
                conf_param = ConfParam.objects.create(
                    model_dict=os.path.join(conf_model_base, '{}_conf_model'.format(username))
                )
                user = User.objects.create(
                    name=username,
                    django_user=django_user,
                    conf_param=conf_param,
                    conf_path=os.path.join(conf_data_base, '{}_conf_data'.format(username))
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
                user = User.objects.get(name=username)
            except:
                return gen_response(
                    200, 
                    {
                        'status': 'logout_error',
                        'detail': 'no user found'
                    }
                )
            django_user = user.django_user
            auth.logout(request)
            return gen_response(200, {})
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
    # print('raw_text:', body)
    # body = request.GET
    try:
        user: User = request.user.user
    except:
        return gen_response(
            200, {
                'status': 'request_error',
                'detail': 'user not login'
            }
        )
    user_conf_param = user.conf_param.model_dict
    user_conf_model = src_confidence.get_model(user_conf_param)
    try:
        data = body['data']
        assert isinstance(data, list)
        model = src_predict.get_model(args)
        conf = src_predict.predict(args, model, tokenizer, data)
        # conf = torch.Tensor([x['prob'] for x in conf])
        conf = [x['prob'] if int(x['label']) else 1 - x['prob'] for x in conf]
        output = src_confidence.predict(user_conf_model, conf)
        ner_output = ner.parse(data)
        title = titler.summarize(data)

        response = [None] * len(data)
        for i, text in enumerate(data):
            # TODO
            new_agenda_data = AgendaData.objects.create(
                raw_text=text,
                output = conf[i] > 0.5
            )
            new_conf_data = ConfData.objects.create(
                conf=conf[i],
                output=int(output[i]),
                user=user,
            )
            is_agenda = bool(conf[i] > 0.5)
            confidence = conf[i]
            if 'CCF' in text: output[i] = False
            dates = ner_output[i]['date']
            dates = [x for x in dates if x[1] is not None]
            times = ner_output[i]['time']
            times = [x for x in times if x[1] is not None]
            locations = ner_output[i]['location']
            response[i] = {
                'id': new_agenda_data.id,
                'conf_id': new_conf_data.id,
                'dates': dates,
                'times': times,
                'locations': locations,
                'title': title[i],
                'raw_text': text,
                'is_agenda': is_agenda,
                'confidence': confidence,
                'confidence_high': bool(output[i])
            }
        [print('response[{}]'.format(i), response[i]) for i in range(len(response))]
        # [logger.info('response[{}]:{}'.format(i, response[i])) for i in range(len(response))]
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
    logger.info('received image')
    body = json.loads(request.body)
    # print('image:', body)
    # body = request.GET
    try:
        user: User = request.user.user
    except:
        return gen_response(
            200, {
                'status': 'request_error',
                'detail': 'user not login'
            }
        )
    user_conf_param = user.conf_param.model_dict
    user_conf_model = src_confidence.get_model(user_conf_param)
    
    try:
        data = body['data']
        assert isinstance(data, list)
        data = [base64.decodebytes(bytes(x, encoding='utf8')) for x in data]
        data = ocr.extract_multiple_images(data)
        logger.info('ocr done')
        # data = [''.join(x) for x in data]
        filtered_data = []
        output = []
        conf = []
        model = src_predict.get_model(args)
        for single_image_data in data:
            logger.info('single_image_data:{}'.format(single_image_data))
            single_image_data = [x for x in single_image_data if len(x) > min_ocr_len]
            if len(single_image_data) == 0:
                # filtered_data.append(None)
                # output.append(None)
                # conf.append(None)
                return gen_response(
                    200, []
                )
            single_conf = src_predict.predict(args, model, tokenizer, single_image_data)
            single_conf = torch.Tensor([x['prob'] if int(x['label']) else 1 - x['prob'] for x in single_conf])
            print('single_conf:', single_conf)
            single_output = src_confidence.predict(user_conf_model, single_conf)
            print('single_output:', single_output)
            argmax_index = torch.argmax(single_conf, dim=-1)
            print('argmax_index:', argmax_index)
            filtered_data.append(single_image_data[argmax_index])
            output.append(int(single_output[argmax_index]))
            conf.append(float(single_conf[argmax_index]))

        print('fileted_data:', filtered_data)
        logger.info('filtered_data:{}'.format(filtered_data))
        print('conf:', conf)
        logger.info('conf:{}'.format(conf))
        print('output:', output)
        logger.info('output:{}'.format(output))
        ner_output = ner.parse(filtered_data)
        title = titler.summarize(filtered_data)

        response = [None] * len(filtered_data)
        for i, text in enumerate(filtered_data):
            new_agenda_data = AgendaData.objects.create(
                raw_text=text,
                output = conf[i] > 0.5
            )
            new_conf_data = ConfData.objects.create(
                conf=conf[i],
                output=int(output[i]),
                user=user,
            )
            is_agenda = bool(conf[i] > 0.5)
            confidence = conf[i]
            dates = ner_output[i]['date']
            dates = [x for x in dates if x[1] is not None]
            times = ner_output[i]['time']
            times = [x for x in times if x[1] is not None]
            locations = ner_output[i]['location']
            response[i] = {
                'id': new_agenda_data.id,
                'conf_id': new_conf_data.id,
                'dates': dates,
                'times': times,
                'locations': locations,
                'title': title[i],
                'raw_text': text,
                'is_agenda': is_agenda,
                'confidence': confidence,
                'confidence_high': bool(output[i])
            }
        [print('response[{}]'.format(i), response[i]) for i in range(len(response))]
        # [logger.info('response[{}]:{}'.format(i, response[i])) for i in range(len(response))]
        res = gen_response(
            200,
            response
        )
        return res

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
    # print('feedback:', body)
    try:
        user: User = request.user.user
    except:
        return gen_response(
            200, {
                'status': 'request_error',
                'detail': 'user not login'
            }
        )    
    try:
        data = body['data']
        assert isinstance(data, list)
        logger.info('data:{}'.format(data))
        for item in data:
            id = item['id']
            conf_id = item['conf_id']
            # text = item['text']
            is_agenda = item['is_agenda']
            is_agenda = bool(is_agenda)
            confidence_high = item['confidence_high']
            confidence_high = bool(confidence_high)
            agenda_data = AgendaData.objects.get(id=id)
            conf_data = ConfData.objects.get(id=conf_id)
            agenda_data.output = is_agenda
            conf_data.output = confidence_high
            agenda_data.save()
            conf_data.save()
            

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
    
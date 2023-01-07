from django.db import models
from django.contrib.auth.models import User as User_django

# Create your models here.

class AgendaData(models.Model):
    raw_text = models.CharField(max_length=2000)
    output = models.IntegerField()

class ConfParam(models.Model):
    model_dict = models.CharField(max_length=200)

class ConfData(models.Model):
    conf = models.FloatField()
    output = models.IntegerField()
    user = models.ForeignKey('User', related_name='conf_data_set', on_delete=models.CASCADE)

class User(models.Model):
    name = models.CharField(unique=True, max_length=50)
    django_user = models.OneToOneField(User_django, on_delete=models.CASCADE)
    conf_param = models.OneToOneField(ConfParam, default=None, on_delete=models.CASCADE)
    conf_path = models.CharField(max_length=200)
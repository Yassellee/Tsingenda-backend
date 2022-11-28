from django.db import models
from django.contrib.auth.models import User as User_django

# Create your models here.

class User(models.Model):
    name = models.CharField(unique=True, max_length=50)
    django_user = models.OneToOneField(User_django, on_delete=models.CASCADE)
from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login),
    path('raw_text/', views.raw_text),
    path('image/', views.image),
    path('feedback/', views.feedback),
]

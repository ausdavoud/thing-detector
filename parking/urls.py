# chat/urls.py
from django.urls import path

from parking import views

urlpatterns = [
    path("", views.index, name="index"),
    path("webcam/", views.webcam, name="room"),
]

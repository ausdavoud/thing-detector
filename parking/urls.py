# chat/urls.py
from django.urls import path

from parking import views

urlpatterns = [
    path("webcam/", views.webcam, name="room"),
]

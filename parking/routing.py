from django.urls import re_path


from parking import consumers

websocket_urlpatterns = [   
    re_path(r'ws/yolo/$', consumers.YOLOConsumer.as_asgi()),
]
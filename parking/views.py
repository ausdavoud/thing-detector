from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, "parking/index.html")

def webcam(request):
    return render(request, "parking/webcam.html")
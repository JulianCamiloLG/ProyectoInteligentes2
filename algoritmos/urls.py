from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static

app_name = 'algoritmos'

urlpatterns = [
    path('', home, name="index"),
    path('elecciones/', elecciones, name="elecciones"),
    path('resultados/', resultados, name="resultados"),
    path('neuronal/', neuronal, name="neuronal"),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
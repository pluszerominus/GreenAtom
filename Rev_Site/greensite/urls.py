from django.urls import path

from . import views

urlpatterns = [
    #path('', views.index, name='index'),
    path('', views.get_reviews, name='get_reviews'),
]
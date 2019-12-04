from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [
    path('', views.record1, name='record'),
    path('detail/', views.record, name='detail'),
    path('recorddb/', views.record2, name='recorddb'),
    path('detaildb/', views.record3, name='detaildb'),    

]
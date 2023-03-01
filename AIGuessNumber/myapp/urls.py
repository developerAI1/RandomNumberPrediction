from django.urls import path,include
from . import views

urlpatterns = [
      path('random/',views.prepare_data,name='save_random_arrays'),
      path('add/',views.add_new_sequence,name='add_new_sequence')
]
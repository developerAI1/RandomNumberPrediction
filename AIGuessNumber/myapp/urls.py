from django.urls import path,include
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns = [
      # path("" , views.home),
      path('save/',views.SaveDAtabase,name='save_random_arrays'),
      # path('model/',views.prepare_data),
      # path('generate/',views.AIGuess),
      # path('history/' ,views.array_history),
      # path('reset/' ,views.Reset_history),
      # path('input/' ,views.add_new_sequence),
      ]
urlpatterns += staticfiles_urlpatterns()
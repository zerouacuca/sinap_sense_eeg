# analysis/urls.py
from django.urls import path
from . import views

# Adicione esta linha
app_name = 'analysis'

urlpatterns = [
    # As URLs aqui já estão sob o prefixo /analysis/
    path('upload/', views.upload_eeg, name='upload_eeg'),
    path('dashboard/<int:eeg_id>/', views.dashboard, name='dashboard'),
    path('channel/<int:channel_id>/', views.channel_detail, name='channel_detail'),
    path('update_topomap/', views.update_topomap, name='update_topomap'),
    path('eeg_list/', views.EEGList.as_view(), name='eeg_list'),
    path('get_events/<int:eeg_id>/', views.get_events, name='get_events'),
]
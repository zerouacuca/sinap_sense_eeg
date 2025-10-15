# eeg_dashboard/urls.py
from django.contrib import admin
from django.urls import path, include
from analysis.views import home, CustomLoginView
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('analysis/', include('analysis.urls', namespace='analysis')),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(next_page='login'), name='logout'),
    path('', home, name='home'),

    # Adicione esta linha para incluir as URLs de autenticação do Django
    path('', include('django.contrib.auth.urls')),
]
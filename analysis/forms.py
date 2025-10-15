# analysis/forms.py
from django import forms
from .models import EEGData

class EEGUploadForm(forms.ModelForm):
    class Meta:
        model = EEGData
        fields = ['original_file', 'sampling_rate', 'age', 'sex']  # Campos atualizados
        labels = {
            'original_file': 'Arquivo EEG (CSV)',
            'sampling_rate': 'Taxa de Amostragem (Hz)',
            'age': 'Idade do Paciente',
            'sex': 'Sexo do Paciente'
        }
        widgets = {
            'sex': forms.Select(choices=EEGData.SEX_CHOICES)
        }

class EEGFilterForm(forms.Form):
    event = forms.CharField(label='Evento', required=False)
    start_time = forms.CharField(label='Início (HH:MM:SS)', required=False)
    end_time = forms.CharField(label='Fim (HH:MM:SS)', required=False)
# analysis/models.py
from django.db import models
import json

class EEGData(models.Model):
    SEX_CHOICES = [
        ('M', 'Masculino'),
        ('F', 'Feminino'),
        ('O', 'Outro'),
    ]
    
    uploaded_at = models.DateTimeField(auto_now_add=True)
    original_file = models.FileField(upload_to='eeg_data/')
    sampling_rate = models.FloatField(default=250.0)
    processed = models.BooleanField(default=False)
    age = models.IntegerField(verbose_name='Idade', null=True, blank=True)  # Novo campo
    sex = models.CharField(  # Novo campo
        max_length=1,
        choices=SEX_CHOICES,
        verbose_name='Sexo',
        null=True,
        blank=True
    )

class EEGChannelAnalysis(models.Model):
    eeg_data = models.ForeignKey(EEGData, on_delete=models.CASCADE)
    channel_name = models.CharField(max_length=50)
    raw_signal = models.TextField()
    highpass = models.TextField()
    lowpass = models.TextField()
    bandpass = models.TextField()
    notch = models.TextField()
    delta_power = models.FloatField()
    theta_power = models.FloatField()
    alpha_power = models.FloatField()
    beta_power = models.FloatField()
    gamma_power = models.FloatField()

    class Meta:
        ordering = ['channel_name']
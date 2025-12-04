"""
    Processa e analisa dados de EEG de um arquivo associado ao objeto eeg_data.
    Este método realiza as seguintes etapas para cada canal de EEG:
    1. Lê os dados brutos do arquivo CSV associado.
    2. Ajusta o timestamp para o formato datetime apropriado.
    3. (Removido) Aplicação de filtros digitais ao sinal.
    4. Calcula a potência média em diferentes bandas de frequência (delta, theta, alpha, beta, gamma) usando PSD (Welch).
    5. Salva os resultados processados e métricas em registros do modelo EEGChannelAnalysis.
    6. Atualiza o status do objeto eeg_data para indicar que o processamento foi concluído.
    Parâmetros:
        eeg_data (EEGData): Instância contendo informações do arquivo de EEG e metadados necessários para o processamento.
    Observações:
        - Não aplica mais filtros ao sinal antes de salvar.
        - Utiliza pandas, numpy, scipy.signal.welch e json para manipulação e análise dos dados.
        - Os resultados são salvos no banco de dados via o modelo EEGChannelAnalysis.
    """
import pandas as pd
import numpy as np
import json
from scipy.signal import welch
from .models import EEGChannelAnalysis



def process_eeg_data(eeg_data, event=None, start_time=None, end_time=None):
    df = pd.read_csv(eeg_data.original_file.path)
    fs = eeg_data.sampling_rate
    
    # Ajustar o timestamp para o formato correto
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

    if event:
        event_start_time = df[df['Marker value'] == event]['Timestamp'].iloc[0]
        event_end_time = event_start_time + pd.to_timedelta(df[df['Marker value'] == event]['Marker timestamp'].iloc[0], unit='s')
        
        if start_time and end_time:
            start_time_dt = pd.to_datetime(start_time, format='%H:%M:%S').time()
            end_time_dt = pd.to_datetime(end_time, format='%H:%M:%S').time()

            start_datetime = event_start_time.replace(hour=start_time_dt.hour, minute=start_time_dt.minute, second=start_time_dt.second)
            end_datetime = event_start_time.replace(hour=end_time_dt.hour, minute=end_time_dt.minute, second=end_time_dt.second)
            
            df = df[(df['Timestamp'] >= start_datetime) & (df['Timestamp'] <= end_datetime)]
        else:
            df = df[(df['Timestamp'] >= event_start_time) & (df['Timestamp'] <= event_end_time)]
    
    df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    # Limpar análises antigas
    EEGChannelAnalysis.objects.filter(eeg_data=eeg_data).delete()
    
    for channel in df.columns[1:9]:
        data = df[channel].values
        timestamp = df['Timestamp'].values
        
        # Não aplicar filtros — manter sinal bruto
        raw_json = json.dumps({'x': timestamp.tolist(), 'y': data.tolist()})

        # Calcular potências via PSD (Welch) sem filtrar o sinal
        bandas = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        # Estimar PSD
        f, Pxx = welch(data, fs=fs, nperseg=min(1024, len(data)))
        
        power_metrics = {}
        for banda, (low, high) in bandas.items():
            idx = np.logical_and(f >= low, f <= high)
            band_power = float(np.trapz(Pxx[idx], f[idx])) if np.any(idx) else 0.0
            power_metrics[f'{banda}_power'] = band_power
        
        # Criar registro — campos de filtros guardam o sinal bruto para compatibilidade
        EEGChannelAnalysis.objects.create(
            eeg_data=eeg_data,
            channel_name=channel,
            raw_signal=raw_json,
            highpass=raw_json,
            lowpass=raw_json,
            bandpass=raw_json,
            notch=raw_json,
            **power_metrics
        )
    
    eeg_data.processed = True
    eeg_data.save()
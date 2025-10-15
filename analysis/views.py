# analysis/views.py
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
import pandas as pd
from .eeg_processor import process_eeg_data
from .forms import EEGUploadForm, EEGFilterForm
from .models import EEGData, EEGChannelAnalysis
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from scipy.interpolate import griddata
import numpy as np
from scipy.signal import butter, lfilter
from django.views.generic import ListView
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
from django.db.models import Q , Sum
import io

def home(request):
    """
    View principal da aplicação. Exibe estatísticas públicas e informações institucionais do laboratório.
    
    Contexto Retornado:
        - stats: Dicionário com dados agregados (total de uploads, canais, processamentos e potência total)
        - laboratory_info: Informações institucionais (missão, equipe e parceiros)
    """
    # Coleta estatísticas públicas do banco de dados
    stats = {
        'total_uploads': EEGData.objects.count(),
        'total_channels': EEGChannelAnalysis.objects.count(),
        'total_processed': EEGData.objects.filter(processed=True).count(),
        'total_power': EEGChannelAnalysis.objects.aggregate(
            total=Sum('alpha_power') + Sum('beta_power') + Sum('gamma_power')
        )['total'] or 0
    }
    
    return render(request, 'home.html', {
        'stats': stats,
        'laboratory_info': {
            'mission': """O SinapSense é um laboratório multidisciplinar dedicado à pesquisa em Neurociência do Consumo...""",
            'team': [
                {'name': 'Dr. João Silva', 'area': 'Neurociência'},
                {'name': 'Dra. Maria Santos', 'area': 'Análise de Dados'},
                # Adicione mais membros
            ],
            'partners': ['UFPR', 'SEPT', 'Empresa X']
        }
    })



class CustomLoginView(LoginView):
    """
    View personalizada para login de usuários.
    Herda da view padrão do Django e altera o template utilizado.
    """
    template_name = 'registration/login.html'
    redirect_authenticated_user = True
    

@login_required
def register(request):
    """
    View para registro de novos usuários.
    Acesso restrito a usuários autenticados (@login_required).
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def upload_eeg(request):
    """
    View para upload de arquivos EEG.
    Processa o formulário e inicia o processamento dos dados após upload.
    """
    if request.method == 'POST':
        form = EEGUploadForm(request.POST, request.FILES)
        if form.is_valid():
            eeg_data = form.save()
            process_eeg_data(eeg_data) # Processamento assíncrono recomendado para produção
            return redirect('analysis:dashboard', eeg_id=eeg_data.id)
    else:
        form = EEGUploadForm()
    return render(request, 'upload.html', {'form': form})

def get_events(request, eeg_id):
    eeg_data = EEGData.objects.get(id=eeg_id)
    df = pd.read_csv(eeg_data.original_file.path)
    events = df['Marker value'].dropna().unique().tolist()
    return JsonResponse({'events': events})


def analyze_sentiment(analyses, age=None, sex=None):
    """
    Analisa o estado emocional com base nas potências das bandas cerebrais.
    
    Parâmetros:
        analyses (QuerySet): Conjunto de análises de canal EEG
        age (int, opcional): Idade do participante para contextualização
        sex (str, opcional): Sexo biológico ('M' ou 'F') para contextualização
    
    Retorna:
        dict: Dicionário com:
            - 'sentiment': Classificação textual do estado
            - 'avg_values': Médias de potência por banda
    """
    # Cálculo das médias das potências por banda
    avg = {
        'delta': np.mean([a.delta_power for a in analyses]),
        'theta': np.mean([a.theta_power for a in analyses]),
        'alpha': np.mean([a.alpha_power for a in analyses]),
        'beta': np.mean([a.beta_power for a in analyses]),
        'gamma': np.mean([a.gamma_power for a in analyses])
    }

    # Lógica de determinação de sentimentos
    sentiment = "Neutro"

    # Relações entre bandas para classificação
    if avg['beta'] > avg['alpha'] * 1.5:
        sentiment = "Agitação/Estresse"
        if avg['gamma'] > 0.08: sentiment += " Intensa"
    elif avg['alpha'] > avg['beta'] * 1.2:
        sentiment = "Relaxamento"
        if avg['theta'] > 0.1: sentiment += " Profundo"
    elif avg['theta'] > avg['alpha'] and avg['theta'] > avg['beta']:
        sentiment = "Sonolência/Criatividade"
    
    # Adiciona contexto demográfico
    if age:
        if age < 18:
            sentiment += " (Jovem)"
        elif 18 <= age <= 65:
            sentiment += " (Adulto)"
        else:
            sentiment += " (Idoso)"
    
    if sex == 'F':
        sentiment += " ♀"
    elif sex == 'M':
        sentiment += " ♂"
    
    # Adiciona indicador de sexo
    return {
        'sentiment': sentiment,
        'avg_values': avg
    }

def create_brain_waves_plot(analyses):
    """
    Gera gráfico interativo das ondas cerebrais médias.
    
    Passos:
        1. Extrai dados temporais do primeiro canal como referência
        2. Aplica filtros de banda em todos os canais
        3. Calcula média dos sinais filtrados
        4. Normaliza amplitudes para visualização combinada
    
    Retorna:
        str: HTML do gráfico Plotly para incorporação em templates
    """
    # Configuração inicial usando o primeiro canal como referência
    first_channel = analyses[0]
    
    # Extrair os dados de tempo do sinal raw
    raw_data = json.loads(first_channel.raw_signal)
    timestamps = raw_data['x']
    
    # Criar gráfico de linhas para as bandas cerebrais
    fig = go.Figure()
    
    # Paleta de cores para as diferentes bandas
    colors = {
        'delta': 'rgba(220, 53, 69, 0.8)',  # Vermelho (Bootstrap danger)
        'theta': 'rgba(255, 193, 7, 0.8)',  # Amarelo (Bootstrap warning)
        'alpha': 'rgba(13, 202, 240, 0.8)',  # Azul claro (Bootstrap info)
        'beta': 'rgba(13, 110, 253, 0.8)',   # Azul (Bootstrap primary)
        'gamma': 'rgba(25, 135, 84, 0.8)'    # Verde (Bootstrap success)
    }
    
    # Definição das faixas de frequência por banda
    bandas = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    # Obter os dados brutos do sinal
    raw_signal = np.array(raw_data['y'])
    
    # Calcular a média dos sinais de todos os canais para cada banda
    avg_signals = {}
    
    # Processamento multicanal
    for banda, (low, high) in bandas.items():
        # Inicializar array para acumular sinais de todos os canais
        band_signals = np.zeros_like(raw_signal)
        
        # Para cada canal, extrair o sinal da banda e acumular
        for analysis in analyses:
            # Obter sinal bruto do canal
            channel_data = json.loads(analysis.raw_signal)
            channel_signal = np.array(channel_data['y'])
            
            # Aplicar filtro de banda
            fs = analysis.eeg_data.sampling_rate
            b, a = butter_bandpass(low, high, fs)
            filtered = lfilter(b, a, channel_signal)
            
            # Acumular o sinal filtrado
            band_signals += filtered
        
        # Calcular a média dos sinais de todos os canais
        avg_signals[banda] = band_signals / len(analyses)
    
    # Adicionar linhas para cada banda
    for banda, signal in avg_signals.items():
        # Normalizar o sinal para melhor visualização
        normalized_signal = signal / np.max(np.abs(signal)) * 0.5
        
        fig.add_trace(go.Scatter(
            x=timestamps,  # Limitar para os primeiros 500 pontos para melhor visualização
            y=normalized_signal,
            name=banda.capitalize(),
            line=dict(color=colors[banda], width=2)
        ))
    
    # Configurar layout
    fig.update_layout(
        title='Sinais das Ondas Cerebrais',
        xaxis_title='Tempo',
        yaxis_title='Amplitude (Normalizada)',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig.to_html(full_html=False)

@login_required
def dashboard(request, eeg_id):
    eeg_data = EEGData.objects.get(id=eeg_id)
    filter_form = EEGFilterForm(request.POST or None)

    event = request.POST.get('event')
    start_time = request.POST.get('start_time')
    end_time = request.POST.get('end_time')

    if request.method == 'POST' and filter_form.is_valid():
        process_eeg_data(eeg_data, event, start_time, end_time)

    analyses = EEGChannelAnalysis.objects.filter(eeg_data=eeg_data)
    bandas = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    # Gráfico de potências
    fig_power = px.line(title='Distribuição de Potência por Canal')
    for banda in bandas:
        values = [getattr(a, f'{banda}_power') for a in analyses]
        fig_power.add_scatter(
            x=[a.channel_name for a in analyses],
            y=values,
            name=banda.capitalize()
        )
    
    # Análise de sentimentos
    sentiment_analysis = analyze_sentiment(analyses, eeg_data.age, eeg_data.sex)
    # Criar gráfico de ondas cerebrais
    brain_waves_plot = create_brain_waves_plot(analyses)
    
    # Componentes do dashboard
    return render(request, 'dashboard.html', {
        'eeg_data': eeg_data,
        'analyses': analyses,
        'bandas': [b.capitalize() for b in bandas],
        'power_plot': fig_power.to_html(full_html=False),
        'topomap_plot': get_topomap(analyses, 'Alpha'),
        'sentiment_analysis': sentiment_analysis,
        'brain_waves_plot': brain_waves_plot,  # Novo gráfico de ondas cerebrais
        'filter_form': filter_form
    })

def get_topomap(analyses, banda):
    """
    Gera mapa topográfico 2D da atividade cerebral para uma banda específica.
    
    Parâmetros:
        analyses (QuerySet): Análises de canal para extrair dados
        banda (str): Banda cerebral a ser visualizada (ex: 'Alpha')
    
    Retorna:
        str: HTML do gráfico Plotly pronto para incorporação
    """
    # Mapeamento de posições dos eletrodos (coordenadas normalizadas)
    posicoes = {
        'EEG Channel 1': (0.5, 0.9),
        'EEG Channel 2': (0.3, 0.7),
        'EEG Channel 3': (0.7, 0.7),
        'EEG Channel 4': (0.2, 0.5),
        'EEG Channel 5': (0.8, 0.5),
        'EEG Channel 6': (0.5, 0.3),
        'EEG Channel 7': (0.3, 0.2),
        'EEG Channel 8': (0.7, 0.2)
    }
    # Extração e processamento dos dados
    valores = [getattr(a, f'{banda.lower()}_power') for a in analyses]
    x = np.array([posicoes[ch.channel_name][0] for ch in analyses])
    y = np.array([posicoes[ch.channel_name][1] for ch in analyses])
    z = np.array(valores)
    
    # Interpolação para superfície suave
    xi, yi = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    
    # Criação do heatmap
    fig = px.imshow(
            zi,
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            color_continuous_scale='Viridis',
            title=f'Mapa de Atividade {banda.capitalize()}'
        )
        
    return fig.to_html(
            full_html=False,
            include_plotlyjs=True,
            div_id='dynamic-topomap',  # ID único
            config={'responsive': True})

@login_required        
def update_topomap(request):
    eeg_id = request.GET.get('eeg_id')
    banda = request.GET.get('banda')
    analyses = EEGChannelAnalysis.objects.filter(eeg_data__id=eeg_id)
    
    # Retorne APENAS o HTML interno do gráfico (sem wrappers)
    return HttpResponse(
        get_topomap(analyses, banda)
    )


def channel_detail(request, channel_id):
    analysis = EEGChannelAnalysis.objects.get(id=channel_id)
    
    # Criar gráfico de filtros
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Original', 'Notch']
    )
    
    signals = {
        'Original': json.loads(analysis.raw_signal),
        'Notch': json.loads(analysis.notch)
    }
    
    for i, (name, data) in enumerate(signals.items(), 1):
        fig.add_trace(
            go.Scatter(
                x=data['x'],
                y=data['y'],
                name=name,
                line=dict(width=1)
            ),
            row=i, col=1
        )
    
    fig.update_layout(height=1200, showlegend=False)
    fig.update_xaxes(title_text='Tempo Decorrido (s)')


    return render(request, 'channel_detail.html', {
        'analysis': analysis,
        'plot_div': fig.to_html(full_html=False),
    })


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

class EEGList(ListView):
    model = EEGData
    template_name = 'eeg_list.html'
    context_object_name = 'eeg_records'
    paginate_by = 10

    def get_queryset(self):
        queryset = super().get_queryset()
        start_date = self.request.GET.get('start_date')
        end_date = self.request.GET.get('end_date')
        
        if start_date and end_date:
            queryset = queryset.filter(
                Q(uploaded_at__date__gte=start_date) &
                Q(uploaded_at__date__lte=end_date)
            )
        return queryset.order_by('-uploaded_at')
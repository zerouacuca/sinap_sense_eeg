# Use uma versão estável do Python
FROM python:3.12

# Cria o diretório da aplicação
RUN mkdir /app

# Define o diretório de trabalho
WORKDIR /app

# Variáveis de ambiente (removido ARG e ENV de senha para simplificar)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Atualiza o pip
RUN pip install --upgrade pip

# Copia e instala dependências
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos do projeto
COPY . /app/

# Executa migrações (o superuser será criado manualmente depois)
RUN python manage.py migrate

# Expõe a porta do Django
EXPOSE 8000

# Inicia o servidor
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
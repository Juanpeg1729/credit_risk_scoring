# Usamos una imagen ligera de Python 3.11 (Bookworm es Debian 12)
FROM python:3.11-slim-bookworm

# Instalamos uv directamente desde su imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Evita archivos .pyc y habilita logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero los archivos de dependencias 
COPY pyproject.toml uv.lock ./

# Instalamos las dependencias del sistema
RUN uv sync

# Copiamos el resto del código
COPY . .

# Añade el entorno virtual al PATH para ejecutar comandos sin "uv run"
ENV PATH="/app/.venv/bin:$PATH"

# Documenta que el contenedor escucha en el puerto 8000
EXPOSE 8000

# Comando que se ejecuta al iniciar el contenedor
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
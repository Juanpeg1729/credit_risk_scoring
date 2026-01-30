# Usamos una imagen ligera de Python 3.11 (Bookworm es Debian 12)
FROM python:3.11-slim-bookworm

# Instalamos uv directamente desde su imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero los archivos de dependencias 
COPY pyproject.toml uv.lock ./

# Instalamos las dependencias del sistema
RUN uv sync --frozen

# Copiamos el resto del código
COPY . .

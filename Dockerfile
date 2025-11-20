# 1. Usamos una imagen base de Python ligera y moderna
FROM python:3.12-slim

# 2. Evitamos que Python genere archivos .pyc y activamos logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Instalamos las dependencias del sistema necesarias para compilar
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Copiamos los archivos de gestión de dependencias primero (para aprovechar caché)
COPY pyproject.toml uv.lock ./

# 6. Instalamos 'uv' y las dependencias del proyecto
RUN pip install uv && uv sync --frozen

# 7. Copiamos el resto del código del proyecto
COPY . .

# 8. Entrenamos el modelo DENTRO del contenedor al construir la imagen
# (Esto asegura que el modelo .pkl esté listo antes de arrancar la API)
RUN uv run python -m src.train

# 9. Exponemos el puerto donde correrá la API
EXPOSE 8000

# 10. Comando para arrancar la API cuando se inicie el contenedor
CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
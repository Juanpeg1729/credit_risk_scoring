# Makefile para Credit Risk Scoring

# Variables
IMAGE_NAME = credit-risk-app
PORT = 8000

# --- Instalación y Entorno ---
install: ## Instalar dependencias con uv
	uv sync

# --- Ejecución Local ---
train: ## Entrenar el modelo (Pipeline completo)
	uv run python -m src.train

api: ## Levantar la API en modo desarrollo
	uv run uvicorn src.api:app --reload

dashboard: ## Abrir el Dashboard de Streamlit
	uv run streamlit run src/dashboard.py

# --- Docker ---
docker-build: ## Construir la imagen de Docker
	docker build -t $(IMAGE_NAME) .

docker-up: ## Levantar TODO (API + Dashboard)
	docker-compose up

docker-down: ## Apagar todo
	docker-compose down

# --- Calidad y Limpieza ---
test: ## Ejecutar tests 
	uv run pytest

clean: ## Limpiar archivos temporales
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/outputs
	rm -rf outputs/
	rm -rf .pytest_cache
	@echo "🧹 Proyecto limpio"

# --- Ayuda ---
help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
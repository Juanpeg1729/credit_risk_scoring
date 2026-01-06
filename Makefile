.PHONY: help install api dashboard docker-build docker-up docker-down clean

help:
	@echo "Comandos disponibles:"
	@echo "  make install       - Instalar dependencias"
	@echo "  make api           - Levantar API"
	@echo "  make dashboard     - Abrir dashboard"
	@echo "  make docker-build  - Construir imagen Docker"
	@echo "  make docker-up     - Levantar servicios"
	@echo "  make docker-down   - Detener servicios"
	@echo "  make clean         - Limpiar archivos temporales"

install:
	uv sync

api:
	uv run python -m uvicorn src.api:app --reload

dashboard:
	uv run python -m streamlit run src/dashboard.py

docker-build:
	docker build -t credit-risk-app .

docker-up:
	docker-compose up

docker-down:
	docker-compose down

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache outputs/
	@echo "Limpieza completada"
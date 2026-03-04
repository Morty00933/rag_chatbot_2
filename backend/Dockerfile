# Одностадийный, максимально быстрый для разработки
# Рекомендуется включить BuildKit перед сборкой:
#   PowerShell: $env:DOCKER_BUILDKIT="1"
#   Bash: export DOCKER_BUILDKIT=1

FROM ghcr.io/astral-sh/uv:0.6-python3.11-bookworm-slim

# Базовые переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Рантайм-библиотеки (нужны для sklearn и др. бинарных колёс)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 1) Копируем только requirements для стабильного кэша
COPY requirements.txt /app/requirements.txt

# 2) Ставим CPU-версию PyTorch до остальных пакетов (важно!)
#    Используем официальный CPU index PyTorch, чтобы исключить ROCm/CUDA.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.4.1"

# 3) Остальные зависимости проекта
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /app/requirements.txt

# 4) Копируем исходники (меняются чаще — не ломают слой зависимостей)
COPY . /app

EXPOSE 8000

# Без --reload: в контейнере он маскирует падения воркера
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]

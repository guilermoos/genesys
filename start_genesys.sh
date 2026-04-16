#!/bin/bash
# Script para iniciar o Genesys com 3 abas no Konsole

cd /home/guilherme/projetos/genesys

# Serviço 1: Redis (só sobe se não estiver rodando)
if ! redis-cli ping &>/dev/null; then
    konsole -e "bash -c 'echo \"=== Genesys Redis ===\"; redis-server; echo \"Pressione Enter para fechar...\"; read'" &
else
    echo "ℹ️  Redis já está rodando em localhost:6379"
fi
sleep 1

# Serviço 2: API FastAPI
konsole -e "bash -c 'source .venv/bin/activate; echo \"=== Genesys API ===\"; python main.py; echo \"Pressione Enter para fechar...\"; read'" &
sleep 2

# Serviço 3: Celery Worker
konsole -e "bash -c 'source .venv/bin/activate; echo \"=== Genesys Celery Worker ===\"; celery -A app.workers.celery_app worker --loglevel=info; echo \"Pressione Enter para fechar...\"; read'" &

echo "✅ Todas as janelas do Konsole foram abertas!"
echo ""
echo "Serviços iniciados:"
echo "  - Redis: localhost:6379"
echo "  - API: http://localhost:8000"
echo "  - Celery: Processando jobs de treinamento"
echo ""
echo "URLs úteis:"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Health: http://localhost:8000/health"
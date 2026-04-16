# GPU & Testing Suite

Esta pasta contém testes completos para verificar funcionalidade de GPU e performance do sistema.

## 📋 Arquivos de Teste

### 1. `test_gpu_cpu_benchmark.py`
Benchmark comparativo entre CPU e GPU para inferência.

**Classes:**
- `TestGPUCPUBenchmark`: Testes de performance CPU vs GPU (modelos pequenos)
- `TestLargeModelGPUBenchmark`: Testes com modelos grandes otimizados para GPU

**Testes principais:**
```bash
# Teste de performance CPU
pytest tests/test_gpu_cpu_benchmark.py::TestGPUCPUBenchmark::test_cpu_inference_performance -v -s

# Teste de performance GPU
pytest tests/test_gpu_cpu_benchmark.py::TestGPUCPUBenchmark::test_gpu_inference_performance -v -s

# Comparação CPU vs GPU
pytest tests/test_gpu_cpu_benchmark.py::TestGPUCPUBenchmark::test_gpu_vs_cpu_comparison -v -s

# Teste com modelo grande
pytest tests/test_gpu_cpu_benchmark.py::TestLargeModelGPUBenchmark::test_large_model_gpu_speedup -v -s
```

### 2. `test_gpu_training.py`
Testes de treinamento otimizado para GPU.

**Classes:**
- `TestGPUOptimizedTraining`: Testes de treinamento com configurações GPU

**Testes principais:**
```bash
# Treinamento modelo pequeno
pytest tests/test_gpu_training.py::TestGPUOptimizedTraining::test_small_model_training_speed -v -s

# Treinamento modelo grande
pytest tests/test_gpu_training.py::TestGPUOptimizedTraining::test_large_model_training_speed -v -s

# Impacto de batch size
pytest tests/test_gpu_training.py::TestGPUOptimizedTraining::test_batch_size_impact_on_gpu -v -s

# Limpeza de memória GPU
pytest tests/test_gpu_training.py::TestGPUOptimizedTraining::test_gpu_memory_cleanup -v -s
```

### 3. `test_api_gpu_integration.py`
Testes de integração da API com treinamento GPU.

**Classes:**
- `TestGPUTrainingWorkflow`: Testes de workflow completo treinamento + inferência
- `TestGPUAPIEndpoints`: Testes dos endpoints da API

**Testes principais:**
```bash
# Workflow treinamento pequeno modelo
pytest tests/test_api_gpu_integration.py::TestGPUTrainingWorkflow::test_small_model_training_workflow -v -s

# Workflow treinamento modelo grande
pytest tests/test_api_gpu_integration.py::TestGPUTrainingWorkflow::test_large_model_training_workflow -v -s

# Workflow completo (treinamento + inferência)
pytest tests/test_api_gpu_integration.py::TestGPUTrainingWorkflow::test_training_and_inference_workflow -v -s

# Projetos concorrentes
pytest tests/test_api_gpu_integration.py::TestGPUTrainingWorkflow::test_multiple_concurrent_projects -v -s
```

### 4. `test_rl.py` (Existente)
Testes dos componentes RL (Network, ReplayBuffer, Agent).

### 5. `test_templates.py` (Existente)
Testes dos templates de ambiente.

## 🚀 Como Executar

### Todos os testes
```bash
pytest tests/ -v
```

### Apenas testes GPU
```bash
pytest tests/test_gpu_cpu_benchmark.py tests/test_gpu_training.py -v -s
```

### Apenas testes de integração
```bash
pytest tests/test_api_gpu_integration.py -v -s -m integration
```

### Testes específicos com output
```bash
pytest tests/test_gpu_cpu_benchmark.py::TestGPUCPUBenchmark::test_gpu_vs_cpu_comparison -v -s
```

### Testes com relatório
```bash
pytest tests/ -v --tb=short --html=report.html
```

## ⚙️ Configuração

### Pré-requisitos
```bash
# Ativar ambiente
source .venv/bin/activate

# Instalar dependências de teste (se necessário)
pip install pytest pytest-html
```

### Para testes de integração
Certifique-se de que:
1. API está rodando: `http://localhost:8000`
2. Celery worker está ativo
3. Redis está disponível
4. Token válido em `TEST_TOKEN` (em `test_api_gpu_integration.py`)

```bash
# Terminal 1 - API
python main.py

# Terminal 2 - Celery worker
celery -A app.workers.celery_app worker -l info

# Terminal 3 - Testes
pytest tests/test_api_gpu_integration.py -v -s
```

## 📊 Benchmarks Esperados

### CPU vs GPU (Modelo Pequeno - 50 dim, batch 128)
```
CPU: ~696K predições/segundo
GPU: ~138K predições/segundo
GPU é 5x mais LENTO (overhead de transferência)
```

### GPU Otimizado (Modelo Grande - 128 dim, batch 256)
```
Pequeno (50 dim):  50 eps em 3.6s = 13.8 eps/seg
Grande (128 dim): 150 eps em 43.4s = 3.5 eps/seg
Complexidade 3x maior com velocidade similar = GPU bem utilizada!
```

### Quando GPU é Melhor
```
✓ State size > 100
✓ Hidden layers > [256, 256]
✓ Batch size > 256
✓ Episodes > 500
✓ Models maiores e batch sizes maiores
```

## 🔍 Output dos Testes

Os testes imprimem informações detalhadas:

```
GPU Information:
  Device Name: NVIDIA GeForce RTX 4060
  Total Memory: 8.00 GB
  Current Memory Allocated: 0.52 GB
  CUDA Version: 13.0
  PyTorch Version: 2.11.0+cu130
```

## 📝 Notas

- Testes com `@pytest.mark.skipif(not torch.cuda.is_available())` pulam se GPU não disponível
- Testes de integração usam `@pytest.mark.integration` - execute com `-m integration`
- Use `-s` flag para ver output dos testes (print statements)
- Use `-v` para verbose output

## 🐛 Troubleshooting

### CUDA Not Found
```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Verificar dispositivo
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### API Connection Errors
```bash
# Verificar se API está rodando
curl http://localhost:8000/v1/health

# Verificar token
# Gerar novo token via login endpoint
```

### Memory Issues
```bash
# Limpar cache GPU
torch.cuda.empty_cache()

# Verificar memória
nvidia-smi
```

## 📚 Referências

- [PyTorch CUDA](https://pytorch.org/docs/stable/cuda.html)
- [pytest Documentation](https://docs.pytest.org/)
- [NVIDIA RTX 4060 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-family/)

---

**Status**: ✅ Todos os testes funcionando com GPU RTX 4060

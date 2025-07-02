```python
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

    Looking in indexes: https://download.pytorch.org/whl/cu118
    Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (2.6.0+cu124)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.11/dist-packages (0.21.0+cu124)
    Requirement already satisfied: torchaudio in /usr/local/lib/python3.11/dist-packages (2.6.0+cu124)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch) (3.18.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.11/dist-packages (from torch) (4.14.0)
    Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch) (3.5)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.6)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch) (2025.3.2)
    INFO: pip is looking at multiple versions of torch to determine which version is compatible with other requirements. This could take a while.
    Collecting torch
      Downloading https://download.pytorch.org/whl/cu118/torch-2.7.1%2Bcu118-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (28 kB)
    Collecting sympy>=1.13.3 (from torch)
      Downloading https://download.pytorch.org/whl/sympy-1.13.3-py3-none-any.whl.metadata (12 kB)
    Collecting nvidia-cuda-nvrtc-cu11==11.8.89 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_cuda_nvrtc_cu11-11.8.89-py3-none-manylinux1_x86_64.whl (23.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.2/23.2 MB[0m [31m53.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-runtime-cu11==11.8.89 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_cuda_runtime_cu11-11.8.89-py3-none-manylinux1_x86_64.whl (875 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m875.6/875.6 kB[0m [31m35.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-cupti-cu11==11.8.87 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_cuda_cupti_cu11-11.8.87-py3-none-manylinux1_x86_64.whl (13.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.1/13.1 MB[0m [31m47.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cudnn-cu11==9.1.0.70 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_cudnn_cu11-9.1.0.70-py3-none-manylinux2014_x86_64.whl (663.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m663.9/663.9 MB[0m [31m2.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cublas-cu11==11.11.3.6 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_cublas_cu11-11.11.3.6-py3-none-manylinux1_x86_64.whl (417.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m417.9/417.9 MB[0m [31m3.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cufft-cu11==10.9.0.58 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_cufft_cu11-10.9.0.58-py3-none-manylinux1_x86_64.whl (168.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m168.4/168.4 MB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-curand-cu11==10.3.0.86 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_curand_cu11-10.3.0.86-py3-none-manylinux1_x86_64.whl (58.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.1/58.1 MB[0m [31m12.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cusolver-cu11==11.4.1.48 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_cusolver_cu11-11.4.1.48-py3-none-manylinux1_x86_64.whl (128.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m128.2/128.2 MB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cusparse-cu11==11.7.5.86 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_cusparse_cu11-11.7.5.86-py3-none-manylinux1_x86_64.whl (204.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m204.1/204.1 MB[0m [31m5.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nccl-cu11==2.21.5 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_nccl_cu11-2.21.5-py3-none-manylinux2014_x86_64.whl (147.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m147.8/147.8 MB[0m [31m7.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nvtx-cu11==11.8.86 (from torch)
      Downloading https://download.pytorch.org/whl/cu118/nvidia_nvtx_cu11-11.8.86-py3-none-manylinux1_x86_64.whl (99 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m99.1/99.1 kB[0m [31m7.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting triton==3.3.1 (from torch)
      Downloading https://download.pytorch.org/whl/triton-3.3.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.5 kB)
    Requirement already satisfied: setuptools>=40.8.0 in /usr/local/lib/python3.11/dist-packages (from triton==3.3.1->torch) (75.2.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torchvision) (2.0.2)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.11/dist-packages (from torchvision) (11.2.1)
    Collecting torch
      Downloading https://download.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp311-cp311-linux_x86_64.whl.metadata (27 kB)
    Requirement already satisfied: triton==3.2.0 in /usr/local/lib/python3.11/dist-packages (from torch) (3.2.0)
    Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch) (3.0.2)
    Downloading https://download.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp311-cp311-linux_x86_64.whl (848.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m848.7/848.7 MB[0m [31m1.2 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: nvidia-nvtx-cu11, nvidia-nccl-cu11, nvidia-cusparse-cu11, nvidia-curand-cu11, nvidia-cufft-cu11, nvidia-cuda-runtime-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-cupti-cu11, nvidia-cublas-cu11, nvidia-cusolver-cu11, nvidia-cudnn-cu11, torch
      Attempting uninstall: torch
        Found existing installation: torch 2.6.0+cu124
        Uninstalling torch-2.6.0+cu124:
          Successfully uninstalled torch-2.6.0+cu124
    Successfully installed nvidia-cublas-cu11-11.11.3.6 nvidia-cuda-cupti-cu11-11.8.87 nvidia-cuda-nvrtc-cu11-11.8.89 nvidia-cuda-runtime-cu11-11.8.89 nvidia-cudnn-cu11-9.1.0.70 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.3.0.86 nvidia-cusolver-cu11-11.4.1.48 nvidia-cusparse-cu11-11.7.5.86 nvidia-nccl-cu11-2.21.5 nvidia-nvtx-cu11-11.8.86 torch-2.6.0+cu118
    


```python
import torch as t
import time
```

## 3.1 Подготовка данных


```python
# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
# - 128 x 512 x 512
# - 256 x 256 x 256
tensor64x1024x1024 = t.randint(11, (64, 1024, 1024), dtype=float)
tensor128x512x512 = t.randint(12, (128, 512, 512), dtype=float)
tensor256x256x256 = t.randint(13, (256, 256, 256), dtype=float)
```

## 3.2 Функция измерения времени


```python
# Используйте torch.cuda.Event() для точного измерения на GPU
# Используйте time.time() для измерения на CPU
def execute_operations(*tensors):
    """
    Выполняет операцию ('matmul', 'add', 'mul', 'transpose', 'sum') на CPU или CUDA.
    Возвращает результат и время выполнения в секундах.

    Вход:
    Тензоры tuple[tensor]

    Выход:
    Время выполнения (секунды) (float)
    """
    operations = {
        'matmul': t.matmul,
        'add': t.add,
        'mul': t.mul,
        'transpose': t.transpose,
        'sum': t.sum
    }
    operation_names = ['matmul', 'add', 'mul', 'transpose', 'sum']
    devices = ['cuda', 'cpu']

    tensors_cpu = [t.to(devices[1]) for t in tensors]
    elapsed_time_cuda = list()
    elapsed_time_cpu = list()

    if t.cuda.is_available():
      tensors_cuda = [t.to(devices[0]) for t in tensors]

      for operation_name in operation_names:
        start_event = t.cuda.Event(enable_timing=True)
        end_event = t.cuda.Event(enable_timing=True)
        t.cuda.synchronize()
        start_event.record()
        if operation_name == 'transpose':
          transp_tensors = list()
          for tensor in tensors_cuda:
            transp_tensors.append(operations[operation_name](tensor, 0, 1))
          result = transp_tensors
        elif operation_name == 'sum':
          sum_tensors = list()
          for tensor in tensors_cuda:
            sum_tensors.append(operations[operation_name](tensor))
          result = sum_tensors
        else:
         result_cudo = operations[operation_name](*tensors_cuda)
        end_event.record()
        t.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0
        elapsed_time_cuda.append(elapsed_time)
    else:
      elapsed_time_cuda = [0] * len(operation_names)

    for operation_name in operation_names:
      start_time = time.time()
      if operation_name == 'transpose':
        transp_tensors = list()
        for tensor in tensors_cpu:
          transp_tensors.append(operations[operation_name](tensor, 0, 1))
        result = transp_tensors
      elif operation_name == 'sum':
        sum_tensors = list()
        for tensor in tensors_cpu:
          sum_tensors.append(operations[operation_name](tensor))
        result = sum_tensors
      else:
        result = operations[operation_name](*tensors_cpu)
      elapsed_time = time.time() - start_time
      elapsed_time_cpu.append(elapsed_time)

    return elapsed_time_cuda, elapsed_time_cpu
```

## 3.3 Сравнение операций


```python
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
cuda, cpu = execute_operations(tensor64x1024x1024, tensor64x1024x1024)

matmul_cuda = cuda[0]
add_cuda = cuda[1]
mul_cuda = cuda[2]
transpose_cuda = cuda[3]
sum_cuda = cuda[4]

matmul_cpu = cpu[0]
add_cpu = cpu[1]
mul_cpu = cpu[2]
transpose_cpu = cpu[3]
sum_cpu = cpu[4]

# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде
print(f"Операция                        | CPU (мс) | GPU (мс) | Ускорение (раз)")
print(f"Матричное умножение             |   {matmul_cpu:.4f}   |   {matmul_cuda:.4f}   |     {1/(matmul_cuda/matmul_cpu):.4f}")
print(f"Поэлементное сложение           |   {add_cpu:.4f}   |   {add_cuda:.4f}   |     {1/(add_cuda/add_cpu):.4f}")
print(f"Поэлементное умножение          |   {mul_cpu:.4f}   |   {mul_cuda:.4f}   |     {1/(mul_cuda/mul_cpu):.4f}")
print(f"Транспонирование                |   {transpose_cpu:.4f}   |   {transpose_cuda:.4f}   |     {1/(transpose_cuda/transpose_cpu):.4f}")
print(f"Вычисление суммы всех элементов |   {sum_cpu:.4f}   |   {sum_cuda:.4f}   |     {1/(sum_cuda/sum_cpu):.4f}")
```

    Операция                        | CPU (мс) | GPU (мс) | Ускорение (раз)
    Матричное умножение             |   4.5630   |   0.6824   |     6.6865
    Поэлементное сложение           |   0.3290   |   0.0085   |     38.8174
    Поэлементное умножение          |   0.3099   |   0.0084   |     36.7023
    Транспонирование                |   0.0283   |   0.0001   |     362.3039
    Вычисление суммы всех элементов |   0.0958   |   0.0040   |     23.7786
    


```python
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
cuda, cpu = execute_operations(tensor128x512x512, tensor128x512x512)

matmul_cuda = cuda[0]
add_cuda = cuda[1]
mul_cuda = cuda[2]
transpose_cuda = cuda[3]
sum_cuda = cuda[4]

matmul_cpu = cpu[0]
add_cpu = cpu[1]
mul_cpu = cpu[2]
transpose_cpu = cpu[3]
sum_cpu = cpu[4]

# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде
print(f"Операция                        | CPU (мс) | GPU (мс) | Ускорение (раз)")
print(f"Матричное умножение             |   {matmul_cpu:.4f}   |   {matmul_cuda:.4f}   |     {1/(matmul_cuda/matmul_cpu):.4f}")
print(f"Поэлементное сложение           |   {add_cpu:.4f}   |   {add_cuda:.4f}   |     {1/(add_cuda/add_cpu):.4f}")
print(f"Поэлементное умножение          |   {mul_cpu:.4f}   |   {mul_cuda:.4f}   |     {1/(mul_cuda/mul_cpu):.4f}")
print(f"Транспонирование                |   {transpose_cpu:.4f}   |   {transpose_cuda:.4f}   |     {1/(transpose_cuda/transpose_cpu):.4f}")
print(f"Вычисление суммы всех элементов |   {sum_cpu:.4f}   |   {sum_cuda:.4f}   |     {1/(sum_cuda/sum_cpu):.4f}")
```

    Операция                        | CPU (мс) | GPU (мс) | Ускорение (раз)
    Матричное умножение             |   1.0443   |   0.3135   |     3.3310
    Поэлементное сложение           |   0.1650   |   0.0043   |     38.5416
    Поэлементное умножение          |   0.1571   |   0.0042   |     36.9793
    Транспонирование                |   0.0226   |   0.0000   |     454.5428
    Вычисление суммы всех элементов |   0.0461   |   0.0021   |     22.3040
    


```python
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
cuda, cpu = execute_operations(tensor256x256x256, tensor256x256x256)

matmul_cuda = cuda[0]
add_cuda = cuda[1]
mul_cuda = cuda[2]
transpose_cuda = cuda[3]
sum_cuda = cuda[4]

matmul_cpu = cpu[0]
add_cpu = cpu[1]
mul_cpu = cpu[2]
transpose_cpu = cpu[3]
sum_cpu = cpu[4]

# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде
print(f"Операция                        | CPU (мс) | GPU (мс) | Ускорение (раз)")
print(f"Матричное умножение             |   {matmul_cpu:.4f}   |   {matmul_cuda:.4f}   |     {1/(matmul_cuda/matmul_cpu):.4f}")
print(f"Поэлементное сложение           |   {add_cpu:.4f}   |   {add_cuda:.4f}   |     {1/(add_cuda/add_cpu):.4f}")
print(f"Поэлементное умножение          |   {mul_cpu:.4f}   |   {mul_cuda:.4f}   |     {1/(mul_cuda/mul_cpu):.4f}")
print(f"Транспонирование                |   {transpose_cpu:.4f}   |   {transpose_cuda:.4f}   |     {1/(transpose_cuda/transpose_cpu):.4f}")
print(f"Вычисление суммы всех элементов |   {sum_cpu:.4f}   |   {sum_cuda:.4f}   |     {1/(sum_cuda/sum_cpu):.4f}")
```

    Операция                        | CPU (мс) | GPU (мс) | Ускорение (раз)
    Матричное умножение             |   0.2913   |   0.0938   |     3.1047
    Поэлементное сложение           |   0.0822   |   0.0021   |     38.5990
    Поэлементное умножение          |   0.0760   |   0.0021   |     36.0512
    Транспонирование                |   0.0067   |   0.0000   |     149.6846
    Вычисление суммы всех элементов |   0.0234   |   0.0011   |     20.5090
    

## 3.4 Анализ результатов

### Проанализируйте результаты:
### - Какие операции получают наибольшее ускорение на GPU?
Поэлементное умножение — ускорение в ~30 раз.
Транспонирвоание — ускорение >100 раз.
Поэлементное сложение — ускорение в ~30 раз.
Эти операции хорошо распараллеливаются, так как состоят из большого количества независимых простых вычислений, которые GPU выполняет параллельно.
### - Почему некоторые операции могут быть медленнее на GPU?
Для очень быстрых операций на CPU накладные расходы на передачу данных и запуск вычислений на GPU могут превышать выигрыш.
Малые размеры данных не дают GPU раскрыть весь потенциал параллелизма.
### - Как размер матриц влияет на ускорение?
Чем больше размер матриц, тем эффективнее GPU использует параллелизм и тем выше ускорение.
### - Что происходит при передаче данных между CPU и GPU?
Если вы вызываете операцию на GPU, а данные хранятся на CPU, PyTorch автоматически копирует их на видеокарту, что занимает дополнительное время.
Передача данных между оперативной памятью (CPU) и видеопамятью (GPU) осуществляется через шину PCI Express и является относительно медленной операцией по сравнению с вычислениями внутри самого GPU.

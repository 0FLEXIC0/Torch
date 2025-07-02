import torch as t
import math

def print_tensor (tensor, name):
    print(name, tensor, tensor.dtype, tensor.ndim)
"""
Выводит имя тензора, тензор, его размерность и тип данных

Аргументы: 
tensor (tensor): Тензор
name (string): имя тензора
Вывод: Имя тензора, тензор, тип данных, размерность
"""

# Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
rnd_tensor2_3x4 = t.randint(2, (3, 4))
print_tensor(rnd_tensor2_3x4, 'rnd_tensor2_3x4')
# - Тензор размером 2x3x4, заполненный нулями
zeros_tensor_2x3x4 = t.zeros(2, 3, 4)
print_tensor(zeros_tensor_2x3x4, 'zeros_tensor_2x3x4')
# - Тензор размером 5x5, заполненный единицами
ones_tensor_5x5 = t.ones(5, 5)
print_tensor(ones_tensor_5x5, 'ones_tensor_5x5')
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
rnd_tensor16_4x4 = t.randint(16, (2, 8)).reshape(4, 4)
print_tensor(rnd_tensor16_4x4, 'rnd_tensor16_4x4')

# Дано: тензор A размером 3x4 и тензор B размером 4x3
A = t.randint(11, (3, 4))
B = t.randint(11, (4, 3))
print_tensor(A, 'A')
print_tensor(B, 'B')
# Выполните:
# - Транспонирование тензора A
tras_tensor_A = A.T
print_tensor(tras_tensor_A, 'tras_tensor_A')
# - Матричное умножение A и B
mult_AandB_tensor = A @ B
print_tensor(mult_AandB_tensor, 'mult_AandB_tensor')
# - Поэлементное умножение A и транспонированного B
mult_AandBtrans_tensor = A * B.T
print_tensor(mult_AandBtrans_tensor, 'mult_AandBtrans_tensor')
# - Вычислите сумму всех элементов тензора A
sum_A = A.sum()
print_tensor(sum_A, 'sum_A')

# Создайте тензор размером 5x5x5
tensor_5x5x5 = t.randint(11, (5, 5, 5))
print_tensor(tensor_5x5x5, 'tensor_5x5x5')
# Извлеките:
# - Первую строку
first_string = tensor_5x5x5[:1, :1, :]
print_tensor(first_string, 'first_string')
# - Последний столбец
last_column = tensor_5x5x5[-1:, :, -1:]
print_tensor(last_column, 'last_column')
# - Подматрицу размером 2x2 из центра тензора
center_2x2 = tensor_5x5x5[2, 1:3, 1:3]
print_tensor(center_2x2, 'center_2x2')
# - Все элементы с четными индексами
even_elements = tensor_5x5x5[::2, ::2, ::2]
print_tensor(even_elements, 'even_elements')

# Создайте тензор размером 24 элемента
tensor_1x24 = t.randint(11, (24, ))
print_tensor(tensor_1x24, 'tensor_1x24')
# Преобразуйте его в формы:
# - 2x12
tensor_2x12 = tensor_1x24.reshape(2, 12)
print_tensor(tensor_2x12, 'tensor_2x12')
# - 3x8
tensor_3x8 = tensor_1x24.reshape(3, 8)
print_tensor(tensor_3x8, 'tensor_3x8')
# - 4x6
tensor_4x6 = tensor_1x24.reshape(4, 6)
print_tensor(tensor_4x6, 'tensor_4x6')
# - 2x3x4
tensor_2x3x4 = tensor_1x24.reshape(2, 3, 4)
print_tensor(tensor_2x3x4, 'tensor_2x3x4')
# - 2x2x2x3
tensor_2x2x2x3 = tensor_1x24.reshape(2, 2, 2, 3)
print_tensor(tensor_2x2x2x3, 'tensor_2x2x2x3')

# Создайте тензоры x, y, z с requires_grad=True
x = t.tensor(1.0, requires_grad=True)
y = t.tensor(2.0, requires_grad=True)
z = t.tensor(3.0, requires_grad=True)
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z
# Найдите градиенты по всем переменным
f.backward()
print(f"Градиент по x: {x.grad}")
print(f"Градиент по y: {y.grad}")
print(f"Градиент по z: {z.grad}")
# Проверьте результат аналитически
# Аналитически результат совпал

# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
# Найдите градиенты по w и b
def mse_with_gradients(x, y_true, w, b):
    """
    Вычисляет MSE и градиенты по параметрам w и b с использованием PyTorch.
    
    Аргументы:
    x (torch.Tensor): Входные значения (тензор)
    y_true (torch.Tensor): Истинные значения (тензор)
    w (torch.Tensor): Весовой коэффициент (требует градиент)
    b (torch.Tensor): Смещение (требует градиент)
    
    Возвращает:
    tuple: (MSE, grad_w, grad_b)
    """
    y_pred = w * x + b
    mse = t.mean((y_pred - y_true)**2)
    
    mse.backward()
    grad_w = w.grad
    grad_b = b.grad
    
    return mse.item(), grad_w, grad_b

x = t.tensor([1.0, 2.0, 3.0])
y_true = t.tensor([2.0, 4.0, 6.0])
w = t.tensor(0.5, requires_grad=True)
b = t.tensor(0.1, requires_grad=True)

mse, grad_w, grad_b = mse_with_gradients(x, y_true, w, b)
print(f"MSE: {mse:.4f}")
print(f"Градиент по w: {grad_w:.4f}")
print(f"Градиент по b: {grad_b:.4f}")

# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
# Найдите градиент df/dx
x = 2.0
f = math.sin(x**2 + 1)  # f(2) = sin(4 + 1) = sin(5) ≈ -0.9589
df_dx = math.cos(x**2 + 1) * 2 * x  # cos(5)*4 ≈ 0.2837*4 ≈ 1.1346

print(f"Функция = {f:.4f}")     # -0.9589
print(f"Градиент = {df_dx:.4f}")  # 1.1346
# Проверьте результат с помощью torch.autograd.grad
x = t.tensor(2.0, requires_grad=True)
f = t.sin(x**2 + 1)
f.backward()
gradient = x.grad

print(f"PyTorch Функция = {f.item():.4f}")       # -0.9589
print(f"PyTorch Градиент = {gradient:.4f}")   # 1.1346
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Определите путь к папке NDVI
NDVI_DIR = "ndvi"

def load_and_normalize_ndvi(file_path, original_min=0, original_max=89):
    """Загружает и нормализует данные NDVI в диапазон от -1 до 1."""
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(float)
        data[data == 0] = np.nan  # Убираем нулевые значения, если это "пустые" значения
        # Нормализуем данные в диапазон -1 до 1
        normalized_data = 2 * (data - original_min) / (original_max - original_min) - 1
        return normalized_data

# Инициализация массива для хранения суммы нормализованных значений NDVI и счетчика валидных значений
sum_ndvi = None
count_ndvi = None

# Обработка всех файлов в папке NDVI
for file_name in os.listdir(NDVI_DIR):
    file_path = os.path.join(NDVI_DIR, file_name)
    
    # Загрузка и нормализация данных NDVI
    ndvi_data = load_and_normalize_ndvi(file_path)
    
    # Инициализация массивов для первого файла
    if sum_ndvi is None:
        sum_ndvi = np.zeros_like(ndvi_data, dtype=float)
        count_ndvi = np.zeros_like(ndvi_data, dtype=int)
    
    # Добавляем нормализованные значения NDVI и увеличиваем счетчик для валидных значений
    valid_mask = ~np.isnan(ndvi_data)
    sum_ndvi[valid_mask] += ndvi_data[valid_mask]
    count_ndvi[valid_mask] += 1

# Рассчитываем среднее NDVI, избегая деления на ноль
average_ndvi = np.divide(sum_ndvi, count_ndvi, where=(count_ndvi != 0))

# Визуализация нормализованной средней карты NDVI
plt.figure(figsize=(10, 6))
plt.imshow(average_ndvi, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(label="Средний NDVI (нормализованный)")
plt.title("Средняя нормализованная карта NDVI по всем временным кадрам")
plt.xlabel("Широта")
plt.ylabel("Долгота")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

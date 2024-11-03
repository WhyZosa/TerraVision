import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Определите путь к папке NDVI
NDVI_DIR = "ndvi"

def classify_ndvi(data):
    """
    Классифицирует значения NDVI по типам земельных участков.
    """
    classification = np.zeros_like(data)
    classification[(data < 0)] = 1  # Вода
    classification[(data >= 0) & (data < 0.2)] = 2  # Пустыня/городские зоны
    classification[(data >= 0.2) & (data < 0.4)] = 3  # Редкая растительность/кустарники
    classification[(data >= 0.4) & (data < 0.6)] = 4  # Средняя растительность
    classification[(data >= 0.6)] = 5  # Плотная растительность (леса)
    return classification

def load_ndvi(file_path):
    """Загружает растровый файл NDVI и нормализует данные."""
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(float)
        data[data == 0] = np.nan  # Убираем нулевые значения, если это "пустые" значения
        return data

# Выберем один файл NDVI для тестирования классификации
sample_file = os.path.join(NDVI_DIR, os.listdir(NDVI_DIR)[0])  # Используем первый файл в папке NDVI
ndvi_data = load_ndvi(sample_file)

# Классификация NDVI
classified_ndvi = classify_ndvi(ndvi_data)

# Определение цветовой схемы и меток
cmap = ListedColormap(['blue', 'green', 'orange', 'brown', 'darkgreen'])
labels = ['Вода', 'Пустыня/городские зоны', 'Редкая растительность/кустарники', 'Средняя растительность', 'Плотная растительность (леса)']
colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'brown', 5: 'darkgreen'}

# Визуализация классификации
plt.figure(figsize=(10, 6))
im = plt.imshow(classified_ndvi, cmap=cmap)
cbar = plt.colorbar(im, ticks=[1, 2, 3, 4, 5])
cbar.ax.set_yticklabels(labels)  # Установка меток для цветовой шкалы
plt.title("Классификация земельных участков на основе NDVI")
plt.xlabel("Широта")
plt.ylabel("Долгота")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

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

# Классификация всех файлов NDVI и объединение результатов
all_classified_ndvi = []

for ndvi_file in sorted(os.listdir(NDVI_DIR)):
    ndvi_path = os.path.join(NDVI_DIR, ndvi_file)
    ndvi_data = load_ndvi(ndvi_path)
    classified_ndvi = classify_ndvi(ndvi_data)
    all_classified_ndvi.append(classified_ndvi)

# Создаем среднюю классификацию по всем временным кадрам
average_classification = np.nanmean(np.dstack(all_classified_ndvi), axis=2)

# Определение цветовой схемы и меток
cmap = ListedColormap(['blue', 'green', 'orange', 'brown', 'darkgreen'])
labels = ['Вода', 'Пустыня/городские зоны', 'Редкая растительность/кустарники', 'Средняя растительность', 'Плотная растительность (леса)']

# Визуализация средней классификации
plt.figure(figsize=(10, 6))
im = plt.imshow(average_classification, cmap=cmap)
cbar = plt.colorbar(im, ticks=[1, 2, 3, 4, 5])
cbar.ax.set_yticklabels(labels)  # Установка меток для цветовой шкалы
plt.title("Средняя классификация земельных участков на основе NDVI")
plt.xlabel("Широта")
plt.ylabel("Долгота")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

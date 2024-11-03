import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
import numpy as np
import rasterio

# Определите пути к папке NDVI
NDVI_DIR = "ndvi"

def load_and_reproject_raster(file_path, target_crs, target_resolution=None):
    """Загружает растровый файл."""
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile
        return data, profile

def load_all_rasters(data_dir):
    """Загружает все растровые файлы из указанной директории."""
    data_dict = {}
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.tif'):
                file_path = os.path.join(root, file)
                data, profile = load_and_reproject_raster(file_path, "EPSG:4326")
                data_dict[file] = {'data': data, 'profile': profile}
    return data_dict

def normalize_ndvi(data):
    """Нормализует данные NDVI к диапазону -1 до 1."""
    data = data.astype(float)
    data[data == 0] = np.nan
    max_val, min_val = np.nanmax(data), np.nanmin(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return 2 * (data - min_val) / (max_val - min_val) - 1

# Загрузка данных NDVI
ndvi_data = load_all_rasters(NDVI_DIR)

# Расчет среднего NDVI для каждого изображения и парсинг даты из имени файла
average_ndvi = {}
for file_name, content in ndvi_data.items():
    ndvi_values = content['data']
    normalized_values = normalize_ndvi(ndvi_values)
    average_ndvi[file_name] = np.nanmean(normalized_values)

# Преобразование имен файлов в даты и сортировка данных по времени
dates = []
ndvi_means = []
for file_name, avg in average_ndvi.items():
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", file_name)
    if date_match:
        date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
        dates.append(date)
        ndvi_means.append(avg)

# Сортировка данных по датам
dates, ndvi_means = zip(*sorted(zip(dates, ndvi_means)))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(dates, ndvi_means, marker='o', color='b', linestyle='-', linewidth=2, markersize=4)
plt.title("Тренд среднего NDVI по времени")
plt.xlabel("Дата")
plt.ylabel("Средний NDVI")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

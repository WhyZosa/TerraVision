import os
import re
from datetime import datetime
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Определите пути к папкам с данными
NDVI_DIR = "ndvi"
SAVI_DIR = "savi"
SWIR_DIR = "swir"

def load_and_reproject_raster(file_path):
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
                data, profile = load_and_reproject_raster(file_path)
                data_dict[file] = {'data': data, 'profile': profile}
    return data_dict

def normalize_index(data):
    """Нормализует индекс к диапазону от -1 до 1, игнорируя нулевые значения."""
    data = data.astype(float)
    data[data == 0] = np.nan  # Маскируем нулевые значения
    max_val, min_val = np.nanmax(data), np.nanmin(data)
    if max_val == min_val:
        return np.zeros_like(data)  # Если все значения одинаковы, возвращаем массив нулей
    return 2 * (data - min_val) / (max_val - min_val) - 1

def calculate_average_index(data_dict):
    """Рассчитывает среднее значение индекса для каждого файла."""
    average_index = {}
    for file_name, content in data_dict.items():
        index_values = content['data']
        normalized_values = normalize_index(index_values)
        average_index[file_name] = np.nanmean(normalized_values)
    return average_index

def extract_dates_and_sort(average_index):
    """Извлекает даты из имен файлов и сортирует данные по времени."""
    dates, index_means = [], []
    for file_name, avg in average_index.items():
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", file_name)
        if date_match:
            date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            dates.append(date)
            index_means.append(avg)
    dates, index_means = zip(*sorted(zip(dates, index_means)))
    return dates, index_means

# Загрузка данных
ndvi_data = load_all_rasters(NDVI_DIR)
savi_data = load_all_rasters(SAVI_DIR)
swir_data = load_all_rasters(SWIR_DIR)

# Расчет среднего значения для NDVI, SAVI и SWIR
average_ndvi = calculate_average_index(ndvi_data)
average_savi = calculate_average_index(savi_data)
average_swir = calculate_average_index(swir_data)

# Преобразование и сортировка данных по времени
dates_ndvi, ndvi_means = extract_dates_and_sort(average_ndvi)
dates_savi, savi_means = extract_dates_and_sort(average_savi)
dates_swir, swir_means = extract_dates_and_sort(average_swir)

# Построение графиков для NDVI, SAVI и SWIR
plt.figure(figsize=(12, 6))
plt.plot(dates_ndvi, ndvi_means, marker='o', color='b', linestyle='-', label='NDVI')
plt.plot(dates_savi, savi_means, marker='s', color='g', linestyle='-', label='SAVI')
plt.plot(dates_swir, swir_means, marker='^', color='r', linestyle='-', label='SWIR')
plt.title("Изменения индексов NDVI, SAVI и SWIR по времени")
plt.xlabel("Дата")
plt.ylabel("Среднее значение индекса")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

# Определите пути к папкам с данными
NDVI_DIR = os.path.join("ndvi")
SAVI_DIR = os.path.join("savi")
SWIR_DIR = os.path.join("swir")

def load_and_reproject_raster(file_path, target_crs, target_resolution=None):
    """
    Загружает растровый файл, выполняет репроекцию и, при необходимости, ресемплинг.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile
        return data, profile

def load_all_rasters(data_dir):
    """
    Загружает все растровые файлы из указанной директории.
    """
    data_dict = {}
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.tif'):
                file_path = os.path.join(root, file)
                data, profile = load_and_reproject_raster(file_path, "EPSG:4326")
                data_dict[file] = {'data': data, 'profile': profile}
    return data_dict

def plot_raster(data, title=""):
    """Построение растра для визуальной проверки."""
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap='viridis', vmin=np.nanmin(data), vmax=np.nanmax(data))
    plt.colorbar()
    plt.title(title)
    plt.show()

# Нормализация данных NDVI к диапазону -1 до 1
def normalize_ndvi(data):
    """Нормализует данные NDVI к диапазону -1 до 1."""
    data = data.astype(float)
    data[data == 0] = np.nan  # Маскируем нулевые значения, если это "пустые" значения
    max_val, min_val = np.nanmax(data), np.nanmin(data)
    if max_val == min_val:
        return np.zeros_like(data)  # Если все значения одинаковы, возвращаем массив нулей
    return 2 * (data - min_val) / (max_val - min_val) - 1

# Загружаем данные из NDVI, SAVI и SWIR
ndvi_data = load_all_rasters(NDVI_DIR)
savi_data = load_all_rasters(SAVI_DIR)
swir_data = load_all_rasters(SWIR_DIR)

# Выводим количество файлов для проверки
print(f"Загружено {len(ndvi_data)} файлов NDVI, {len(savi_data)} файлов SAVI, {len(swir_data)} файлов SWIR.")

# Пример нормализации и отображения одного из файлов NDVI
sample_file = list(ndvi_data.keys())[0]
data = ndvi_data[sample_file]['data']

# Применение нормализации к одному из NDVI файлов
normalized_ndvi = normalize_ndvi(data)

# Отобразим нормализованный NDVI для проверки
plot_raster(normalized_ndvi, title=f"Normalized {sample_file}")

# Расчет среднего NDVI для каждого изображения в папке NDVI
average_ndvi = {}
for file_name, content in ndvi_data.items():
    ndvi_values = content['data']
    normalized_values = normalize_ndvi(ndvi_values)
    average_ndvi[file_name] = np.nanmean(normalized_values)  # Среднее, игнорируя NaN

# Вывод среднего значения NDVI по каждому файлу
for file_name, avg in average_ndvi.items():
    print(f"{file_name}: Средний NDVI = {avg:.4f}")

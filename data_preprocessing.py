import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

# Определите пути к папкам с данными
NDVI_DIR = r"ndvi"
SAVI_DIR = r"savi"
SWIR_DIR = r"swir"

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
    plt.imshow(data, cmap='viridis', vmin=np.min(data), vmax=np.max(data))
    plt.colorbar()
    plt.title(title)
    plt.show()

# Загружаем данные из NDVI, SAVI и SWIR
ndvi_data = load_all_rasters(NDVI_DIR)
savi_data = load_all_rasters(SAVI_DIR)
swir_data = load_all_rasters(SWIR_DIR)

# Выводим количество файлов для проверки
print(f"Загружено {len(ndvi_data)} файлов NDVI, {len(savi_data)} файлов SAVI, {len(swir_data)} файлов SWIR.")

# Пример отображения одного из файлов NDVI
sample_file = list(ndvi_data.keys())[0]
data = ndvi_data[sample_file]['data']

# Проверка уникальных значений
unique_values = np.unique(data)
print("Уникальные значения в данных:", unique_values)

# Выводим минимум и максимум для проверки диапазона значений
print("Минимум:", np.min(data))
print("Максимум:", np.max(data))

# Проверка наличия маски данных
with rasterio.open(os.path.join(NDVI_DIR, sample_file)) as src:
    if src.count > 1:
        print("Маска присутствует")
        mask = src.read_masks(1)
        unique_mask_values = np.unique(mask)
        print("Уникальные значения маски:", unique_mask_values)
    else:
        print("Маска отсутствует")

# Построение растра
plot_raster(data, title=sample_file)

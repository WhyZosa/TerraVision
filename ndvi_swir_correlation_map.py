import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Определите пути к папкам NDVI и SWIR
NDVI_DIR = "ndvi"
SWIR_DIR = "swir"

def load_data(file_path):
    """Загружает растровый файл данных."""
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(float)
        data[data == 0] = np.nan  # Убираем нулевые значения
        return data

# Инициализация списков для хранения всех значений NDVI и SWIR
all_ndvi_values = []
all_swir_values = []

# Загрузка данных NDVI и SWIR и добавление их в списки
for ndvi_file, swir_file in zip(sorted(os.listdir(NDVI_DIR)), sorted(os.listdir(SWIR_DIR))):
    ndvi_path = os.path.join(NDVI_DIR, ndvi_file)
    swir_path = os.path.join(SWIR_DIR, swir_file)
    
    ndvi_data = load_data(ndvi_path)
    swir_data = load_data(swir_path)
    
    all_ndvi_values.append(ndvi_data)
    all_swir_values.append(swir_data)

# Преобразуем списки в трехмерные массивы для расчета корреляции
ndvi_stack = np.dstack(all_ndvi_values)
swir_stack = np.dstack(all_swir_values)

# Рассчитываем корреляцию между NDVI и SWIR по каждому пикселю
correlation_map = np.full(ndvi_stack[:, :, 0].shape, np.nan)  # Инициализация карты с NaN

for i in range(ndvi_stack.shape[0]):
    for j in range(ndvi_stack.shape[1]):
        ndvi_series = ndvi_stack[i, j, :]
        swir_series = swir_stack[i, j, :]
        valid_mask = ~np.isnan(ndvi_series) & ~np.isnan(swir_series)
        
        if valid_mask.sum() > 1:
            if np.std(ndvi_series[valid_mask]) > 0 and np.std(swir_series[valid_mask]) > 0:
                correlation_map[i, j] = np.corrcoef(ndvi_series[valid_mask], swir_series[valid_mask])[0, 1]
            else:
                correlation_map[i, j] = np.nan  # Устанавливаем NaN для некорректных значений

# Визуализация карты корреляции
plt.figure(figsize=(10, 6))
plt.imshow(correlation_map, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label="Коэффициент корреляции")
plt.title("Карта корреляции между NDVI и SWIR")
plt.xlabel("Широта")
plt.ylabel("Долгота")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

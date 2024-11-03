import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Определите пути к папкам NDVI и SAVI
NDVI_DIR = "ndvi"
SAVI_DIR = "savi"

def load_data(file_path):
    """Загружает растровый файл данных."""
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(float)
        data[data == 0] = np.nan  # Убираем нулевые значения
        return data

# Инициализация списков для хранения всех значений NDVI и SAVI
all_ndvi_values = []
all_savi_values = []

# Загрузка данных NDVI и SAVI и добавление их в списки
for ndvi_file, savi_file in zip(sorted(os.listdir(NDVI_DIR)), sorted(os.listdir(SAVI_DIR))):
    ndvi_path = os.path.join(NDVI_DIR, ndvi_file)
    savi_path = os.path.join(SAVI_DIR, savi_file)
    
    ndvi_data = load_data(ndvi_path)
    savi_data = load_data(savi_path)
    
    all_ndvi_values.append(ndvi_data)
    all_savi_values.append(savi_data)

# Преобразуем списки в трехмерные массивы для расчета корреляции
ndvi_stack = np.dstack(all_ndvi_values)
savi_stack = np.dstack(all_savi_values)

# Рассчитываем корреляцию между NDVI и SAVI по каждому пикселю
correlation_map = np.full(ndvi_stack[:, :, 0].shape, np.nan)  # Инициализация карты с NaN

for i in range(ndvi_stack.shape[0]):
    for j in range(ndvi_stack.shape[1]):
        ndvi_series = ndvi_stack[i, j, :]
        savi_series = savi_stack[i, j, :]
        valid_mask = ~np.isnan(ndvi_series) & ~np.isnan(savi_series)
        
        if valid_mask.sum() > 1:
            # Проверка, что стандартное отклонение не равно нулю
            if np.std(ndvi_series[valid_mask]) > 0 and np.std(savi_series[valid_mask]) > 0:
                correlation_map[i, j] = np.corrcoef(ndvi_series[valid_mask], savi_series[valid_mask])[0, 1]
            else:
                correlation_map[i, j] = np.nan  # Устанавливаем NaN для некорректных значений

# Визуализация карты корреляции
plt.figure(figsize=(10, 6))
plt.imshow(correlation_map, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label="Коэффициент корреляции")
plt.title("Карта корреляции между NDVI и SAVI")
plt.xlabel("Широта")
plt.ylabel("Долгота")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

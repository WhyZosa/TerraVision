import os
import numpy as np
import pandas as pd
import rasterio

# Определение путей к папкам с индексами
NDVI_DIR = "ndvi"
SAVI_DIR = "savi"
SWIR_DIR = "swir"

def calculate_statistics(data_stack):
    """Вычисляет среднее, стандартное отклонение и медиану для каждого индекса."""
    mean_val = np.nanmean(data_stack, axis=2)
    std_val = np.nanstd(data_stack, axis=2)
    median_val = np.nanmedian(data_stack, axis=2)
    return mean_val, std_val, median_val

def load_data_stack(directory):
    """Загружает данные из всех файлов в директории и создает стек."""
    data_list = []
    for file_name in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(float)
            data[data == 0] = np.nan  # Убираем нулевые значения
            data_list.append(data)
    return np.dstack(data_list)

# Загрузка данных и расчет статистики
ndvi_stack = load_data_stack(NDVI_DIR)
savi_stack = load_data_stack(SAVI_DIR)
swir_stack = load_data_stack(SWIR_DIR)

ndvi_mean, ndvi_std, ndvi_median = calculate_statistics(ndvi_stack)
savi_mean, savi_std, savi_median = calculate_statistics(savi_stack)
swir_mean, swir_std, swir_median = calculate_statistics(swir_stack)

# Создаем таблицу с результатами
summary_data = {
    'Индекс': ['NDVI', 'SAVI', 'SWIR'],
    'Среднее': [np.nanmean(ndvi_mean), np.nanmean(savi_mean), np.nanmean(swir_mean)],
    'Стандартное отклонение': [np.nanmean(ndvi_std), np.nanmean(savi_std), np.nanmean(swir_std)],
    'Медиана': [np.nanmean(ndvi_median), np.nanmean(savi_median), np.nanmean(swir_median)]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df)

# Сохранение данных в CSV файл
output_file = "summary_statistics.csv"
summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Данные сохранены в файл: {output_file}")

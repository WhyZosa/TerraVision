import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

# Определите пути к папкам NDVI, SAVI и SWIR
NDVI_DIR = "ndvi"
SAVI_DIR = "savi"
SWIR_DIR = "swir"

def load_index_data(data_dir):
    """Загружает данные индекса и возвращает DataFrame с датами и значениями индекса."""
    dates = []
    values = []
    for file_name in sorted(os.listdir(data_dir)):
        date_str = file_name.split('_')[-1].split('.')[0]  # Извлекаем дату из имени файла
        date = datetime.strptime(date_str, '%Y-%m-%d')
        dates.append(date)

        file_path = os.path.join(data_dir, file_name)
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(float)
            data[data == 0] = np.nan
            values.append(np.nanmean(data))  # Рассчитываем среднее значение индекса для каждой даты
    
    # Создаем DataFrame с указанием частоты 'ME' (месячная частота с конкретной датой окончания)
    return pd.DataFrame({"Date": dates, "Value": values}).set_index("Date").asfreq('ME')

def forecast_sarima(data, forecast_periods=12):
    """Прогнозирует временной ряд с использованием модели SARIMA."""
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = model.fit(disp=False)
    forecast = sarima_fit.get_forecast(steps=forecast_periods)
    forecast_index = pd.date_range(data.index[-1] + pd.Timedelta(days=30), periods=forecast_periods, freq='ME')
    forecast_values = forecast.predicted_mean
    return forecast_index, forecast_values

# Загрузка данных для NDVI, SAVI и SWIR
ndvi_data = load_index_data(NDVI_DIR)
savi_data = load_index_data(SAVI_DIR)
swir_data = load_index_data(SWIR_DIR)

# Прогнозирование на следующие 12 месяцев для каждого индекса
forecast_periods = 12
ndvi_forecast_index, ndvi_forecast = forecast_sarima(ndvi_data['Value'], forecast_periods)
savi_forecast_index, savi_forecast = forecast_sarima(savi_data['Value'], forecast_periods)
swir_forecast_index, swir_forecast = forecast_sarima(swir_data['Value'], forecast_periods)

# Визуализация прогноза
plt.figure(figsize=(12, 8))

# NDVI
plt.subplot(3, 1, 1)
plt.plot(ndvi_data.index, ndvi_data['Value'], label="NDVI (фактические)", color="blue")
plt.plot(ndvi_forecast_index, ndvi_forecast, label="NDVI (прогноз)", color="orange", linestyle="--")
plt.title("Прогноз NDVI с помощью SARIMA")
plt.xlabel("Дата")
plt.ylabel("Средний NDVI")
plt.legend()

# SAVI
plt.subplot(3, 1, 2)
plt.plot(savi_data.index, savi_data['Value'], label="SAVI (фактические)", color="green")
plt.plot(savi_forecast_index, savi_forecast, label="SAVI (прогноз)", color="orange", linestyle="--")
plt.title("Прогноз SAVI с помощью SARIMA")
plt.xlabel("Дата")
plt.ylabel("Средний SAVI")
plt.legend()

# SWIR
plt.subplot(3, 1, 3)
plt.plot(swir_data.index, swir_data['Value'], label="SWIR (фактические)", color="red")
plt.plot(swir_forecast_index, swir_forecast, label="SWIR (прогноз)", color="orange", linestyle="--")
plt.title("Прогноз SWIR с помощью SARIMA")
plt.xlabel("Дата")
plt.ylabel("Средний SWIR")
plt.legend()

plt.tight_layout()
plt.show()

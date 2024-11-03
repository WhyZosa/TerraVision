# terra_vision_gui.py

import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QPushButton, QFileDialog, QTabWidget, QMessageBox, QHBoxLayout,
    QProgressDialog
)
from PyQt5.QtCore import Qt
import rasterio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TerraVisionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TerraVision")
        self.setGeometry(100, 100, 1000, 800)
        self.initUI()
        
        # Инициализируем списки для данных
        self.ndvi_files = []
        self.savi_files = []
        self.swir_files = []
        
        # Инициализируем данные
        self.ndvi_data = {}
        self.savi_data = {}
        self.swir_data = {}
        
        # Загрузка модели MLP для классификации земель
        self.mlp_model = None
        self.load_mlp_model()
    
    def initUI(self):
        # Создаем вкладки
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Вкладка для загрузки данных
        self.data_load_tab = QWidget()
        self.tabs.addTab(self.data_load_tab, "Загрузка данных")
        self.data_load_layout = QVBoxLayout()
        self.data_load_tab.setLayout(self.data_load_layout)

        # Создаем кнопки для загрузки данных NDVI, SAVI, SWIR
        self.load_ndvi_button = QPushButton("Загрузить данные NDVI")
        self.load_ndvi_button.clicked.connect(self.load_ndvi_data)
        self.data_load_layout.addWidget(self.load_ndvi_button)

        self.load_savi_button = QPushButton("Загрузить данные SAVI")
        self.load_savi_button.clicked.connect(self.load_savi_data)
        self.data_load_layout.addWidget(self.load_savi_button)

        self.load_swir_button = QPushButton("Загрузить данные SWIR")
        self.load_swir_button.clicked.connect(self.load_swir_data)
        self.data_load_layout.addWidget(self.load_swir_button)

        # Метки для отображения статуса загрузки
        self.ndvi_status_label = QLabel("Данные NDVI не загружены")
        self.data_load_layout.addWidget(self.ndvi_status_label)

        self.savi_status_label = QLabel("Данные SAVI не загружены")
        self.data_load_layout.addWidget(self.savi_status_label)

        self.swir_status_label = QLabel("Данные SWIR не загружены")
        self.data_load_layout.addWidget(self.swir_status_label)

        # Кнопка для начала анализа после загрузки данных
        self.start_analysis_button = QPushButton("Начать анализ")
        self.start_analysis_button.clicked.connect(self.start_analysis)
        self.start_analysis_button.setEnabled(False)
        self.data_load_layout.addWidget(self.start_analysis_button)

        # Вкладка для отображения результатов анализа
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Результаты анализа")
        self.analysis_layout = QVBoxLayout()
        self.analysis_tab.setLayout(self.analysis_layout)

        # Создаем область для графиков
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.analysis_layout.addWidget(self.canvas)

        # Добавляем кнопки для построения графиков
        self.buttons_layout = QHBoxLayout()
        self.analysis_layout.addLayout(self.buttons_layout)

        self.plot_ndvi_trend_button = QPushButton("Построить тренд NDVI")
        self.plot_ndvi_trend_button.clicked.connect(self.plot_ndvi_trend)
        self.plot_ndvi_trend_button.setEnabled(False)
        self.buttons_layout.addWidget(self.plot_ndvi_trend_button)

        self.plot_ndvi_savi_corr_button = QPushButton("Корреляция NDVI и SAVI")
        self.plot_ndvi_savi_corr_button.clicked.connect(self.plot_ndvi_savi_correlation)
        self.plot_ndvi_savi_corr_button.setEnabled(False)
        self.buttons_layout.addWidget(self.plot_ndvi_savi_corr_button)

        self.plot_ndvi_swir_corr_button = QPushButton("Корреляция NDVI и SWIR")
        self.plot_ndvi_swir_corr_button.clicked.connect(self.plot_ndvi_swir_correlation)
        self.plot_ndvi_swir_corr_button.setEnabled(False)
        self.buttons_layout.addWidget(self.plot_ndvi_swir_corr_button)

        self.forecast_indices_button = QPushButton("Прогноз индексов")
        self.forecast_indices_button.clicked.connect(self.forecast_indices)
        self.forecast_indices_button.setEnabled(False)
        self.buttons_layout.addWidget(self.forecast_indices_button)

        self.classify_land_button = QPushButton("Классификация земель")
        self.classify_land_button.clicked.connect(self.classify_land)
        self.classify_land_button.setEnabled(False)
        self.buttons_layout.addWidget(self.classify_land_button)

        # Добавляем новую кнопку "Рекомендации по посадке"
        self.planting_recommendation_button = QPushButton("Рекомендации по посадке")
        self.planting_recommendation_button.clicked.connect(self.recommend_planting)
        self.planting_recommendation_button.setEnabled(False)
        self.buttons_layout.addWidget(self.planting_recommendation_button)

        # Метка для отображения результатов
        self.analysis_result_label = QLabel("Пожалуйста, загрузите данные и начните анализ.")
        self.analysis_result_label.setAlignment(Qt.AlignCenter)
        self.analysis_layout.addWidget(self.analysis_result_label)

    def load_mlp_model(self):
        # Загружаем модель MLP
        try:
            import joblib
            self.mlp_model = joblib.load('mlp_model.pkl')
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель MLP:\n{e}")

    def load_ndvi_data(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите файлы данных NDVI", "", "TIFF Files (*.tif);;All Files (*)", options=options)
        if files:
            try:
                self.ndvi_files = files
                self.ndvi_data = self.load_all_rasters(self.ndvi_files)
                self.ndvi_status_label.setText(f"Загружено файлов NDVI: {len(files)}")
                self.check_all_data_loaded()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные NDVI:\n{e}")

    def load_savi_data(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите файлы данных SAVI", "", "TIFF Files (*.tif);;All Files (*)", options=options)
        if files:
            try:
                self.savi_files = files
                self.savi_data = self.load_all_rasters(self.savi_files)
                self.savi_status_label.setText(f"Загружено файлов SAVI: {len(files)}")
                self.check_all_data_loaded()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные SAVI:\n{e}")

    def load_swir_data(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите файлы данных SWIR", "", "TIFF Files (*.tif);;All Files (*)", options=options)
        if files:
            try:
                self.swir_files = files
                self.swir_data = self.load_all_rasters(self.swir_files)
                self.swir_status_label.setText(f"Загружено файлов SWIR: {len(files)}")
                self.check_all_data_loaded()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные SWIR:\n{e}")

    def load_all_rasters(self, file_list):
        data_dict = {}
        for file_path in file_list:
            file_name = os.path.basename(file_path)
            with rasterio.open(file_path) as src:
                data = src.read(1)
                profile = src.profile
            data_dict[file_name] = {'data': data, 'profile': profile}
        return data_dict

    def check_all_data_loaded(self):
        if self.ndvi_files and self.savi_files and self.swir_files:
            self.start_analysis_button.setEnabled(True)
        else:
            self.start_analysis_button.setEnabled(False)

    def start_analysis(self):
        try:
            self.analysis_result_label.setText("Анализ данных выполнен успешно.")
            # Активируем кнопки для дальнейшего анализа
            self.plot_ndvi_trend_button.setEnabled(True)
            self.plot_ndvi_savi_corr_button.setEnabled(True)
            self.plot_ndvi_swir_corr_button.setEnabled(True)
            self.forecast_indices_button.setEnabled(True)
            self.classify_land_button.setEnabled(True)
            self.planting_recommendation_button.setEnabled(True)
            # Переключаемся на вкладку с результатами анализа
            self.tabs.setCurrentWidget(self.analysis_tab)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить анализ данных:\n{e}")

    def plot_ndvi_trend(self):
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            dates = []
            ndvi_means = []
            for file_name, content in self.ndvi_data.items():
                ndvi_values = content['data']
                ndvi_values = ndvi_values.astype(float)
                ndvi_values[ndvi_values == 0] = np.nan
                avg = np.nanmean(ndvi_values)
                date_str = self.extract_date_from_filename(file_name)
                if date_str:
                    date = pd.to_datetime(date_str)
                    dates.append(date)
                    ndvi_means.append(avg)
            if dates:
                # Сортировка данных по датам
                dates, ndvi_means = zip(*sorted(zip(dates, ndvi_means)))
                # Построение графика
                ax.plot(dates, ndvi_means, marker='o', color='b', linestyle='-', linewidth=2, markersize=4)
                ax.set_title("Тренд среднего NDVI по времени")
                ax.set_xlabel("Дата")
                ax.set_ylabel("Средний NDVI")
                ax.grid(True)
                self.canvas.draw()
            else:
                QMessageBox.warning(self, "Предупреждение", "Не удалось извлечь даты из имен файлов NDVI.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить тренд NDVI:\n{e}")

    def extract_date_from_filename(self, filename):
        import re
        match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
        if match:
            return match.group(1)
        return None

    def plot_ndvi_savi_correlation(self):
        try:
            progress = QProgressDialog("Вычисление корреляции NDVI и SAVI...", None, 0, 0, self)
            progress.setWindowTitle("Пожалуйста, подождите")
            progress.setCancelButton(None)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            # Оптимизированный код для расчета корреляции
            ndvi_stack, savi_stack = self.get_stacks(self.ndvi_data, self.savi_data)
            if ndvi_stack is None or savi_stack is None:
                progress.close()
                return
            correlation_map = self.calculate_correlation_map_optimized(ndvi_stack, savi_stack)
            cax = ax.imshow(correlation_map, cmap='coolwarm', vmin=-1, vmax=1)
            self.figure.colorbar(cax, ax=ax, label="Коэффициент корреляции")
            ax.set_title("Карта корреляции между NDVI и SAVI")
            ax.axis('off')
            self.canvas.draw()

            progress.close()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить корреляцию NDVI и SAVI:\n{e}")

    def plot_ndvi_swir_correlation(self):
        try:
            progress = QProgressDialog("Вычисление корреляции NDVI и SWIR...", None, 0, 0, self)
            progress.setWindowTitle("Пожалуйста, подождите")
            progress.setCancelButton(None)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            # Оптимизированный код для расчета корреляции
            ndvi_stack, swir_stack = self.get_stacks(self.ndvi_data, self.swir_data)
            if ndvi_stack is None or swir_stack is None:
                progress.close()
                return
            correlation_map = self.calculate_correlation_map_optimized(ndvi_stack, swir_stack)
            cax = ax.imshow(correlation_map, cmap='coolwarm', vmin=-1, vmax=1)
            self.figure.colorbar(cax, ax=ax, label="Коэффициент корреляции")
            ax.set_title("Карта корреляции между NDVI и SWIR")
            ax.axis('off')
            self.canvas.draw()

            progress.close()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить корреляцию NDVI и SWIR:\n{e}")

    def get_stacks(self, data1, data2):
        # Убедимся, что файлы соответствуют друг другу
        keys1 = sorted(data1.keys())
        keys2 = sorted(data2.keys())

        if len(keys1) != len(keys2):
            QMessageBox.warning(self, "Предупреждение", "Количество файлов NDVI и других индексов не совпадает.")
            return None, None

        stack1 = []
        stack2 = []
        for key1, key2 in zip(keys1, keys2):
            arr1 = data1[key1]['data'].astype(float)
            arr1[arr1 == 0] = np.nan
            arr2 = data2[key2]['data'].astype(float)
            arr2[arr2 == 0] = np.nan
            stack1.append(arr1)
            stack2.append(arr2)
        stack1 = np.stack(stack1, axis=-1)
        stack2 = np.stack(stack2, axis=-1)
        return stack1, stack2

    def calculate_correlation_map_optimized(self, stack1, stack2):
        """
        Оптимизированная функция для расчета корреляции между двумя стековыми массивами.
        """
        # Проверяем, что размеры стеков совпадают
        if stack1.shape != stack2.shape:
            raise ValueError("Размеры стеков не совпадают.")

        # Получаем размеры массива
        rows, cols, bands = stack1.shape

        # Преобразуем в двумерные массивы
        stack1_flat = stack1.reshape(-1, bands)
        stack2_flat = stack2.reshape(-1, bands)

        # Создаем маску валидных значений
        valid_mask = ~np.isnan(stack1_flat) & ~np.isnan(stack2_flat)

        # Инициализируем массив для хранения корреляций
        correlation_flat = np.full(stack1_flat.shape[0], np.nan)

        # Вычисляем средние и стандартные отклонения
        with np.errstate(invalid='ignore'):
            mean1 = np.nanmean(stack1_flat, axis=1)
            mean2 = np.nanmean(stack2_flat, axis=1)
            std1 = np.nanstd(stack1_flat, axis=1)
            std2 = np.nanstd(stack2_flat, axis=1)

        # Вычисляем ковариацию
        with np.errstate(invalid='ignore', divide='ignore'):
            cov = np.nansum((stack1_flat - mean1[:, None]) * (stack2_flat - mean2[:, None]), axis=1) / (np.sum(valid_mask, axis=1) - 1)
            correlation_flat = cov / (std1 * std2)
            correlation_flat[(std1 == 0) | (std2 == 0) | (np.sum(valid_mask, axis=1) <= 1)] = np.nan

        # Преобразуем обратно в двумерный массив
        correlation_map = correlation_flat.reshape(rows, cols)
        return correlation_map

    def forecast_indices(self):
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            indices = {'NDVI': self.ndvi_data, 'SAVI': self.savi_data, 'SWIR': self.swir_data}
            colors = {'NDVI': 'blue', 'SAVI': 'green', 'SWIR': 'red'}
            for index_name, data_dict in indices.items():
                dates = []
                values = []
                for file_name, content in data_dict.items():
                    index_values = content['data']
                    index_values = index_values.astype(float)
                    index_values[index_values == 0] = np.nan
                    avg = np.nanmean(index_values)
                    date_str = self.extract_date_from_filename(file_name)
                    if date_str:
                        date = pd.to_datetime(date_str)
                        dates.append(date)
                        values.append(avg)
                    else:
                        print(f"Не удалось извлечь дату из имени файла: {file_name}")
                if dates:
                    # Создаём DataFrame и переиндексируем с использованием полной даты
                    data = pd.DataFrame({'Date': dates, index_name: values})
                    data.set_index('Date', inplace=True)
                    # Ресемплирование данных до ежемесячной частоты и интерполяция пропущенных значений
                    data = data.resample('MS').mean()
                    data[index_name] = data[index_name].interpolate(method='linear')
                    series = data[index_name]
                    # Проверка количества данных
                    if len(series) < 24:
                        QMessageBox.warning(self, "Предупреждение", f"Недостаточно данных для прогнозирования {index_name}. Требуется минимум 2 года данных.")
                        continue
                    # Прогнозирование
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                    sarima_fit = model.fit(disp=False)
                    forecast = sarima_fit.get_forecast(steps=12)
                    forecast_index = forecast.predicted_mean.index
                    forecast_values = forecast.predicted_mean.values
                    # Построение графика
                    ax.plot(series.index, series.values, label=f"{index_name} (фактические)", color=colors[index_name])
                    ax.plot(forecast_index, forecast_values, label=f"{index_name} (прогноз)", linestyle="--", color=colors[index_name])
                else:
                    QMessageBox.warning(self, "Предупреждение", f"Не удалось извлечь даты из имен файлов {index_name}.")
            ax.set_title("Прогноз индексов с помощью SARIMA")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Значение индекса")
            ax.legend()
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить прогнозирование:\n{e}")

    def classify_land(self):
        try:
            if self.mlp_model is None:
                QMessageBox.critical(self, "Ошибка", "Модель MLP не загружена.")
                return
            # Используем средние значения по всем загруженным данным
            ndvi = np.nanmean([np.nanmean(d['data']) for d in self.ndvi_data.values()])
            savi = np.nanmean([np.nanmean(d['data']) for d in self.savi_data.values()])
            swir = np.nanmean([np.nanmean(d['data']) for d in self.swir_data.values()])
            # Классифицируем тип земельного участка
            data_point = pd.DataFrame([[ndvi, savi, swir]], columns=['NDVI', 'SAVI', 'SWIR'])
            prediction = self.mlp_model.predict(data_point)
            class_mapping = {
                1: 'Вода',
                2: 'Пустыня/городские зоны',
                3: 'Редкая растительность/кустарники',
                4: 'Средняя растительность',
                5: 'Плотная растительность (леса)'
            }
            classification = class_mapping.get(prediction[0], "Неизвестный класс")
            QMessageBox.information(self, "Классификация земель", f"Тип земельного участка: {classification}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить классификацию:\n{e}")

    def recommend_planting(self):
        try:
            if self.mlp_model is None:
                QMessageBox.critical(self, "Ошибка", "Модель MLP не загружена.")
                return

            # Используем загруженные данные для генерации рекомендаций по посадке
            # Предполагаем, что mlp_classification.py подготовил необходимые данные и модель
            # Здесь мы можем использовать те же данные, что и для классификации земель

            # Получаем средние значения индексов
            ndvi = np.nanmean([np.nanmean(d['data']) for d in self.ndvi_data.values()])
            savi = np.nanmean([np.nanmean(d['data']) for d in self.savi_data.values()])
            swir = np.nanmean([np.nanmean(d['data']) for d in self.swir_data.values()])

            # Подготавливаем данные для модели
            data_point = pd.DataFrame([[ndvi, savi, swir]], columns=['NDVI', 'SAVI', 'SWIR'])

            # Получаем предсказание класса земель
            land_class = self.mlp_model.predict(data_point)[0]

            # На основе класса земель даём рекомендации по посадке
            recommendations = {
                1: "Рекомендуется посадка водных растений или создание водоёмов.",
                2: "Требуется улучшение почвы. Рекомендуется посадка засухоустойчивых растений.",
                3: "Подходящие культуры: кустарники и редкая растительность.",
                4: "Можно сажать разнообразные сельскохозяйственные культуры.",
                5: "Идеально подходит для лесных культур и плотной растительности."
            }

            recommendation = recommendations.get(land_class, "Нет рекомендаций для данного класса земель.")

            QMessageBox.information(self, "Рекомендации по посадке", recommendation)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось получить рекомендации по посадке:\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TerraVisionGUI()
    gui.show()
    sys.exit(app.exec_())

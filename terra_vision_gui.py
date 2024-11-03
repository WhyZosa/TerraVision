import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
import matplotlib.pyplot as plt
from use_mlp_model import classify_land_use  # Импортируем функцию классификации

class TerraVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TerraVision - Анализ и рекомендации по посадке")
        self.setGeometry(200, 200, 800, 600)

        # Инициализация интерфейса
        layout = QVBoxLayout()

        # Кнопка для загрузки NDVI данных
        self.ndvi_button = QPushButton("Загрузить NDVI данные")
        self.ndvi_button.clicked.connect(self.load_ndvi_data)
        layout.addWidget(self.ndvi_button)

        # Кнопка для загрузки SAVI данных
        self.savi_button = QPushButton("Загрузить SAVI данные")
        self.savi_button.clicked.connect(self.load_savi_data)
        layout.addWidget(self.savi_button)

        # Кнопка для загрузки SWIR данных
        self.swir_button = QPushButton("Загрузить SWIR данные")
        self.swir_button.clicked.connect(self.load_swir_data)
        layout.addWidget(self.swir_button)

        # Кнопка для получения рекомендаций
        self.recommendation_button = QPushButton("Рекомендации по посадке")
        self.recommendation_button.clicked.connect(self.show_recommendations)
        layout.addWidget(self.recommendation_button)

        # Поле для вывода результатов
        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        # Установка центрального виджета
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Переменные для хранения путей к файлам
        self.ndvi_value = None
        self.savi_value = None
        self.swir_value = None

    def load_ndvi_data(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите файлы NDVI", "", "TIFF files (*.tif)")
        if files:
            # В реальном сценарии тут нужно будет считать данные из файлов
            self.ndvi_value = 0.45  # Примерное значение
            print(f"Файлы NDVI загружены: {files}")

    def load_savi_data(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите файлы SAVI", "", "TIFF files (*.tif)")
        if files:
            # В реальном сценарии тут нужно будет считать данные из файлов
            self.savi_value = 0.25  # Примерное значение
            print(f"Файлы SAVI загружены: {files}")

    def load_swir_data(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите файлы SWIR", "", "TIFF files (*.tif)")
        if files:
            # В реальном сценарии тут нужно будет считать данные из файлов
            self.swir_value = 270  # Примерное значение
            print(f"Файлы SWIR загружены: {files}")

    def show_recommendations(self):
        # Проверка на наличие данных
        if self.ndvi_value is not None and self.savi_value is not None and self.swir_value is not None:
            # Используем функцию классификации из модели
            try:
                classification = classify_land_use(self.ndvi_value, self.savi_value, self.swir_value)
                self.result_label.setText(f"Тип земельного участка: {classification}")
                print(f"Тип земельного участка: {classification}")
            except Exception as e:
                self.result_label.setText(f"Ошибка при классификации: {e}")
                print(f"Ошибка при классификации: {e}")
        else:
            self.result_label.setText("Пожалуйста, загрузите данные для всех индексов.")
            print("Пожалуйста, загрузите данные для всех индексов.")

# Запуск приложения
app = QApplication(sys.argv)
window = TerraVisionApp()
window.show()
sys.exit(app.exec_())

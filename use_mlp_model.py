import joblib
import pandas as pd

# Загрузка модели
mlp_model = joblib.load('mlp_model.pkl')

def classify_land_use(ndvi, savi, swir):
    """
    Классифицирует тип земельного участка на основе индексов NDVI, SAVI и SWIR.
    """
    # Создаем DataFrame с именами признаков
    data = pd.DataFrame([[ndvi, savi, swir]], columns=['NDVI', 'SAVI', 'SWIR'])
    prediction = mlp_model.predict(data)
    
    class_mapping = {
        1: 'Вода',
        2: 'Пустыня/городские зоны',
        3: 'Редкая растительность/кустарники',
        4: 'Средняя растительность',
        5: 'Плотная растительность (леса)'
    }
    
    return class_mapping[prediction[0]]

# Пример использования
ndvi_value = 0.45
savi_value = 0.25
swir_value = 270
classification = classify_land_use(ndvi_value, savi_value, swir_value)
print(f"Тип земельного участка: {classification}")

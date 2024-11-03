import joblib
import pandas as pd

# Загрузка модели классификации
mlp_model = joblib.load('mlp_model.pkl')

def classify_land_use(ndvi, savi, swir):
    """
    Классифицирует тип земельного участка на основе индексов NDVI, SAVI и SWIR.
    """
    # Создаем DataFrame с именами признаков для модели
    data = pd.DataFrame([[ndvi, savi, swir]], columns=['NDVI', 'SAVI', 'SWIR'])
    prediction = mlp_model.predict(data)
    
    # Соответствие классов с типами земель
    class_mapping = {
        1: 'Вода',
        2: 'Пустыня/городские зоны',
        3: 'Редкая растительность/кустарники',
        4: 'Средняя растительность',
        5: 'Плотная растительность (леса)'
    }
    
    return class_mapping[prediction[0]]

def generate_recommendations(ndvi, savi, swir):
    """
    Генерирует рекомендации по посадке на основе типа земельного участка и значений индексов.
    """
    # Определение типа земельного участка
    land_type = classify_land_use(ndvi, savi, swir)
    
    # Генерация рекомендаций по типу земельного участка
    if land_type == 'Плотная растительность (леса)':
        recommendation = 'Поддерживайте текущие культуры, возможно, высадка лесных растений.'
    elif land_type == 'Средняя растительность':
        recommendation = 'Рекомендуется высаживать устойчивые сельскохозяйственные культуры.'
    elif land_type == 'Редкая растительность/кустарники':
        recommendation = 'Рекомендуется высаживать устойчивые к засухе растения.'
    elif land_type == 'Пустыня/городские зоны':
        recommendation = 'Зона требует улучшения состояния почвы перед посадкой культур.'
    elif land_type == 'Вода':
        recommendation = 'Подходит для аквакультуры или восстановления водных экосистем.'
    else:
        recommendation = 'Нет рекомендаций для данного типа земель.'

    return recommendation

# Пример использования
ndvi_value = 0.45
savi_value = 0.25
swir_value = 270
print(f"Рекомендации по посадке: {generate_recommendations(ndvi_value, savi_value, swir_value)}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Шаг 1: Подготовка данных
def generate_synthetic_data(num_samples=1000):
    """
    Генерирует синтетические данные с NDVI, SAVI, SWIR и метками классов.
    """
    np.random.seed(42)
    ndvi = np.random.uniform(-1, 1, num_samples)
    savi = np.random.uniform(0, 1, num_samples)
    swir = np.random.uniform(200, 300, num_samples)
    
    # Пример классификации: 1 - вода, 2 - пустыня/город, 3 - кустарники, 4 - средняя растительность, 5 - леса
    labels = np.where(ndvi < 0, 1, 
             np.where((ndvi >= 0) & (ndvi < 0.2), 2, 
             np.where((ndvi >= 0.2) & (ndvi < 0.4), 3, 
             np.where((ndvi >= 0.4) & (ndvi < 0.6), 4, 5))))
    
    data = pd.DataFrame({'NDVI': ndvi, 'SAVI': savi, 'SWIR': swir, 'Label': labels})
    return data

# Создаем синтетические данные
data = generate_synthetic_data(1000)

# Разделение данных на признаки и метки
X = data[['NDVI', 'SAVI', 'SWIR']]
y = data['Label']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 2: Создание модели MLP
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

# Обучение модели
mlp.fit(X_train, y_train)

# Шаг 3: Оценка модели
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Шаг 4: Сохранение модели и подготовка для интеграции
import joblib
joblib.dump(mlp, 'mlp_model.pkl')
print("Модель сохранена как 'mlp_model.pkl'")

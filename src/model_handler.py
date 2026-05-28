import os
import joblib
import pandas as pd
from src.transformers import StringAnomalyCleaner, ScreenResolutionTransformer, CityAgreggatorTransformer, WebTimeFeatureExtractor, BrandBasedOSImputer, FillerScalerDropper
# Путь к сохраненной модели
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/model.pkl')

def load_model():
    """Загружает модель из файла."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Файл модели не найден по пути: {MODEL_PATH}")
        
    with open(MODEL_PATH, 'rb') as pkl_file:
        model = joblib.load(pkl_file)
    return model

# Загружаем модель ОДИН РАЗ при импорте этого файла
_model = load_model()

def predict(features_dict: dict) -> int:
    """
    Принимает словарь с данными визита, 
    делает предсказание и возвращает 0 или 1.
    """
    # Превращаем словарь в DataFrame
    df = pd.DataFrame([features_dict])
    
    # Делаем предсказание (0 или 1)
    prediction = _model.predict(df)
    
    # Возвращаем как обычное число int
    return int(prediction[0])
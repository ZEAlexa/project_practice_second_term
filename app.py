import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from src.model_handler import load_model, predict

model = load_model()

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Проверка, что сервер жив и модель загружена."""
    return jsonify({"status": "UP", "model_loaded": model is not None}), 200

@app.route('/predict', methods=['POST'])
def predict_function():
    """
    Эндпоинт для прогнозирования поведения пользователя.
    
    Входные данные (JSON):
        features (list): Список из 18 признаков
            
    Выходные данные (JSON):
        prediction (int): 1 (Совершит целевое действие) или 0 (не совершит).
        
    Ошибки:
        400: Если передано не 18 признаков или тело запроса пустое.
        500: Внутренняя ошибка.
    """
    try:
        data = request.json

        if not data or 'features' not in data:
            return jsonify({"error": "No features provided"}), 400
        
        if len(data['features']) != 18:
            return jsonify({"error": f"Expected 18 features, got {len(data['features'])}"}), 400
        
        result = predict(data['features'])
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return 'Welcome to the service'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


import os
import requests
import pandas as pd
import numpy as np
import time

"""
    Тест предсказаний модели
"""
url = 'http://localhost:5000/predict'
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/test_sample.csv')

if __name__ == '__main__':
    try:
        df = pd.read_csv(DATA_PATH)
        
        random_row = np.random.randint(0, df.shape[0]-1)
        features = df.iloc[random_row].to_dict()

        features = {
            k: (None if pd.isna(v) else v) for k, v in features.items()
        }

        start_time = time.perf_counter()

        r = requests.post(url, json={'features': features})

        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000 

        print(r.status_code)
        print(f"Время отклика сервиса: {latency:.2f} мс")

        if r.status_code == 200:
            print(f"Предсказание модели: {r.json()['prediction']}")
        else:
            print(r.text)

    except Exception as e:
        print(f"Ошибка: {str(e)}")


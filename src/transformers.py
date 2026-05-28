from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, TargetEncoder

RANDOM_STATE = 42

class StringAnomalyCleaner(BaseEstimator, TransformerMixin):
    """
    Выявляет скрытые пропуски в данных
    """
    def __init__(self):
        # Список текстовых масок, которые на самом деле являются пропусками
        self.anomaly_markers = ['(none)', '(not set)', 'nan', 'none', 'not set']
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Проходим по всем колонкам
        for col in X_out.columns:
            # Работаем только с текстовыми (категориальными) признаками
            if X_out[col].dtype == 'object' or isinstance(X_out[col].dtype, pd.CategoricalDtype):

                X_out[col] = X_out[col].astype(str).str.strip()
                
                is_anomaly = X_out[col].str.lower().isin(self.anomaly_markers)
                X_out.loc[is_anomaly, col] = np.nan                
                
        return X_out
    
class ScreenResolutionTransformer(BaseEstimator, TransformerMixin):
    """
        Класс для кодирования категорий разрешения экрана в числовые признаки. 
        Добавляет новые числовые признаки:
            screen_width (ширина)
            screen_width (высота)
        Исходный device_screen_resolution удаляется
        Пропуски заменяются на медианные значения.
    """
    def __init__(self):
        self.median_width = None
        self.median_height = None
        
    def fit(self, X, y=None):
        # На этапе обучения вычисляем медианы, чтобы заполнить ими будущие пропуски
        widths = []
        heights = []
        
        for val in X['device_screen_resolution'].dropna():
            try:
                parts = str(val).lower().split('x')
                if len(parts) == 2:
                    widths.append(int(parts[0]))
                    heights.append(int(parts[1]))
            except ValueError:
                continue
                
        # Запоминаем медианы (защита от утечки данных)
        self.median_width = np.median(widths) if widths else 360
        self.median_height = np.median(heights) if heights else 640
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        widths = []
        heights = []
        
        for val in X_out['device_screen_resolution']:
            try:
                # Очищаем строку и бьем по знаку 'x'
                parts = str(val).lower().replace(' ', '').split('x')
                if len(parts) == 2:
                    widths.append(int(parts[0]))
                    heights.append(int(parts[1]))
                else:
                    widths.append(self.median_width)
                    heights.append(self.median_height)
            except ValueError:
                # Если в строке мусор или 'other', ставим медиану
                widths.append(self.median_width)
                heights.append(self.median_height)
                
        # Создаем новые числовые фичи
        X_out['screen_width'] = widths
        X_out['screen_height'] = heights
        # X_out['screen_area'] = X_out['screen_width'] * X_out['screen_height']
        
        # Удаляем старую текстовую колонку с 5000+ категориями
        return X_out.drop(columns=['device_screen_resolution'])
    
class CityAgreggatorTransformer(BaseEstimator, TransformerMixin):
    """
        Класс для предобработки категориального признака geo_city 
        Оставляет топ 30 самых частых городов, остальные меняет на other
    """
    def __init__(self, top_n=30):
        self.top_n = top_n
        self.top_cities = None
        
    def fit(self, X, y=None):
        # На этапе обучения находим ТОП-N самых частых городов
        self.top_cities = X['geo_city'].value_counts().head(self.top_n).index.tolist()
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # Если города нет в ТОП-N, заменяем его на 'other'
        is_not_top = ~X_out['geo_city'].isin(self.top_cities)
        X_out.loc[is_not_top, 'geo_city'] = 'other'
        return X_out
    
class WebTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
        Класс, для feature engineering временных меток. 
        Добавляет:
            visit_hour - час начала сессии
            is_night - признак ночного времени
            day_of_week - день недели
            is_weekend - признак выходных
            day_of_month - день месяца
            days_from_start - день с начала запуска исследования
        Исходные visit_date, visit_time удаляются
    """
    def __init__(self):
        self.min_date = None
        
    def fit(self, X, y=None):
        self.min_date = pd.to_datetime(X['visit_date']).min()
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X_out = X.copy()
        dates = pd.to_datetime(X_out['visit_date'])
        
        hours = X_out['visit_time'].astype(str).str.split(':').str[0].astype(int)
        X_out['visit_hour'] = hours
        X_out['is_night'] = ((hours >= 0) & (hours < 6)).astype(int)
        
        X_out['day_of_week'] = dates.dt.dayofweek.astype(int)
        
        X_out['is_weekend'] = dates.dt.dayofweek.isin([5, 6]).astype(int)
        X_out['day_of_month'] = dates.dt.day.astype(int)
        X_out['days_from_start'] = (dates - self.min_date).dt.days
        
        return X_out.drop(columns=['visit_date', 'visit_time'])
    
class BrandBasedOSImputer(BaseEstimator, TransformerMixin):
    """
        Класс, заполняющий пропуски в поле device_os, опираясь на данные из device_brand
    """
    def __init__(self):
        self.brand_to_os_map = {}
        self.default_os = 'Android'
        
    def fit(self, X, y=None):
        valid_data = X[['device_brand', 'device_os']].dropna()
        
        if not valid_data.empty:
            # Находим моду для каждого бренда
            counts = valid_data.value_counts().reset_index(name='count')
            top_per_brand = counts.drop_duplicates(subset=['device_brand'], keep='first')
            self.brand_to_os_map = pd.Series(
                top_per_brand['device_os'].values, 
                index=top_per_brand['device_brand']
            ).to_dict()
        self.is_fitted_ = True    
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        is_os_missing = X_out['device_os'].isna()
        
        if is_os_missing.any():
            imputed_os = X_out.loc[is_os_missing, 'device_brand'].map(self.brand_to_os_map).fillna(self.default_os)
            
            # Заполняем пропуски в исходной колонке
            X_out.loc[is_os_missing, 'device_os'] = imputed_os
            
        return X_out
    
class FillerScalerDropper(BaseEstimator, TransformerMixin):
    """
        Класс 
            заполняет оставшиеся пропуски в данных, 
            кодирует категориальные признаки, 
            стандартизирует данные.
        Удаляет из возвращаемых данных session_id и client_id
    """
    def __init__(self):

        # Инструменты для стандартизации и кодирования
        self.scaler = StandardScaler()
        self.target_encoder = TargetEncoder(random_state=RANDOM_STATE, cv=5)
        
        # Числовые признаки
        self.numeric_cols = [
            'visit_number', 'visit_hour', 'day_of_week', 
            'day_of_month', 'days_from_start',
            'screen_width', 'screen_height'#, 'screen_area'
        ]

        # категориальные признаки
        self.categorical_cols = [
            'device_category', 'device_os', 'utm_source', 
            'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
            'device_brand', 'device_model', 'device_browser', 
            'geo_country', 'geo_city'
        ]

        self.bool_cols = ['is_night', 'is_weekend']

    def fit(self, X, y=None):

        # Для обучения заменяем пропуски в числовых столбцах на 0
        X_num = X[self.numeric_cols].fillna(0)
        
        # А в категориальных на 'other'
        X_cat = X[self.categorical_cols].fillna('other').astype(str)

        # Обучаем 
        self.scaler.fit(X_num)
        self.target_encoder.fit(X_cat, y)
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X_out = X.copy()

        # заполняем пропуски уже на реальном датасете        
        X_num = X_out[self.numeric_cols].fillna(0)
        X_cat = X_out[self.categorical_cols].fillna('other').astype(str)

        # и трансформируем
        X_num_scaled = self.scaler.transform(X_num)
        X_cat_encoded = self.target_encoder.transform(X_cat)

        # кладем все данные в DataFrame-ы
        df_num = pd.DataFrame(X_num_scaled, columns=self.numeric_cols, index=X_out.index)
        df_cat = pd.DataFrame(X_cat_encoded, columns=self.categorical_cols, index=X_out.index)
        df_bin = X_out[self.bool_cols].fillna(0).astype(int)

        # Собираем итоговый DataFrame        
        final_df = pd.concat([df_num, df_cat, df_bin], axis=1)
        
        return final_df
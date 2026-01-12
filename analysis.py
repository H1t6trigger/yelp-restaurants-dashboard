import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def load_data():
    # Загружаем оба датафрейма
    biz = pd.read_csv("data/biz_clean.csv")
    reviews = pd.read_csv("data/reviews_clean.csv")
    return biz, reviews

def prepare_category_features(df, top_n=10):
    """
    Подготавливает One-Hot Encoded признаки категорий.

    Args:
        df (pd.DataFrame): Датафрейм с колонкой 'categories'.
        top_n (int): Количество топ-категорий для кодирования.

    Returns:
        pd.DataFrame: DataFrame с One-Hot Encoded признаками.
    """
    # 1. Разбиваем строки категорий и подсчитываем
    all_categories = []
    indices = [] # Сохраняем индекс оригинальной строки для каждой категории
    for idx, cat_str in df[['categories']].dropna().iterrows():
        categories_list = [cat.strip() for cat in cat_str['categories'].split(',')]
        all_categories.extend(categories_list)
        indices.extend([idx] * len(categories_list))

    cat_series = pd.Series(all_categories, index=indices, name='category')

    # 2. Получаем топ-N категорий
    top_cats = cat_series.value_counts().head(top_n).index.tolist()

    # 3. Создаем бинарный датафрейм для топ-категорий
    category_binary = pd.DataFrame(index=df.index)
    for cat in top_cats:
        # Проверяем, содержится ли категория в строке для каждой строки датафрейма
        category_binary[f'is_{cat}'] = df['categories'].fillna('').str.contains(cat, regex=False)

    # Преобразуем boolean в int (False/True -> 0/1)
    category_binary = category_binary.astype(int)

    return category_binary


#Кластеризация ресторанов
def cluster_restaurants(biz, n_clusters=3):
    X = biz[["stars", "review_count", "latitude", "longitude"]].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)

    biz_clustered = biz.loc[X.index].copy()
    biz_clustered["cluster"] = labels

    sil = silhouette_score(X_scaled, labels)

    return biz_clustered, sil

#Регрессия рейтинга
def regression_model(biz, use_categories=False): # <-- Добавлен параметр
    target_col = 'stars'
    feature_cols = ['review_count']

    # Подготовка признаков
    X_raw = biz[feature_cols].copy()

    if use_categories:
        # Подготовить признаки категорий
        cat_features = prepare_category_features(biz, top_n=10) # Используем топ-10
        # Объединяем с основными признаками
        X_raw = pd.concat([X_raw, cat_features], axis=1)

    # Убедимся, что нет пропусков в признаках и целевой переменной
    combined_data = X_raw.join(biz[target_col], how='inner').dropna()
    X = combined_data[X_raw.columns]
    y = combined_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    return model, r2, X_test, y_test, preds, X.columns.tolist() # <-- Возвращаем имена признаков для информации

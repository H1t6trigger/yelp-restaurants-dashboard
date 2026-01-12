import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

from analysis import load_data, cluster_restaurants, regression_model

# --- Настройка страницы ---
st.set_page_config(page_title="Yelp Restaurants Dashboard", layout="wide")

# --- Загрузка данных ---
biz, reviews = load_data()

# --- Навигация ---
page = st.sidebar.radio(
    "Навигация",
    ["Raw Data Visualization", "Analysis Results", "Column Translations"]
)

# --- Страница 1: Raw Data Visualization ---
if page == "Raw Data Visualization":
    st.title("Обзор данных ресторанов Yelp")

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Количество ресторанов", len(biz))
    col2.metric("Средний рейтинг ресторана", round(biz["stars"].mean(), 2))
    col3.metric("Среднее число отзывов", int(biz["review_count"].mean()))

    # --- НОВАЯ ВИЗУАЛИЗАЦИЯ: Распределение оценок отзывов ---
    st.subheader("Распределение оценок отзывов")
    if reviews is not None and not reviews.empty:
        # Подсчитываем количество отзывов для каждой оценки
        stars_dist = reviews['stars'].value_counts().sort_index()

        # Создаём DataFrame для Plotly
        dist_df = pd.DataFrame({'Оценка': stars_dist.index, 'Количество': stars_dist.values})

        # Строим столбчатую диаграмму
        fig_stars_dist = px.bar(dist_df, x='Оценка', y='Количество',
                                title="Распределение оценок отзывов",
                                labels={'Оценка': 'Оценка', 'Количество': 'Количество отзывов'})
        st.plotly_chart(fig_stars_dist, width='stretch')
    else:
        st.warning("Данные об отзывах не загружены.")


    # --- НОВАЯ ВИЗУАЛИЗАЦИЯ: Топ-10 категорий ---
    st.subheader("Топ-10 категорий ресторанов")
    if 'categories' in biz.columns:
        # Преобразование строковых категорий в список
        all_categories = []
        for cat_str in biz['categories'].dropna():
            all_categories.extend([cat.strip() for cat in cat_str.split(',')])

        # Исключим слово "Restaurants", если она была в процессе подготовки данных
        # all_categories = [c for c in all_categories if c != 'Restaurants'] # <- Можно добавить, если нужно

        cat_counts = pd.Series(all_categories).value_counts().head(10)

        # Создаём DataFrame для Plotly
        cat_df = pd.DataFrame({'Категория': cat_counts.index, 'Количество': cat_counts.values})

        # Строим горизонтальную столбчатую диаграмму
        fig_cat = px.bar(cat_df, x='Количество', y='Категория', orientation='h',
                         title="Топ-10 категорий ресторанов",
                         labels={'Количество': 'Количество', 'Категория': 'Категория'})
        st.plotly_chart(fig_cat, width='stretch')
    else:
        st.warning("Столбец 'categories' не найден в данных.")


    # Фильтры
    st.subheader("Фильтрация данных")
    col1, col2 = st.columns(2)

    with col1:
        min_rating = st.slider("Минимальный рейтинг ресторана", 1.0, 5.0, 3.0)

    with col2:
        min_reviews, max_reviews = st.slider(
            "Количество отзывов у ресторана",
            int(biz["review_count"].min()),
            int(biz["review_count"].max()),
            (int(biz["review_count"].quantile(0.1)), int(biz["review_count"].quantile(0.9)))
        )

    # Применение фильтров к датафрейму ресторанов
    filtered_biz = biz[
        (biz["stars"] >= min_rating) &
        (biz["review_count"] >= min_reviews) &
        (biz["review_count"] <= max_reviews)
    ]

    st.dataframe(filtered_biz.head(100))

    # Гистограмма средних рейтингов ресторанов
    st.subheader("Распределение средних рейтингов ресторанов")
    st.write("График показывает, как часто встречаются различные средние рейтинги среди ресторанов.")
    fig = px.histogram(filtered_biz, x="stars", nbins=10)
    st.plotly_chart(fig, width='stretch')

    # Географическое распределение
    st.subheader("Географическое распределение ресторанов")
    st.write("Точки на карте отображают местоположение ресторанов с цветовой кодировкой по среднему рейтингу.")
    fig_map = px.scatter_map(
        filtered_biz,
        lat="latitude",
        lon="longitude",
        color="stars",
        zoom=3,
        height=500,
        map_style="open-street-map"
    )
    st.plotly_chart(fig_map, width='stretch')

# --- Страница 2: Analysis Results ---
elif page == "Analysis Results":
    st.title("Результаты анализа")

    # Проверка, есть ли данные
    if biz is None or biz.empty:
        st.warning("Данные о ресторанах не загружены.")
        st.stop()

    # Кластеризация
    n_clusters = st.slider("Количество кластеров", 2, 6, 3)
    try:
        clustered, sil = cluster_restaurants(biz, n_clusters)
        st.metric("Silhouette Score", round(sil, 3))

        st.subheader("Кластеризация ресторанов")
        st.write("График показывает, как рестораны были разделены на кластеры на основе количества отзывов и среднего рейтинга.")
        fig_cluster = px.scatter(
            clustered,
            x="review_count",
            y="stars",
            color="cluster",
            title="Кластеры ресторанов"
        )
        st.plotly_chart(fig_cluster, width='stretch')
    except Exception as e:
        st.error(f"Ошибка при кластеризации: {e}")

    # Регрессия
    st.subheader("Регрессия среднего рейтинга ресторана")
    # --- Добавляем чекбокс для включения категорий ---
    use_categories_for_regression = st.checkbox("Использовать топ-10 категорий для улучшения модели", value=False)
    # --------------------------

    try:
        # --- Передаём параметр в функцию ---
        model, r2, X_test, y_test, preds, feature_names = regression_model(biz, use_categories=use_categories_for_regression)
        # --------------------------
        st.metric("R² модели", round(r2, 3))

        st.write("Сравнение реальных значений среднего рейтинга ресторана с предсказанными моделью.")
        reg_df = pd.DataFrame({
            "Реальные значения": y_test.values, # <-- .values для корректной обработки Series
            "Предсказания": preds
        })
        fig_reg = px.scatter(
            reg_df,
            x="Реальные значения",
            y="Предсказания",
            title="Реальные vs Предсказанные рейтинги ресторанов"
        )
        # Добавим диагональную линию для лучшего восприятия
        # Используем минимальные и максимальные значения из реальных данных
        min_val = reg_df["Реальные значения"].min()
        max_val = reg_df["Реальные значения"].max()
        fig_reg.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                          line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_reg, width='stretch')
    except Exception as e:
        st.error(f"Ошибка при регрессии: {e}")

# --- Страница 3: Column Translations ---
elif page == "Column Translations":
    st.title("Переводы названий колонок")

    # Словарь переводов
    column_translations = {
        "business_id": "ID ресторана",
        "name": "Название",
        "city": "Город",
        "state": "Штат",
        "stars": "Средний рейтинг",
        "review_count": "Количество отзывов",
        "categories": "Категории",
        "latitude": "Широта",
        "longitude": "Долгота",
        "review_id": "ID отзыва",
        "user_id": "ID пользователя",
        "text": "Текст отзыва",
        "date": "Дата отзыва",
        "text_len": "Длина текста",
        "cluster": "Кластер"
    }

    st.subheader("Словарь переводов колонок")
    st.write("Таблица содержит оригинальные названия колонок и их перевод на русский язык.")

    translation_df = pd.DataFrame(
        list(column_translations.items()),
        columns=["Column Name", "Translation"]
    )

    st.dataframe(translation_df)

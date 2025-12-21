import streamlit as st
import pandas as pd
import plotly.express as px

from analysis import load_data, cluster_restaurants, regression_model

# --- Настройка страницы ---
st.set_page_config(page_title="Yelp Restaurants Dashboard", layout="wide")

# --- Загрузка данных ---
biz = load_data()

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
    col2.metric("Средний рейтинг", round(biz["stars"].mean(), 2))
    col3.metric("Среднее число отзывов", int(biz["review_count"].mean()))

    # Таблица с фильтром
    st.subheader("Пример данных")
    min_rating = st.slider("Минимальный рейтинг", 1.0, 5.0, 3.0)
    filtered = biz[biz["stars"] >= min_rating]
    st.dataframe(filtered.head(100))

    # Гистограмма рейтингов
    fig = px.histogram(filtered, x="stars", nbins=10)
    st.plotly_chart(fig, width='stretch')

    # Географическое распределение
    fig_map = px.scatter_map(
        filtered,
        lat="latitude",
        lon="longitude",
        color="stars",
        zoom=3,
        height=500
    )
    st.plotly_chart(fig_map, width='stretch')

# --- Страница 2: Analysis Results ---
elif page == "Analysis Results":
    st.title("Результаты анализа")

    # Кластеризация
    n_clusters = st.slider("Количество кластеров", 2, 6, 3)
    clustered, sil = cluster_restaurants(biz, n_clusters)
    st.metric("Silhouette Score", round(sil, 3))
    fig_cluster = px.scatter(
        clustered,
        x="review_count",
        y="stars",
        color="cluster",
        title="Кластеры ресторанов"
    )
    st.plotly_chart(fig_cluster, width='stretch')

    # Регрессия
    model, r2, X_test, y_test, preds = regression_model(biz)
    st.metric("R² модели", round(r2, 3))
    reg_df = pd.DataFrame({
        "Реальные значения": y_test,
        "Предсказания": preds
    })
    fig_reg = px.scatter(
        reg_df,
        x="Реальные значения",
        y="Предсказания",
        title="Реальные vs Предсказанные рейтинги"
    )
    st.plotly_chart(fig_reg, width='stretch')

    st.markdown("""
    **Выводы:**
    - Количество отзывов оказывает умеренное влияние на рейтинг ресторана.
    - Кластеризация позволила выделить группы ресторанов с различными уровнями популярности.
    - Интерактивные фильтры позволяют исследовать данные в разрезе различных сценариев.
    """)

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

    translation_df = pd.DataFrame(
        list(column_translations.items()),
        columns=["Column Name", "Translation"]
    )

    st.dataframe(translation_df)

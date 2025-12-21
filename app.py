import streamlit as st
import pandas as pd
import plotly.express as px

from analysis import load_data, cluster_restaurants, regression_model


st.set_page_config(page_title="Yelp Restaurants Dashboard", layout="wide")

biz = load_data()

#Навигация 
page = st.sidebar.radio(
    "Навигация",
    ["Raw Data Visualization", "Analysis Results"]
)

#Страница 1
if page == "Raw Data Visualization":
    st.title("Обзор данных ресторанов Yelp")


#KPI
col1, col2, col3 = st.columns(3)
col1.metric("Количество ресторанов", len(biz))
col2.metric("Средний рейтинг", round(biz["stars"].mean(), 2))
col3.metric("Среднее число отзывов", int(biz["review_count"].mean()))


#Таблица
st.subheader("Пример данных")
st.dataframe(biz.head(100))


#Фильтры
min_rating = st.slider("Минимальный рейтинг", 1.0, 5.0, 3.0)
filtered = biz[biz["stars"] >= min_rating]


#Гистограмма рейтингов
fig = px.histogram(filtered, x="stars", nbins=10)
st.plotly_chart(fig, use_container_width=True)


#География
fig_map = px.scatter_mapbox(
    filtered,
    lat="latitude",
    lon="longitude",
    color="stars",
    zoom=3,
    height=500
)
fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map, use_container_width=True)


#СТРАНИЦА 2 — Analysis Results
if page == "Analysis Results":
    st.title("Результаты анализа")


#Кластеризация
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
st.plotly_chart(fig_cluster, use_container_width=True)


#Регрессия
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
st.plotly_chart(fig_reg, use_container_width=True)


#Выводы
st.markdown("""
**Выводы:**
- Количество отзывов оказывает умеренное влияние на рейтинг ресторана.
- Кластеризация позволила выделить группы ресторанов с различными уровнями популярности.
- Интерактивные фильтры позволяют исследовать данные в разрезе различных сценариев.
""")


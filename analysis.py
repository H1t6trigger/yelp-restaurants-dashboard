import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def load_data():
    biz = pd.read_csv("data/biz_clean.csv")
    return biz

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
def regression_model(biz):
    X = biz[["review_count"]]
    y = biz["stars"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    return model, r2, X_test, y_test, preds


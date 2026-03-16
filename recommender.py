"""
🎬 Movie Recommendation System
Алгоритм: SVD++ (улучшенный Collaborative Filtering)
Датасет: MovieLens 100K (загружается автоматически)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import urllib.request
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ─────────────────────────────────────────────

class DataLoader:
    """Загрузка датасета MovieLens 100K"""

    DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    DATA_DIR = "data"

    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)

    def download(self):
        zip_path = os.path.join(self.DATA_DIR, "ml-100k.zip")
        extract_path = os.path.join(self.DATA_DIR, "ml-100k")

        if not os.path.exists(extract_path):
            print("📥 Загрузка датасета MovieLens 100K...")
            urllib.request.urlretrieve(self.DATASET_URL, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(self.DATA_DIR)
            os.remove(zip_path)
            print("✅ Датасет загружен!\n")
        else:
            print("✅ Датасет уже загружен.\n")
        return extract_path

    def load(self):
        path = self.download()

        # Оценки
        ratings = pd.read_csv(
            os.path.join(path, "u.data"),
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )

        # Фильмы
        movies = pd.read_csv(
            os.path.join(path, "u.item"),
            sep='|', encoding='latin-1',
            names=['movie_id', 'title', 'release_date', 'video_release_date',
                   'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                   'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                   'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        )[['movie_id', 'title', 'Action', 'Adventure', 'Animation',
           'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
           'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
           'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]

        print(f"📊 Статистика датасета:")
        print(f"   Пользователей : {ratings['user_id'].nunique()}")
        print(f"   Фильмов       : {ratings['movie_id'].nunique()}")
        print(f"   Оценок        : {len(ratings)}")
        print(f"   Диапазон оценок: {ratings['rating'].min()} – {ratings['rating'].max()}\n")

        return ratings, movies


# ─────────────────────────────────────────────
#  2. SVD++ COLLABORATIVE FILTERING
# ─────────────────────────────────────────────

class SVDPlusPlusRecommender:
    """
    SVD++ — расширенный матричный метод.
    Учитывает:
      • явные оценки пользователей
      • неявные сигналы (факт просмотра)
      • смещения пользователей и фильмов
    """

    def __init__(self, n_factors=50, n_iterations=20, learning_rate=0.005,
                 regularization=0.02):
        self.n_factors = n_factors
        self.n_iter = n_iterations
        self.lr = learning_rate
        self.reg = regularization

        # Параметры модели (инициализируются при обучении)
        self.global_mean = 0
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None
        self.implicit_factors = None  # ← SVD++ дополнение

        self.user_index = {}
        self.item_index = {}
        self.user_items = {}        # неявные сигналы

    def _build_index(self, ratings):
        users = ratings['user_id'].unique()
        items = ratings['movie_id'].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {m: i for i, m in enumerate(items)}
        self.n_users = len(users)
        self.n_items = len(items)

    def _build_implicit(self, ratings):
        """Строим множество просмотренных фильмов для каждого пользователя"""
        for _, row in ratings.iterrows():
            uid = self.user_index[row['user_id']]
            iid = self.item_index[row['movie_id']]
            self.user_items.setdefault(uid, set()).add(iid)

    def fit(self, ratings):
        print("🧠 Обучение SVD++ модели...")
        self._build_index(ratings)
        self._build_implicit(ratings)

        self.global_mean = ratings['rating'].mean()

        # Инициализация параметров
        scale = 0.1
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.user_factors = np.random.normal(0, scale, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, scale, (self.n_items, self.n_factors))
        self.implicit_factors = np.random.normal(0, scale, (self.n_items, self.n_factors))

        # SGD обучение
        data = ratings[['user_id', 'movie_id', 'rating']].values
        for iteration in range(self.n_iter):
            np.random.shuffle(data)
            total_loss = 0

            for user_id, movie_id, rating in data:
                if user_id not in self.user_index or movie_id not in self.item_index:
                    continue

                u = self.user_index[user_id]
                i = self.item_index[movie_id]

                # SVD++: добавляем неявные факторы
                implicit_items = list(self.user_items.get(u, set()))
                if implicit_items:
                    norm_factor = 1.0 / np.sqrt(len(implicit_items))
                    implicit_sum = norm_factor * self.implicit_factors[implicit_items].sum(axis=0)
                else:
                    implicit_sum = np.zeros(self.n_factors)

                # Предсказание
                pred = (self.global_mean
                        + self.user_bias[u]
                        + self.item_bias[i]
                        + np.dot(self.user_factors[u] + implicit_sum,
                                 self.item_factors[i]))
                pred = np.clip(pred, 1, 5)
                error = rating - pred
                total_loss += error ** 2

                # Обновление параметров (градиентный спуск)
                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])

                uf_update = error * self.item_factors[i] - self.reg * self.user_factors[u]
                if_update = error * (self.user_factors[u] + implicit_sum) - self.reg * self.item_factors[i]
                self.user_factors[u] += self.lr * uf_update
                self.item_factors[i] += self.lr * if_update

                # Обновление неявных факторов
                if implicit_items:
                    for j in implicit_items:
                        impl_upd = (norm_factor * error * self.item_factors[i]
                                    - self.reg * self.implicit_factors[j])
                        self.implicit_factors[j] += self.lr * impl_upd

            rmse = np.sqrt(total_loss / len(data))
            if (iteration + 1) % 5 == 0:
                print(f"   Итерация {iteration+1:2d}/{self.n_iter}  |  RMSE: {rmse:.4f}")

        print("✅ Модель обучена!\n")
        return self

    def predict(self, user_id, movie_id):
        """Предсказание оценки для пары пользователь-фильм"""
        if user_id not in self.user_index or movie_id not in self.item_index:
            return self.global_mean

        u = self.user_index[user_id]
        i = self.item_index[movie_id]

        implicit_items = list(self.user_items.get(u, set()))
        if implicit_items:
            norm_factor = 1.0 / np.sqrt(len(implicit_items))
            implicit_sum = norm_factor * self.implicit_factors[implicit_items].sum(axis=0)
        else:
            implicit_sum = np.zeros(self.n_factors)

        pred = (self.global_mean
                + self.user_bias[u]
                + self.item_bias[i]
                + np.dot(self.user_factors[u] + implicit_sum, self.item_factors[i]))
        return float(np.clip(pred, 1, 5))

    def recommend(self, user_id, movies_df, n=10, exclude_seen=True):
        """Топ-N рекомендаций для пользователя"""
        if user_id not in self.user_index:
            print(f"⚠️  Пользователь {user_id} не найден.")
            return pd.DataFrame()

        u = self.user_index[user_id]
        seen = self.user_items.get(u, set())

        scores = []
        for movie_id, i in self.item_index.items():
            if exclude_seen and i in seen:
                continue
            score = self.predict(user_id, movie_id)
            scores.append((movie_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:n]

        result = pd.DataFrame(top, columns=['movie_id', 'predicted_rating'])
        result = result.merge(movies_df[['movie_id', 'title']], on='movie_id')
        result['predicted_rating'] = result['predicted_rating'].round(2)
        return result[['title', 'predicted_rating']].reset_index(drop=True)


# ─────────────────────────────────────────────
#  3. HYBRID RECOMMENDER (SVD++ + Content)
# ─────────────────────────────────────────────

class HybridRecommender:
    """
    Гибридная система: SVD++ + Content-Based Filtering
    Объединяет предсказания из двух моделей с весами.
    """

    def __init__(self, cf_weight=0.7, cb_weight=0.3):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.svd_model = SVDPlusPlusRecommender()
        self.item_similarity = None
        self.movies_df = None

    def _build_content_similarity(self, movies):
        """Косинусное сходство на основе жанров"""
        genres = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
        genre_matrix = movies[genres].fillna(0).values.astype(float)
        genre_matrix = normalize(genre_matrix)
        self.item_similarity = cosine_similarity(genre_matrix)
        self.movies_df = movies.reset_index(drop=True)

    def fit(self, ratings, movies):
        print("🔀 Обучение гибридной модели (SVD++ + Content)...")
        self.svd_model.fit(ratings)
        print("🎭 Построение матрицы жанрового сходства...")
        self._build_content_similarity(movies)
        print("✅ Гибридная модель готова!\n")
        return self

    def recommend(self, user_id, n=10):
        """Гибридные рекомендации"""
        cf_recs = self.svd_model.recommend(user_id, self.movies_df, n=n * 3)
        if cf_recs.empty:
            return cf_recs

        # Нормализуем CF оценки в [0,1]
        cf_recs['cf_score'] = (cf_recs['predicted_rating'] - 1) / 4

        # Контентные оценки: берём среднее сходство с просмотренными фильмами
        u = self.svd_model.user_index.get(user_id)
        seen_indices = list(self.svd_model.user_items.get(u, set()))

        cb_scores = {}
        for _, row in cf_recs.iterrows():
            mid = self.movies_df[self.movies_df['title'] == row['title']]['movie_id']
            if mid.empty:
                cb_scores[row['title']] = 0
                continue
            mid = mid.values[0]
            if mid not in self.svd_model.item_index:
                cb_scores[row['title']] = 0
                continue
            item_idx = self.movies_df[self.movies_df['movie_id'] == mid].index
            if len(item_idx) == 0 or not seen_indices:
                cb_scores[row['title']] = 0
                continue
            sim = self.item_similarity[item_idx[0], seen_indices].mean()
            cb_scores[row['title']] = sim

        cf_recs['cb_score'] = cf_recs['title'].map(cb_scores).fillna(0)
        cf_recs['hybrid_score'] = (self.cf_weight * cf_recs['cf_score']
                                   + self.cb_weight * cf_recs['cb_score'])
        cf_recs = cf_recs.sort_values('hybrid_score', ascending=False).head(n)
        cf_recs['final_rating'] = (cf_recs['hybrid_score'] * 4 + 1).round(2)

        return cf_recs[['title', 'predicted_rating', 'final_rating']].reset_index(drop=True)


# ─────────────────────────────────────────────
#  4. ОЦЕНКА КАЧЕСТВА
# ─────────────────────────────────────────────

def evaluate_model(model, test_ratings):
    """Вычисляем RMSE и MAE на тестовой выборке"""
    predictions, actuals = [], []
    for _, row in test_ratings.iterrows():
        pred = model.predict(row['user_id'], row['movie_id'])
        predictions.append(pred)
        actuals.append(row['rating'])

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    return rmse, mae


# ─────────────────────────────────────────────
#  5. ТОЧКА ВХОДА
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🎬 MOVIE RECOMMENDATION SYSTEM — SVD++  ")
    print("=" * 55 + "\n")

    # Загрузка данных
    loader = DataLoader()
    ratings, movies = loader.load()

    # Разбивка train/test
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=0.2, random_state=42
    )
    print(f"📂 Train: {len(train_ratings)} | Test: {len(test_ratings)}\n")

    # ── Вариант 1: чистый SVD++ ──────────────────
    print("━" * 40)
    print(" МОДЕЛЬ 1: SVD++ Collaborative Filtering")
    print("━" * 40)
    svd_model = SVDPlusPlusRecommender(
        n_factors=50,
        n_iterations=20,
        learning_rate=0.005,
        regularization=0.02
    )
    svd_model.fit(train_ratings)

    rmse, mae = evaluate_model(svd_model, test_ratings.head(2000))
    print(f"\n📈 Метрики качества SVD++:")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}\n")

    # Рекомендации для пользователя
    user_id = 1
    print(f"🎯 Топ-10 рекомендаций для пользователя #{user_id} (SVD++):")
    recs = svd_model.recommend(user_id, movies, n=10)
    print(recs.to_string(index=False))

    # ── Вариант 2: Гибридная модель ─────────────
    print("\n" + "━" * 40)
    print(" МОДЕЛЬ 2: Hybrid SVD++ + Content-Based")
    print("━" * 40 + "\n")
    hybrid = HybridRecommender(cf_weight=0.7, cb_weight=0.3)
    hybrid.fit(train_ratings, movies)

    print(f"🎯 Топ-10 гибридных рекомендаций для пользователя #{user_id}:")
    hybrid_recs = hybrid.recommend(user_id, n=10)
    print(hybrid_recs.to_string(index=False))
    print("\n  predicted_rating — оценка SVD++")
    print("  final_rating     — гибридная оценка (SVD++ + жанры)\n")

    # ── Схожие фильмы ────────────────────────────
    print("━" * 40)
    print(" ПОХОЖИЕ ФИЛЬМЫ (Content-Based)")
    print("━" * 40)
    target_movie = movies.iloc[0]['title']
    print(f"\nФильм: «{target_movie}»")
    target_idx = 0
    sims = hybrid.item_similarity[target_idx]
    sim_indices = sims.argsort()[::-1][1:6]
    similar = movies.iloc[sim_indices][['title']].copy()
    similar['similarity'] = sims[sim_indices].round(3)
    print(similar.to_string(index=False))

    print("\n✅ Готово! Система рекомендаций работает.")
    print("=" * 55)


if __name__ == "__main__":
    main()

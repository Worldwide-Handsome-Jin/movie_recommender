# 🎬 Movie Recommendation System — SVD++

## Стек
- Python 3.10+
- Pandas, NumPy, Scikit-learn
- Алгоритм: **SVD++** + Hybrid Content-Based

---

## 📁 Структура проекта

```
movie_recommender/
├── recommender.py     # Основной файл системы
├── requirements.txt   # Зависимости
├── README.md
└── data/              # Создаётся автоматически при запуске
    └── ml-100k/       # MovieLens 100K датасет
```

---

## 🚀 Запуск в PyCharm — пошагово

### 1. Открыть проект
`File → Open` → выбрать папку `movie_recommender`

### 2. Создать виртуальное окружение
`File → Settings → Project → Python Interpreter`
→ `Add Interpreter → Add Local Interpreter → Virtualenv Environment`
→ `OK`

### 3. Установить зависимости
Открыть терминал в PyCharm (`Alt+F12`):
```bash
pip install -r requirements.txt
```

### 4. Запустить
Открыть `recommender.py` → нажать `▶ Run` (Shift+F10)

---

## 🧠 Алгоритмы

### SVD++ (основной)
Матричная факторизация с неявными сигналами.  
Отличие от обычного SVD: учитывает **сам факт просмотра** фильма,  
даже без оценки — через вектор `implicit_factors`.

```
rating(u, i) = μ + b_u + b_i + (p_u + |N(u)|^(-½) Σ y_j) · q_i
```

| Параметр | Значение | Описание |
|---|---|---|
| `n_factors` | 50 | Размерность скрытого пространства |
| `n_iterations` | 20 | Число эпох SGD |
| `learning_rate` | 0.005 | Шаг градиентного спуска |
| `regularization` | 0.02 | L2-регуляризация |

### Hybrid (SVD++ + Content-Based)
Финальная оценка = `0.7 × CF_score + 0.3 × CB_score`

- **CF** (Collaborative Filtering): SVD++ предсказание
- **CB** (Content-Based): косинусное сходство жанровых векторов

---

## 📊 Ожидаемые результаты

```
RMSE : ~0.92   (чем ниже — тем лучше; 1.0 = отличный результат)
MAE  : ~0.73
```

---

## 💡 Как использовать модель

```python
from recommender import DataLoader, SVDPlusPlusRecommender, HybridRecommender

# Загрузка
loader = DataLoader()
ratings, movies = loader.load()

# SVD++
model = SVDPlusPlusRecommender(n_factors=50, n_iterations=20)
model.fit(ratings)

# Рекомендации для пользователя с ID=42
recs = model.recommend(user_id=42, movies_df=movies, n=10)
print(recs)

# Предсказать оценку конкретного фильма
score = model.predict(user_id=42, movie_id=100)

# Гибридная модель
hybrid = HybridRecommender(cf_weight=0.7, cb_weight=0.3)
hybrid.fit(ratings, movies)
hybrid_recs = hybrid.recommend(user_id=42, n=10)
```

---

## ❓ Частые вопросы

**Q: Долго обучается?**  
A: ~2–4 минуты на 100K оценках. Уменьшите `n_iterations=10` для ускорения.

**Q: Как улучшить качество?**  
A: Увеличьте `n_factors` до 100 и `n_iterations` до 30.

**Q: Cold Start (новый пользователь)?**  
A: Для новых пользователей система возвращает `global_mean`. Можно расширить логикой популярных фильмов.

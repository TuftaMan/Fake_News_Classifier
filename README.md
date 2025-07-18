📰 Fake News Classifier (Определение фейковых новостей)

Проект по классификации новостей как "FAKE" или "REAL" с помощью модели машинного обучения. Используется TfidfVectorizer и PassiveAggressiveClassifier.

🔍 Описание проекта
Цель проекта — определить, является ли новость достоверной или фейковой, на основе её текста. Это может быть полезно для медиа-компаний, исследователей, социальных платформ и пользователей, желающих фильтровать информацию.

📁 Структура проекта
bash
Копировать
Редактировать
fake-news-classifier/
├── data/
│   └── fake_news.csv           # Исходный датасет
├── models/
│   └── fake_news_model.pkl     # Сохранённая обученная модель
├── src/
│   ├── data_loader.py          # Загрузка и разбивка данных
│   ├── model.py                # Построение модели
│   └── visualization.py        # Метрики и визуализация
├── main.py                     # Основной скрипт обучения
├── predict.py                  # Интерактивный скрипт предсказания
└── README.md                   # Описание проекта
🧠 Используемые технологии
Python

Scikit-learn

Pandas

Matplotlib, Seaborn

Joblib

🚀 Как запустить
Клонируй репозиторий

bash
Копировать
Редактировать
git clone https://github.com/твой-username/fake-news-classifier.git
cd fake-news-classifier
Создай виртуальное окружение и установи зависимости

bash
Копировать
Редактировать
python -m venv .venv
source .venv/bin/activate     # или .venv\Scripts\activate на Windows
pip install -r requirements.txt
Добавь CSV-дataset в папку data/ под названием fake_news.csv

Запусти обучение модели

bash
Копировать
Редактировать
python main.py
Для предсказаний вручную:

bash
Копировать
Редактировать
python predict.py
🧪 Пример вывода
sql
Копировать
Редактировать
Введите текст новости: Scientists discover new planet near Earth.
Эта новость — REAL
📊 Результаты
Модель достигла точности около 95% на тестовой выборке.

Визуализация — матрица ошибок (confusion matrix), классификационный отчёт (precision, recall, f1-score) показывают высокую эффективность классификации.
# 📰 Fake News Detector

Проект по классификации новостей на **настоящие (REAL)** и **фейковые (FAKE)** с помощью методов машинного обучения.

# 📂 Структура проекта


``` project/ 
├── data/ # Датасет 
    fake_news.csv 
├── models/ # Сохранённая модель
├── src/ │ 
    ├── data_loader.py # Загрузка и разбивка данных 
    ├── model.py # Построение ML модели 
    └── visualization.py # Оценка модели и визуализация 
├── main.py # Обучение и сохранение модели
├── predict.py # Предсказание на новых данных 
└── README.md # Описание проекта 
```

## 📊 Используемые технологии

- `scikit-learn`
- `PassiveAggressiveClassifier`
- `TfidfVectorizer`
- `joblib`
- `matplotlib`, `seaborn` — визуализация

## 🚀 Как запустить
```bash
1. Установи зависимости:
   
pip install -r requirements.txt

2. Запусти обучение модели:

python main.py

3. Протестируй свою новость:
   
python predict.py
```

## 📈 Результаты

Классификационная модель достигает высокой точности на тестовой выборке, с визуализацией матрицы ошибок.

## 📦 Данные

Датасет fake_news.csv должен быть размещён в папке data/.
Он должен содержать два столбца: text (текст новости) и label (FAKE или REAL).

## 🤖 Пример использования

Введите текст новости: 
Coronavirus vaccine causes 5G infection

Эта новость — FAKE

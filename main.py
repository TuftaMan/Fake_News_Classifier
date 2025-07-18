from src.data_loader import load_data, split_data
from src.model import build_model
from src.visualization import evaluate_model, plot_confusion_matrix
import joblib

def main():
    # Загрузка и разбитие данных
    X, y = load_data("data/fake_news.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Построение модели
    model = build_model()

    # Обучение
    model.fit(X_train, y_train)

    # Сохранение модели

    joblib.dump(model, 'models/fake_news_model.pkl')

    # Предсказание
    y_pred = model.predict(X_test)

    # Оценка и визуализация
    evaluate_model(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)

if __name__ == '__main__':
    main()

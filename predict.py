import joblib

def main():
    model = joblib.load('models/fake_news_model.pkl')
    print("Модель загружена. Введите новость для проверки.")
    print("Нажмите Enter без текста — чтобы выйти.\n")

    while True:
        news = input("Введите текст новости: ").strip()
        if not news:
            print("👋 Выход из программы.")
            break
        prediction = model.predict([news])[0]
        print(f"Эта новость — {prediction.upper()}\n")

if __name__ == '__main__':
    main()
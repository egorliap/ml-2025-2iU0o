import numpy as np
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    CategoricalNB,
    ComplementNB
)
from sklearn.feature_extraction.text import CountVectorizer

# Цвета ANSI


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def separator(name):
    print(f"\n{Colors.HEADER}{'='*20} {name} {'='*20}{Colors.ENDC}")


def section(name):
    print(f"\n{Colors.CYAN}--- {name} ---{Colors.ENDC}")


def result(text):
    print(f"{Colors.GREEN}{Colors.BOLD}>>> {text}{Colors.ENDC}")

# 1. MultinomialNB - Пример из презентации (Спам-фильтр)


def example_multinomial_nb():
    separator("MultinomialNB (Пример из презентации)")

    # Расширенная обучающая выборка
    train_messages = [
        "Hi, how are you?",                      # Not Spam
        "Congratulations, you won a prize!",     # Spam
        "Buy the product now and get a discount!",  # Spam
        "Let's walk this evening",               # Not Spam
        "Hello friend, let's meet tomorrow",     # Not Spam
        "Win cash prizes immediately!",          # Spam
        "Urgent: your account needs verification",  # Spam
        "Hope you are doing well",               # Not Spam
        "Meeting at 5 PM in the office",         # Not Spam
        "Exclusive offer just for you"           # Spam
    ]
    train_labels = [
        "Not Spam", "Spam", "Spam", "Not Spam", "Not Spam",
        "Spam", "Spam", "Not Spam", "Not Spam", "Spam"
    ]

    # Тестовое сообщение
    test_message_str = "Hi, you won a discount and you can get the prize this evening"
    test_message = [test_message_str]

    # Преобразование текста в мешок слов (Bag of Words)
    # Используем CountVectorizer для подсчета частот слов
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_messages)
    X_test = vectorizer.transform(test_message)

    # Обучение модели
    # alpha=1.0 - сглаживание Лапласа (как указано в презентации)
    # MultinomialNB вычисляет вероятности P(слово|класс) на основе частот слов.
    # Для нового сообщения перемножает вероятности встретившихся слов для каждого класса.
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, train_labels)

    # Предсказание
    predicted = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    print(f"Количество обучающих сообщений: {len(train_messages)}")
    print(f"Обучающие сообщения (первые 5): {train_messages[:5]}...")
    print(f"Метки (первые 5): {train_labels[:5]}...")
    print(
        f"Тестовое сообщение: {Colors.WARNING}{test_message[0]}{Colors.ENDC}")
    result(f"Предсказанный класс: {predicted[0]}")
    print(f"Вероятности классов {clf.classes_}: {probs[0]}")

    # --- Подробный расчет (как в презентации) ---
    section("Подробный расчет MultinomialNB")
    feature_names = vectorizer.get_feature_names_out()

    # Получаем индексы слов из тестового сообщения
    test_indices = X_test.nonzero()[1]

    for i, class_label in enumerate(clf.classes_):
        print(f"\n{Colors.BLUE}Для класса '{class_label}':{Colors.ENDC}")

        # 1. Априорная вероятность P(C)
        # sklearn хранит log(P(C)) в class_log_prior_
        class_prior = np.exp(clf.class_log_prior_[i])
        print(f"  P({class_label}) = {class_prior:.4f}")

        # 2. Вероятности слов P(w|C)
        # sklearn хранит log(P(w|C)) в feature_log_prob_
        word_probs = []
        log_prob_sum = clf.class_log_prior_[i]

        print("  Слова сообщения:")
        for idx in test_indices:
            word = feature_names[idx]
            count = X_test[0, idx]
            # log(P(w|C))
            word_log_prob = clf.feature_log_prob_[i, idx]
            word_prob = np.exp(word_log_prob)

            # Учитываем, сколько раз слово встретилось
            log_prob_sum += word_log_prob * count
            word_probs.append(f"P('{word}'|{class_label})={word_prob:.4f}")
            print(f"    - '{word}': {word_prob:.4f} (count: {count})")

        # 3. Итоговая (ненормализованная) вероятность (в логарифмах, так как числа очень малые)
        # P(C|M) ~ P(C) * Product(P(w|C))
        # Log P(C|M) = Log P(C) + Sum(Log P(w|C))
        print(
            f"  Log P({class_label}|Message) = {clf.class_log_prior_[i]:.4f} (prior) + {log_prob_sum - clf.class_log_prior_[i]:.4f} (words) = {log_prob_sum:.4f}")
        print(
            f"  P({class_label}|Message) [unnormalized] ≈ {Colors.BOLD}{np.exp(log_prob_sum):.4e}{Colors.ENDC}")


# 2. GaussianNB - Непрерывные данные
def example_gaussian_nb():
    separator("GaussianNB (Непрерывные данные)")
    # Пример: Классификация людей по росту (см) и весу (кг)
    # Класс 0: Дети, Класс 1: Взрослые
    # Предполагаем нормальное распределение признаков внутри классов
    # Добавляем больше данных
    X_train = np.array([
        # Дети (рост 100-140, вес 20-35)
        [100, 20], [110, 22], [120, 25], [105, 21], [115, 24],
        [130, 30], [125, 28], [135, 32], [102, 19], [108, 23],
        # Взрослые (рост 160-190, вес 60-90)
        [170, 70], [175, 80], [180, 85], [165, 65], [160, 60],
        [185, 90], [172, 72], [178, 82], [168, 68], [182, 88]
    ])
    y_train = ['Child'] * 10 + ['Adult'] * 10

    X_test = np.array([[115, 23], [172, 75]])

    clf = GaussianNB()
    # GaussianNB строит нормальное распределение (колокол Гаусса) для каждого признака внутри каждого класса.
    # Он запоминает среднее значение (mu) и стандартное отклонение (sigma) для (Рост|Ребенок), (Вес|Ребенок) и т.д.
    # При предсказании вычисляет плотность вероятности для значений тестового примера.
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Признаки (Рост, Вес)")
    print(f"Тестовые данные:\n{X_test}")
    result(f"Предсказания: {predictions}")

    # --- Подробный расчет ---
    section("Подробный расчет GaussianNB (для первого примера [115, 23])")
    test_sample = X_test[0]  # [115, 23]

    for i, class_label in enumerate(clf.classes_):
        print(f"\n{Colors.BLUE}Для класса '{class_label}':{Colors.ENDC}")
        # Априорная вероятность
        class_prior = clf.class_prior_[i]
        print(f"  P({class_label}) = {class_prior:.4f}")

        # Параметры Гауссианы (mu, sigma^2)
        means = clf.theta_[i]
        vars = clf.var_[i]

        log_prob_sum = np.log(class_prior)

        print("  Признаки:")
        for j, feature_val in enumerate(test_sample):
            mu = means[j]
            sigma2 = vars[j]
            sigma = np.sqrt(sigma2)

            # Формула плотности вероятности Гаусса:
            # P(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2 / (2*sigma^2))
            prob_density = (1 / np.sqrt(2 * np.pi * sigma2)) * \
                np.exp(-((feature_val - mu)**2) / (2 * sigma2))

            print(
                f"    - Признак {j} (val={feature_val}): mu={mu:.2f}, sigma={sigma:.2f} => P(x|{class_label}) = {prob_density:.4e}")
            log_prob_sum += np.log(prob_density)

        print(f"  Log P({class_label}|Data) = {log_prob_sum:.4f}")
        print(
            f"  Score (unnormalized density product) ≈ {Colors.BOLD}{np.exp(log_prob_sum):.4e}{Colors.ENDC}")

# 3. BernoulliNB - Бинарные данные


def example_bernoulli_nb():
    separator("BernoulliNB (Бинарные данные)")
    # Пример: Классификация текстов или объектов по наличию/отсутствию признаков
    # 1 - признак есть, 0 - признака нет
    # Признаки: [слово "скидка", слово "бесплатно", слово "привет", слово "купить", слово "работа"]
    # Расширяем набор признаков и данных
    X_train = np.array([
        # Spam (акцент на "скидка", "бесплатно", "купить")
        [1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        # Not Spam (акцент на "привет", "работа")
        [0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],  # Смешанный, но контекст работы
        [0, 0, 1, 0, 0]
    ])
    y_train = ['Spam'] * 5 + ['Not Spam'] * 5

    # Тест: есть "бесплатно" и "купить" (2-й и 4-й признаки), нет остальных
    X_test = np.array([[0, 1, 0, 1, 0]])

    clf = BernoulliNB()
    # BernoulliNB работает с бинарными данными (0 или 1).
    # Он вычисляет вероятность P(признак=1|класс).
    # В отличие от MultinomialNB, он явно штрафует за ОТСУТСТВИЕ признаков, которые должны быть в классе (P(x=0|y)).
    clf.fit(X_train, y_train)

    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Признаки (скидка, бесплатно, привет, купить, работа)")
    print(f"Обучающие данные (первые 3 строки):\n{X_train[:3]}")
    result(f"Тест {X_test[0]} -> Класс: {clf.predict(X_test)[0]}")

    # --- Подробный расчет ---
    section("Подробный расчет BernoulliNB")
    test_sample = X_test[0]
    feature_names_list = ["скидка", "бесплатно", "привет", "купить", "работа"]

    for i, class_label in enumerate(clf.classes_):
        print(f"\n{Colors.BLUE}Для класса '{class_label}':{Colors.ENDC}")
        class_prior = np.exp(clf.class_log_prior_[i])
        print(f"  P({class_label}) = {class_prior:.4f}")

        log_prob_sum = clf.class_log_prior_[i]

        print("  Признаки:")
        for j, feature_val in enumerate(test_sample):
            # Вероятность того, что признак = 1
            prob_1 = np.exp(clf.feature_log_prob_[i, j])

            # Если в тесте 1, берем P(x=1|y). Если 0, берем P(x=0|y) = 1 - P(x=1|y)
            if feature_val == 1:
                p_val = prob_1
                desc = f"{feature_names_list[j]}=YES"
            else:
                p_val = 1 - prob_1
                desc = f"{feature_names_list[j]}=NO "

            print(
                f"    - {desc}: P(x={feature_val}|{class_label}) = {p_val:.4f}")
            log_prob_sum += np.log(p_val)

        print(f"  Log P({class_label}|Data) = {log_prob_sum:.4f}")
        print(
            f"  P({class_label}|Data) [unnormalized] ≈ {Colors.BOLD}{np.exp(log_prob_sum):.4e}{Colors.ENDC}")

# 4. CategoricalNB - Категориальные данные


def example_categorical_nb():
    separator("CategoricalNB (Категориальные данные)")
    # Пример: Предпочтения.
    # Признак 1: Погода (0=Солнечно, 1=Дождь, 2=Снег, 3=Облачно)
    # Признак 2: Время (0=Утро, 1=Вечер, 2=День)
    # Целевая переменная: Активность (0=Прогулка, 1=Кино, 2=Дом)

    # Расширяем данные
    X_train = np.array([
        # Солнечно/Облачно + Утро/День -> Прогулка
        [0, 0], [0, 2], [3, 0], [3, 2],
        [1, 0], [1, 1], [2, 1], [1, 2],  # Дождь/Снег + Вечер/День -> Кино
        # Снег/Дождь + Вечер или Солнце+Вечер -> Дом/Кино (смешанно)
        [2, 1], [1, 1], [2, 0], [0, 1]
    ])
    # Уточним метки
    y_train = [
        'Walk', 'Walk', 'Walk', 'Walk',
        'Cinema', 'Cinema', 'Cinema', 'Cinema',
        'Home', 'Home', 'Home', 'Walk'
    ]
    # Данные выше немного условны для примера

    # Тест: Дождь (1), Утро (0) -> ?
    X_test = np.array([[1, 0]])

    clf = CategoricalNB()
    # CategoricalNB предназначен для категориальных признаков, где порядок чисел не важен.
    # Он считает частоты каждой категории для каждого признака в каждом классе.
    # Например, P(Погода=Дождь|Кино).
    clf.fit(X_train, y_train)

    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Признаки (Погода, Время)")
    result(f"Тест {X_test[0]} -> Класс: {clf.predict(X_test)[0]}")

    # --- Подробный расчет ---
    section("Подробный расчет CategoricalNB")
    test_sample = X_test[0]
    feature_names_list = ["Погода", "Время"]

    for i, class_label in enumerate(clf.classes_):
        print(f"\n{Colors.BLUE}Для класса '{class_label}':{Colors.ENDC}")
        class_prior = np.exp(clf.class_log_prior_[i])
        print(f"  P({class_label}) = {class_prior:.4f}")

        log_prob_sum = clf.class_log_prior_[i]

        print("  Признаки:")
        for j, feature_val in enumerate(test_sample):
            # clf.feature_log_prob_ это список массивов, по массиву на каждый признак
            # feature_log_prob_[j] имеет форму (n_classes, n_categories_for_feature_j)

            log_prob_cat = clf.feature_log_prob_[j][i, feature_val]
            prob_cat = np.exp(log_prob_cat)

            print(
                f"    - {feature_names_list[j]}={feature_val}: P(x={feature_val}|{class_label}) = {prob_cat:.4f}")
            log_prob_sum += log_prob_cat

        print(f"  Log P({class_label}|Data) = {log_prob_sum:.4f}")
        print(
            f"  P({class_label}|Data) [unnormalized] ≈ {Colors.BOLD}{np.exp(log_prob_sum):.4e}{Colors.ENDC}")

# 5. ComplementNB - Несбалансированные данные


def example_complement_nb():
    separator("ComplementNB (Несбалансированные данные)")
    # Хорошо работает с несбалансированными выборками (например, много "Нормальных" и мало "Аномалий")

    # Увеличиваем дисбаланс и количество данных
    # Класс A: 50 примеров, Класс B: 5 примеров
    X_train = np.concatenate([
        np.tile([1, 0], (50, 1)),  # 50 примеров [1, 0] для класса A
        np.tile([0, 1], (5, 1))   # 5 примеров [0, 1] для класса B
    ])

    y_train = ['A'] * 50 + ['B'] * 5

    # Тест: признаки указывают скорее на B [0, 1]
    X_test = np.array([[0, 1]])

    cnb = ComplementNB()
    # ComplementNB - это модификация MultinomialNB.
    # Вместо P(x_i | y) он оценивает P(x_i | НЕ y) — то есть вероятность встретить признак во ВСЕХ ДРУГИХ классах.
    # Это помогает, когда один класс доминирует по количеству примеров, выравнивая веса.
    cnb.fit(X_train, y_train)

    print(
        f"Количество примеров A: {y_train.count('A')}, B: {y_train.count('B')}")
    print(f"Тест: {X_test[0]}")
    result(f"ComplementNB предсказание: {cnb.predict(X_test)[0]}")

    # --- Подробный расчет ---
    section("Подробный расчет ComplementNB")
    # ComplementNB использует веса (weights), а не простые вероятности.
    # Weight w_ci = log( theta_ci / sum(theta_ci) )
    # где theta_ci оценивается по дополнению (все классы КРОМЕ c)

    test_sample = X_test[0]

    # feature_log_prob_ в ComplementNB хранит веса w_ci (уже нормализованные особым образом в реализации sklearn)
    # Но для наглядности покажем "счет" (score) который является dot product

    for i, class_label in enumerate(cnb.classes_):
        print(f"\n{Colors.BLUE}Для класса '{class_label}':{Colors.ENDC}")
        # В ComplementNB нет class_log_prior в классическом смысле, если fit_prior=False (default=True но он не используется так явно)
        # Основное - это сумма весов признаков

        score = 0
        print("  Признаки (вклад в Score):")
        for j, count in enumerate(test_sample):
            if count > 0:
                weight = cnb.feature_log_prob_[i, j]
                contribution = weight * count
                print(
                    f"    - Признак {j} (count={count}): weight={weight:.4f} => contribution={contribution:.4f}")
                score += contribution

        print(
            f"  Итоговый Score для класса {class_label} = {Colors.BOLD}{score:.4f}{Colors.ENDC}")

    # Класс выбирается по минимальному score? Нет, в sklearn ComplementNB выбирает argmax.
    # Но стоит помнить что внутри ComplementNB формула специфическая.


if __name__ == "__main__":
    example_multinomial_nb()
    example_gaussian_nb()
    example_bernoulli_nb()
    example_categorical_nb()
    example_complement_nb()

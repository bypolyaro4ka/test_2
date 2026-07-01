# Модель: Лассо и гребневая регрессия (классический метод).
# Запуск: .venv/bin/python3 models/classic/model_lasso_ridge.py

# Импорты библиотек
import sys
import os

# Пути к скриптам моделей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from utils import (
    load_data, evaluate_model, save_results,
    plot_predictions, print_metrics,
)

print("=" * 60)
print("МОДЕЛЬ: Лассо и гребневая регрессия")
print("=" * 60)

# Загружаем обучающую и тестовую выборки из датасета
data = load_data()
X_train, X_test = data['X_train'], data['X_test'] # Обучающая и тестовая выборки
y_train, y_test = data['y_train'], data['y_test'] # Правильные ответы для обучения и для теста

# Лассо-регрессия (Lasso)
print("\n--- Лассо-регрессия (Lasso) ---")
lasso = Lasso(alpha=0.1, random_state=42) # alpha это сила штрафа, что бы у модели не было больших весов, а random_state фиксация результатов модели, что бы каждый раз у нас были одинаковые результаты модели
lasso.fit(X_train, y_train) # Даем модели образцы и 8 признаков бетона и так же даем реальные значения прочности бетона, модели подбирает 8 весов и старается минимизировать ошибку
y_pred_lasso = lasso.predict(X_test) # Тут уже предсказание модели из тестовых данных которые мы взяли из общего датасета, эти данные модель не видела.

metrics_lasso = evaluate_model(y_test, y_pred_lasso) # Сравниваем полученные предсказания с реальными данными из датасета. За сравнение отвечает metrics_lasso в utils.py
print_metrics('Лассо-регрессия', metrics_lasso) # Вывод всех метрик в консоль
# Сохраняем результаты для повторного быстрого запуска и для дашборда с отображением информации
save_results(
    'lasso', metrics_lasso, y_pred_lasso,
    extra={'display_name': 'Лассо-регрессия', 'type': 'classic'}
)
# Ну тут просто строим график который потом сохраняется в results/plots
plot_predictions(
    'Лассо-регрессия', y_test, y_pred_lasso,
    metrics_lasso, color='#AB63FA'
)

# Гребневая регрессия (Ridge)
print("\n--- Гребневая регрессия (Ridge) ---")
ridge_alpha = 10.0
ridge = Ridge(alpha=ridge_alpha, random_state=42)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

metrics_ridge = evaluate_model(y_test, y_pred_ridge)
print_metrics('Гребневая регрессия', metrics_ridge)
save_results(
    'ridge', metrics_ridge, y_pred_ridge,
    extra={
        'display_name': 'Гребневая регрессия',
        'type': 'classic',
        'alpha': ridge_alpha,
    }
)
plot_predictions(
    'Гребневая регрессия', y_test, y_pred_ridge,
    metrics_ridge, color='#FFA15A'
)

# Совмещённый график
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, name, y_pred, color in [
    (axes[0], 'Лассо-регрессия', y_pred_lasso, '#AB63FA'),
    (axes[1], 'Гребневая регрессия', y_pred_ridge, '#FFA15A'),
]:
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='black',
               linewidth=0.5, color=color)
    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
    r2 = evaluate_model(y_test, y_pred)['R2']
    ax.set_title('{}\nR² = {:.4f}'.format(name, r2), fontsize=13)
    ax.set_xlabel('Реальные (МПа)')
    ax.set_ylabel('Предсказанные (МПа)')
    ax.grid(True, alpha=0.3)

plt.suptitle(
    'Лассо и гребневая регрессия: сравнение',
    fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig(
    'results/plots/прогноз_лассо_гребневая_сравнение.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("\nСохранено: results/plots/прогноз_лассо_гребневая_сравнение.png")

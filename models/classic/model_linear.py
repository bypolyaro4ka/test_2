# Модель: Линейная регрессия (классический метод).
# Запуск: .venv/bin/python3 models/classic/model_linear.py

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.linear_model import LinearRegression
from utils import (
    load_data, evaluate_model, save_results,
    plot_predictions, print_metrics,
)

MODEL_NAME = 'linear_regression'
DISPLAY_NAME = 'Линейная регрессия'

print("=" * 60)
print("МОДЕЛЬ: {}".format(DISPLAY_NAME))
print("=" * 60)

# Загрузка данных
data = load_data()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Обучение
model = LinearRegression()
model.fit(X_train, y_train)
print("Модель обучена")

# Предсказание
y_pred = model.predict(X_test)

# Оценка
metrics = evaluate_model(y_test, y_pred)
print_metrics(DISPLAY_NAME, metrics)

# Сохранение
save_results(
    MODEL_NAME, metrics, y_pred,
    extra={'display_name': DISPLAY_NAME, 'type': 'classic'}
)
plot_predictions(DISPLAY_NAME, y_test, y_pred, metrics, color='#636EFA')

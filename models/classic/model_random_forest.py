# Модель: Random Forest Regressor (классический метод).
# Запуск: .venv/bin/python3 models/classic/model_random_forest.py

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from utils import (
    load_data, evaluate_model, save_results,
    plot_predictions, print_metrics,
)

MODEL_NAME = 'random_forest'
DISPLAY_NAME = 'Random Forest'

print("=" * 60)
print("МОДЕЛЬ: {}".format(DISPLAY_NAME))
print("=" * 60)

# Загрузка данных
data = load_data()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Обучение
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Модель обучена (100 деревьев)")

# Предсказание
y_pred = model.predict(X_test)

# Оценка
metrics = evaluate_model(y_test, y_pred)
print_metrics(DISPLAY_NAME, metrics)

# Важность признаков
feature_importance = model.feature_importances_
feature_names = data['feature_names']

plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(feature_importance)
plt.barh(
    [feature_names[i] for i in sorted_idx],
    feature_importance[sorted_idx],
    color='forestgreen', edgecolor='black', alpha=0.8
)
plt.xlabel('Важность признака', fontsize=12)
plt.title('Random Forest: важность признаков', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(
    'results/plots/важность_признаков_RF.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Сохранено: results/plots/важность_признаков_RF.png")

# Сохранение
save_results(
    MODEL_NAME, metrics, y_pred,
    extra={'display_name': DISPLAY_NAME, 'type': 'classic'}
)
plot_predictions(DISPLAY_NAME, y_test, y_pred, metrics, color='#00CC96')

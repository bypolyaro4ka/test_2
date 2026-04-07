"""
Утилиты для загрузки данных и сохранения результатов моделей.
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    """Загружает подготовленные данные из results/data/prepared_data.pkl"""
    path = 'results/data/prepared_data.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Данные не найдены. Сначала запустите: python data_preprocessing.py"
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def evaluate_model(y_true, y_pred):
    """Вычисляет метрики качества модели."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        'MSE': round(float(mse), 2),
        'RMSE': round(float(np.sqrt(mse)), 2),
        'MAE': round(float(mean_absolute_error(y_true, y_pred)), 2),
        'R2': round(float(r2_score(y_true, y_pred)), 4),
    }


def save_results(model_name, metrics, y_pred, extra=None):
    """Сохраняет метрики и предсказания модели."""
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)

    # Метрики
    result = {'model': model_name, **metrics}
    if extra:
        result.update(extra)
    with open(f'results/metrics/{model_name}.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Предсказания
    np.save(f'results/predictions/{model_name}.npy', y_pred)

    print("\nСохранено:")
    print("  results/metrics/{}.json".format(model_name))
    print("  results/predictions/{}.npy".format(model_name))


def plot_predictions(model_name, y_true, y_pred, metrics, color='#1f77b4'):
    """Строит график предсказания vs реальность и сохраняет."""
    os.makedirs('results/plots', exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='black',
                linewidth=0.5, color=color, s=40)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Идеальный прогноз')
    plt.xlabel('Реальные значения (МПа)', fontsize=12)
    plt.ylabel('Предсказанные значения (МПа)', fontsize=12)
    plt.title(f'{model_name}\nR² = {metrics["R2"]:.4f}, RMSE = {metrics["RMSE"]:.2f}',
              fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = model_name.replace('/', '_')
    filename = f'results/plots/прогноз_{safe_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {filename}")


def print_metrics(model_name, metrics):
    """Выводит метрики в консоль."""
    print(f"\n{'=' * 50}")
    print(f"Результаты: {model_name}")
    print(f"{'=' * 50}")
    print(f"  MSE  = {metrics['MSE']}")
    print(f"  RMSE = {metrics['RMSE']}")
    print(f"  MAE  = {metrics['MAE']}")
    print(f"  R²   = {metrics['R2']}")

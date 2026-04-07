# Модель: НС с L2-регуляризацией (2 скрытых слоя).
# Замена: Лассо и гребневая регрессия.
# Архитектура: Input(8) -> Dense(128, ReLU, L2) -> Dense(64, ReLU, L2) -> Dense(1)
# Запуск: .venv/bin/python3 models/neural_network/model_nn_regularized.py


# Импортируем все библиотеки
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import matplotlib.pyplot as plt

# Эта тема нужна чтобы TensorFlow не спамил в консоль все логи, а только ошибки
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from utils import (
    load_data, evaluate_model, save_results,
    plot_predictions, print_metrics,
)

# Тут мы фиксируем результат, так как каждый раз при переобучении у нас будет новый результат и этими сидами мы их фиксируем
np.random.seed(42)
tf.random.set_seed(42)

# Ну тут имена модели сколько эпох обучения у модели и сколько данных берется за раз (32)
MODEL_NAME = 'nn_regularized'
DISPLAY_NAME = 'Лассо/гребневая регрессия NN'
EPOCHS = 200
BATCH_SIZE = 32

print("=" * 60)
print("МОДЕЛЬ: {}".format(DISPLAY_NAME))
print("Замена: Лассо и гребневая регрессия")
print("=" * 60)

# Загрузка данных как и в обычных моделях
data = load_data()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Построение модели
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)), # Входной слой, ждем 8 признаков бетона (8 чисел)
    layers.Dense(
        128, activation='relu',
        kernel_regularizer=regularizers.l2(0.001) # Первый скрытый слой (128 нейронов) но с регуляризацией. Она добавляет штрафы за ошибки. Если вес большой (5.0), то его квадрат будет (25.0), что увеличит наши потери, и в этот момент наш кент Адам будет пытаться его уменьшить, и по итогу сделает все веса примерно одинаково маленькими. Без этой регуляризации сеть может запомнить шум в данных, и веса будут расти до огромных значений ну и по итогу приведет нас к плохой модели. С регуляризацией наши веса меньше, модель хорошая и лучше работает на новых данных
    ),
    layers.Dense(
        64, activation='relu', # Тот же принцип
        kernel_regularizer=regularizers.l2(0.001)
    ),
    layers.Dense(1) # Как в прошлой модели показывает нам прочность одним числом для сравнения.
], name='Regularized_NN')

model.compile(optimizer='adam', loss='mse', metrics=['mae']) # Адам это наш кент, он обновляет веса, даптирует скорость обучения отталкиваясь от весов. Получается веса которые редко обновляются получают большой шаг. MSE - это функция потерь, задача минимизировать среднее квадратов разницы между предсказанием и реальностью. MAE тема котооая вообще не влияет на обучение, просто показывает нам логи обучения и дополнительные метрики
model.summary() # Таблица с архитектурой нашей сети, в общем все параметры, названия и тд.

# Обучение
history = model.fit(
    X_train, y_train, # На чем учимся + правильные ответы
    epochs=EPOCHS, batch_size=BATCH_SIZE, # Эпохи и пачки которыми мы берем данные (выше о них)
    validation_split=0.2, verbose=0 # Забираем 20% всех данных для честного теста, что бы сеть их не видела
)
print("Обучение завершено за {} эпох".format(EPOCHS))

# Предсказание. Прогоняем наши 20% которые мы взяли из датасета раньше, только уже без дропов, что бы работали все нейроны. На выходе получаем число.
y_pred = model.predict(X_test, verbose=0).flatten()

# Оценка. Сравниваем показатели с реальными данными и определяем точность модели.
metrics = evaluate_model(y_test, y_pred)
print_metrics(DISPLAY_NAME, metrics)

# График обучения
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Обучение', linewidth=2)
plt.plot(history.history['val_loss'], label='Валидация', linewidth=2)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Потери (MSE)', fontsize=12)
plt.title('{}: динамика обучения'.format(DISPLAY_NAME), fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    'results/plots/обучение_лассо_гребневая_NN.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Сохранено: results/plots/обучение_лассо_гребневая_NN.png")

# Сохранение истории обучения
hist_data = {
    k: [float(v) for v in vals]
    for k, vals in history.history.items()
}
with open('results/metrics/{}_history.json'.format(MODEL_NAME), 'w') as f:
    json.dump(hist_data, f)

# Сохранение результатов
arch = (
    'Input(8) -> Dense(128, ReLU, L2=0.001) '
    '-> Dense(64, ReLU, L2=0.001) -> Dense(1)'
)
save_results(MODEL_NAME, metrics, y_pred, extra={
    'display_name': DISPLAY_NAME,
    'type': 'neural_network',
    'replaces': 'Лассо/гребневая регрессия',
    'architecture': arch,
    'epochs': EPOCHS,
    'params': int(model.count_params()),
})
plot_predictions(DISPLAY_NAME, y_test, y_pred, metrics, color='#FFA15A')

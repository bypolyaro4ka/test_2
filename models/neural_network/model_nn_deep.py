# Модель: Глубокая НС с Dropout (3 скрытых слоя).
# Замена: Random Forest Regressor.
# Архитектура: Input(8) -> Dense(256, ReLU) -> Dropout(0.2) -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(64, ReLU) -> Dense(1)
# Запуск: .venv/bin/python3 models/neural_network/model_nn_deep.py


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
from tensorflow.keras import layers
from utils import (
    load_data, evaluate_model, save_results,
    plot_predictions, print_metrics,
)

# Тут мы фиксируем результат, так как каждый раз при переобучении у нас будет новый результат и этими сидами мы их фиксируем
np.random.seed(42)
tf.random.set_seed(42)

# Ну тут имена модели сколько эпох обучения у модели и сколько данных берется за раз (32)
MODEL_NAME = 'nn_deep'
DISPLAY_NAME = 'Random Forest NN'
EPOCHS = 200
BATCH_SIZE = 32

print("=" * 60)
print("МОДЕЛЬ: {}".format(DISPLAY_NAME))
print("Замена: Random Forest Regressor")
print("=" * 60)

# Загрузка данных как и в обычных моделях
data = load_data()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Построение модели
model = keras.Sequential([ # Создаем сеть из слоев и они идут последовательно как в коде
    layers.Input(shape=(X_train.shape[1],)), # Входной слой, ждем массив из 8 чисел (признаков бетона)
    layers.Dense(256, activation='relu'), # Первый слой 256 нейронов. Ну тут долго, в общем берем наши 8 чисел, дальше умножаем каждое на свой вес (8 весов по итогу), прибавляем к этой теме смещение ну и на конец прогоняем через ReLU, если результат >0 то оставляем, если меньше то ставим там 0. Получается 256 нейронов делают параллельно 256 вычислений для поиска своих закономерностей
    layers.Dropout(0.2), # Отключаем 20% нейронов, что бы сеть полагалась не только на лучшие нейроны, а на все в критической ситуации. В обучении будут работать все нейроны
    layers.Dense(128, activation='relu'), # Примерно по тому же принципу, что и выше только уменьшаем количество нейронов для выявления ключевых паттернов
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1) # Тут входной слой с 1 нейроном з активации. Берем 64 значения умножаем на их веса суммируем и выдаем одно число, это число и есть наша прочность.
], name='Deep_NN_Dropout')

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
    'results/plots/обучение_random_forest_NN.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Сохранено: results/plots/обучение_random_forest_NN.png")

# Сохранение истории обучения
hist_data = {
    k: [float(v) for v in vals]
    for k, vals in history.history.items()
}
with open('results/metrics/{}_history.json'.format(MODEL_NAME), 'w') as f:
    json.dump(hist_data, f)

# Сохранение результатов
arch = (
    'Input(8) -> Dense(256, ReLU) -> Dropout(0.2) '
    '-> Dense(128, ReLU) -> Dropout(0.2) '
    '-> Dense(64, ReLU) -> Dense(1)'
)
save_results(MODEL_NAME, metrics, y_pred, extra={
    'display_name': DISPLAY_NAME,
    'type': 'neural_network',
    'replaces': 'Random Forest Regressor',
    'architecture': arch,
    'epochs': EPOCHS,
    'params': int(model.count_params()),
})
plot_predictions(DISPLAY_NAME, y_test, y_pred, metrics, color='#00CC96')

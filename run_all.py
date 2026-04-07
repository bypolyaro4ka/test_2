"""
Запуск всех скриптов по порядку.

После запуска можно открыть дашборд:
  .venv/bin/python3 dashboard.py

Запуск: .venv/bin/python3 run_all.py
"""

import subprocess
import sys
import time

PYTHON = sys.executable

scripts = [
    ('Предобработка данных',       'data_preprocessing.py'),
    ('Линейная регрессия',         'models/classic/model_linear.py'),
    ('Лассо/гребневая регрессия',  'models/classic/model_lasso_ridge.py'),
    ('Random Forest',              'models/classic/model_random_forest.py'),
    ('Линейная регрессия NN',      'models/neural_network/model_nn_simple.py'),
    ('Лассо/гребневая NN',         'models/neural_network/model_nn_regularized.py'),
    ('Random Forest NN',           'models/neural_network/model_nn_deep.py'),
]

print("=" * 60)
print("ЗАПУСК ВСЕХ МОДЕЛЕЙ")
print("=" * 60)

total_start = time.time()

for i, (name, script) in enumerate(scripts, 1):
    print("\n" + "-" * 60)
    print("[{}/{}] {} ({})".format(i, len(scripts), name, script))
    print("-" * 60)

    start = time.time()
    result = subprocess.run([PYTHON, script], capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print("\n❌ ОШИБКА в {} (код {})".format(script, result.returncode))
        sys.exit(1)

    print("\n✅ {} — завершено за {:.1f} сек".format(name, elapsed))

total_elapsed = time.time() - total_start

print("\n" + "=" * 60)
print("ВСЕ МОДЕЛИ ЗАВЕРШЕНЫ за {:.1f} сек".format(total_elapsed))
print("=" * 60)
print("\nДля запуска дашборда:")
print("  .venv/bin/python3 dashboard.py")

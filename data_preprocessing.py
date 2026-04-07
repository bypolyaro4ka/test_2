"""
Загрузка, предобработка и исследовательский анализ данных (EDA).

Загружает датасет Concrete Compressive Strength (UCI, 1030 записей),
выполняет EDA с визуализацией и сохраняет подготовленные данные
в results/data/ для использования другими скриптами.

Запуск: .venv/bin/python3 data_preprocessing.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

np.random.seed(42)

os.makedirs('results/data', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

# -- 1. Загрузка данных --
print("=" * 60)
print("ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 60)

dataset = fetch_ucirepo(id=165)
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

columns_ru = {
    'Cement': 'Цемент (кг/м³)',
    'Blast Furnace Slag': 'Шлак (кг/м³)',
    'Fly Ash': 'Зола (кг/м³)',
    'Water': 'Вода (кг/м³)',
    'Superplasticizer': 'Добавка (кг/м³)',
    'Coarse Aggregate': 'Крупный щебень (кг/м³)',
    'Fine Aggregate': 'Мелкий щебень (кг/м³)',
    'Age': 'Возраст (дни)',
    'Concrete compressive strength': 'Прочность бетона (МПа)'
}
df.columns = [columns_ru.get(c, c) for c in df.columns]

rows, cols = df.shape
print("Размер датасета: {} строк, {} столбцов".format(rows, cols))
print("\nПервые 5 строк:")
print(df.head())
print("\nСтатистика:")
print(df.describe().round(2))
print("\nПропуски:")
print(df.isnull().sum())

# -- 2. EDA: визуализация --
print("\n" + "=" * 60)
print("ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ (EDA)")
print("=" * 60)

# Корреляционная матрица
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Корреляционная матрица признаков бетона', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(
    'results/plots/01_корреляционная_матрица.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Сохранено: results/plots/01_корреляционная_матрица.png")

# Рассеяние: цемент vs прочность
plt.figure(figsize=(10, 6))
plt.scatter(df['Цемент (кг/м³)'], df['Прочность бетона (МПа)'],
            alpha=0.5, edgecolors='black', linewidth=0.5)
plt.xlabel('Цемент (кг/м³)', fontsize=12)
plt.ylabel('Прочность бетона (МПа)', fontsize=12)
plt.title('Зависимость прочности от количества цемента', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    'results/plots/02_рассеяние_цемент_прочность.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Сохранено: results/plots/02_рассеяние_цемент_прочность.png")

# Рассеяние: возраст vs прочность
plt.figure(figsize=(10, 6))
plt.scatter(df['Возраст (дни)'], df['Прочность бетона (МПа)'],
            alpha=0.5, edgecolors='black', linewidth=0.5, color='green')
plt.xlabel('Возраст бетона (дни)', fontsize=12)
plt.ylabel('Прочность бетона (МПа)', fontsize=12)
plt.title('Зависимость прочности от возраста бетона', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    'results/plots/03_рассеяние_возраст_прочность.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Сохранено: results/plots/03_рассеяние_возраст_прочность.png")

# Распределение целевой переменной
plt.figure(figsize=(10, 6))
plt.hist(df['Прочность бетона (МПа)'], bins=30, edgecolor='black',
         alpha=0.7, color='steelblue')
plt.xlabel('Прочность бетона (МПа)', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.title('Распределение прочности бетона на сжатие', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    'results/plots/04_распределение_прочности.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Сохранено: results/plots/04_распределение_прочности.png")

# Парный график
sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15})
plt.suptitle('Парный график всех признаков', y=1.01, fontsize=14)
plt.savefig(
    'results/plots/05_парный_график.png',
    dpi=100, bbox_inches='tight'
)
plt.close()
print("Сохранено: results/plots/05_парный_график.png")

# -- 3. Подготовка данных для моделей --
print("\n" + "=" * 60)
print("ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛЕЙ")
print("=" * 60)

X = df.drop('Прочность бетона (МПа)', axis=1)
y = df['Прочность бетона (МПа)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Обучающая выборка: {} записей".format(X_train.shape[0]))
print("Тестовая выборка: {} записей".format(X_test.shape[0]))

# Сохраняем данные
data = {
    'X_train': X_train_scaled,
    'X_test': X_test_scaled,
    'y_train': y_train.values,
    'y_test': y_test.values,
    'feature_names': list(X.columns),
    'scaler': scaler,
}
with open('results/data/prepared_data.pkl', 'wb') as f:
    pickle.dump(data, f)

df.to_csv('results/data/dataset.csv', index=False)

print("\nСохранено:")
print("  results/data/prepared_data.pkl — подготовленные данные")
print("  results/data/dataset.csv — полный датасет")
print("\nПредобработка завершена!")
